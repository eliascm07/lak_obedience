"""
walk.py — Script principal de marcha bípeda para el LAK Obedience.

Cómo correrlo:
    python3 walk.py --urdf /ruta/a/lak_obedience.urdf
    python3 walk.py --urdf /ruta/a/lak_obedience.urdf --render
    python3 walk.py --urdf /ruta/a/lak_obedience.urdf --info-only

Estructura del loop:
    1. Cargar modelo Pinocchio (LakObedience)
    2. Calcular postura inicial + parámetros calibrados del robot
    3. Pre-calcular trayectorias de pies y referencia de ZMP (offline)
    4. Sintetizar ganancias del controlador preview (offline, 1 sola vez)
    5. Loop principal:
          a. Leer q del simulador
          b. update_control → target CoM
          c. solve_inverse_kinematics → q_des
          d. Aplicar q_des al simulador
          e. Avanzar simulación
"""

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin

from model             import LakObedience, q_from_base_and_joints
from preview_control   import (
    PreviewControllerParams,
    compute_preview_control_matrices,
    update_control,
    compute_zmp_ref,
    cubic_spline_interpolation,
)
from foot              import (
    compute_steps_sequence,
    compute_feet_trajectories,
    BezierCurveFootPathGenerator,
)
from inverse_kinematic import InvKinSolverParams, solve_inverse_kinematics
from simulation_mujoco_2        import MujocoSimulator


# ─────────────────────────────────────────────────────────────────────
# Parámetros de marcha — ajustados para el LAK Obedience
# (robot pequeño: 2.28 kg, piernas de 15 cm, CoM a ~0.22 m)
# ─────────────────────────────────────────────────────────────────────
class WalkParams:
    dt           = 1.0 / 500.0   # timestep simulación [s]
    t_ss         = 0.8            # duración single support [s]
    t_ds         = 0.3            # duración double support [s]
    t_init       = 2.0            # tiempo inicialización [s]
    t_end        = 1.0            # tiempo final [s]
    n_steps      = 6              # número de pasos
    l_stride     = 0.03           # longitud de paso [m] — pequeño para empezar
    foot_height  = 0.015          # altura máxima del pie en swing [m]
    t_preview    = 1.6            # horizonte de preview [s]
    zc           = 0.22           # altura del CoM [m] — calibrar con print_info()

    # Pesos de la cinemática inversa
    w_com   = 10.0
    w_torso =  5.0
    w_mf    = 50.0
    mu      = 1e-4


def snap_feet_to_ground(oMf_lf: pin.SE3, oMf_rf: pin.SE3) -> tuple:
    """Proyecta ambos pies al plano z=0 manteniendo orientación."""
    def snap(oMf):
        R = oMf.rotation
        p = oMf.translation.copy()
        p[2] = 0.0
        return pin.SE3(R, p)
    return snap(oMf_lf), snap(oMf_rf)


def compute_base_from_foot(model, data, q, foot_frame_id, oMf_target):
    """
    Calcula la pose del base_link necesaria para que el pie esté
    en la pose objetivo: oMb_new = oMf_target * (bMf)^-1
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p_b = q[:3]; q_b = q[3:7]
    oMb_cur = pin.SE3(pin.Quaternion(q_b).toRotationMatrix(), p_b)
    bMf = oMb_cur.inverse() * data.oMf[foot_frame_id]
    return oMf_target * bMf.inverse()


def main():
    parser = argparse.ArgumentParser(description="LAK Obedience — Marcha bípeda LIPM")
    parser.add_argument("--urdf", type=Path, required=True,
                        help="Ruta al lak_obedience.urdf")
    parser.add_argument("--render", action="store_true",
                        help="Abrir visor 3D de MuJoCo")
    parser.add_argument("--info-only", action="store_true",
                        help="Solo mostrar info del modelo y salir")
    args = parser.parse_args()

    params = WalkParams()

    # ── 1. Cargar modelo ──────────────────────────────────────────────
    print("Cargando modelo LAK Obedience...")
    lak = LakObedience(args.urdf, reduced=True)
    lak.print_info()

    if args.info_only:
        return

    # ── 2. Postura inicial ────────────────────────────────────────────
    q_init = lak.set_and_get_default_pose()

    # Pies en z=0 y base ajustada
    oMf_lf0 = lak.data.oMf[lak.left_foot_id].copy()
    oMf_rf0 = lak.data.oMf[lak.right_foot_id].copy()
    oMf_lf_tgt, oMf_rf_tgt = snap_feet_to_ground(oMf_lf0, oMf_rf0)
    oMf_torso = lak.data.oMf[lak.torso_id].copy()

    oMb_init = compute_base_from_foot(
        lak.model, lak.data, q_init, lak.left_foot_id, oMf_lf_tgt
    )
    q_init = q_from_base_and_joints(q_init, oMb_init)
    pin.forwardKinematics(lak.model, lak.data, q_init)
    pin.updateFramePlacements(lak.model, lak.data)

    # Target inicial del CoM: punto medio entre pies a altura zc
    feet_mid = 0.5 * (oMf_lf_tgt.translation + oMf_rf_tgt.translation)
    com_initial_target = np.array([feet_mid[0], feet_mid[1], params.zc])

    # ── 3. IK inicial (una sola iteración para alinear) ───────────────
    ik_params = InvKinSolverParams(
        fixed_foot_frame  = lak.left_foot_id,
        moving_foot_frame = lak.right_foot_id,
        torso_frame       = lak.torso_id,
        model             = lak.model,
        data              = lak.data,
        w_com             = params.w_com,
        w_torso           = params.w_torso,
        w_mf              = params.w_mf,
        mu                = params.mu,
        dt                = params.dt,
        locked_joints     = lak.get_locked_joints_for_ik(),
    )
    q_init, _ = solve_inverse_kinematics(
        q_init, com_initial_target, oMf_lf_tgt, oMf_rf_tgt, oMf_torso, ik_params
    )

    # ── 4. Inicializar simulador ───────────────────────────────────────
    print("Inicializando simulador MuJoCo...")
    sim = MujocoSimulator(args.urdf, lak, dt=params.dt, render=args.render)
    sim.reset(q_init)

    # ── 5. Pre-calcular trayectorias de pies y ZMP ─────────────────────
    lf_initial_pose = oMf_lf_tgt.translation.copy()
    rf_initial_pose = oMf_rf_tgt.translation.copy()

    steps_pose, steps_ids = compute_steps_sequence(
        rf_initial_pose = rf_initial_pose,
        lf_initial_pose = lf_initial_pose,
        n_steps         = params.n_steps,
        l_stride        = params.l_stride,
    )

    t_vec, lf_path, rf_path, phases = compute_feet_trajectories(
        rf_initial_pose = rf_initial_pose,
        lf_initial_pose = lf_initial_pose,
        n_steps         = params.n_steps,
        steps_pose      = steps_pose,
        t_ss            = params.t_ss,
        t_ds            = params.t_ds,
        t_init          = params.t_init,
        t_final         = params.t_end,
        dt              = params.dt,
        traj_generator  = BezierCurveFootPathGenerator(params.foot_height),
    )

    # Referencia completa de ZMP
    zmp_ref = compute_zmp_ref(
        t                = t_vec,
        com_initial_pose = com_initial_target[:2],
        steps            = steps_pose[:, :2],
        ss_t             = params.t_ss,
        ds_t             = params.t_ds,
        t_init           = params.t_init,
        t_final          = params.t_end,
        interp_fn        = cubic_spline_interpolation,
    )

    # ── 6. Sintetizar controlador preview ─────────────────────────────
    n_preview = int(round(params.t_preview / params.dt))
    ctrl_params = PreviewControllerParams(
        zc             = params.zc,
        g              = 9.81,
        Qe             = np.array([[1.0]]),
        Qx             = np.zeros((3, 3)),
        R              = np.array([[1e-6]]),
        n_preview_steps= n_preview,
    )
    ctrler_mat = compute_preview_control_matrices(ctrl_params, params.dt)

    # Padear ZMP para que el horizonte siempre tenga suficientes muestras
    zmp_padded = np.vstack([
        zmp_ref,
        np.repeat(zmp_ref[-1:], n_preview, axis=0)
    ])

    # Estado aumentado inicial del controlador [e_int, pos, vel, acc]
    x_k = np.array([0.0, com_initial_target[0], 0.0, 0.0])
    y_k = np.array([0.0, com_initial_target[1], 0.0, 0.0])

    print(f"Iniciando marcha: {params.n_steps} pasos, stride={params.l_stride:.3f}m")
    print(f"Total de timesteps: {len(phases)}")
    print("─" * 55)

    # ── 7. Loop principal ─────────────────────────────────────────────
    for k, _ in enumerate(phases[:-2]):

        # a) Leer estado del simulador → actualizar modelo Pinocchio
        q_cur = sim.get_q()
        pin.forwardKinematics(lak.model, lak.data, q_cur)
        pin.updateFramePlacements(lak.model, lak.data)

        # b) Control preview → posición deseada del CoM
        zmp_horizon = zmp_padded[k+1 : k+n_preview]
        _, x_k, y_k = update_control(
            ctrler_mat, zmp_padded[k], zmp_horizon,
            x_k.copy(), y_k.copy()
        )
        com_target = np.array([x_k[1], y_k[1], params.zc])

        # c) IK: determinar pie fijo/swing según la fase actual
        if phases[k] < 0.0:
            # Fase izquierda en swing
            ik_params.fixed_foot_frame  = lak.right_foot_id
            ik_params.moving_foot_frame = lak.left_foot_id
            oMf_swing = pin.SE3(oMf_lf_tgt.rotation, lf_path[k])
            q_des, _ = solve_inverse_kinematics(
                q_cur, com_target, oMf_rf_tgt, oMf_swing, oMf_torso, ik_params
            )
            # Actualizar pose objetivo del pie izq cuando termina el swing
            if k + 1 < len(phases) and phases[k+1] > 0.0:
                oMf_lf_tgt = pin.SE3(oMf_lf_tgt.rotation, lf_path[k+1])
        else:
            # Fase derecha en swing
            ik_params.fixed_foot_frame  = lak.left_foot_id
            ik_params.moving_foot_frame = lak.right_foot_id
            oMf_swing = pin.SE3(oMf_rf_tgt.rotation, rf_path[k])
            q_des, _ = solve_inverse_kinematics(
                q_cur, com_target, oMf_lf_tgt, oMf_swing, oMf_torso, ik_params
            )
            if k + 1 < len(phases) and phases[k+1] < 0.0:
                oMf_rf_tgt = pin.SE3(oMf_rf_tgt.rotation, rf_path[k+1])

        # d) Aplicar al simulador y avanzar
        sim.apply_position_control(q_des)
        sim.step()

        # Log periódico
        if k % 500 == 0:
            com_real = sim.get_com_position()
            rf_f, lf_f = sim.get_contact_forces()
            print(f"  k={k:5d}  CoM_ref=({com_target[0]:.3f},{com_target[1]:.3f})"
                  f"  CoM_real=z{com_real[2]:.3f}"
                  f"  Frf={rf_f:.1f}N  Flf={lf_f:.1f}N")

    print("─" * 55)
    print("Marcha completada.")

    # Loop infinito para visualización si hay render activo
    if args.render:
        print("Presiona Ctrl+C para salir del visor.")
        try:
            while True:
                sim.step()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()