"""
inverse_kinematic.py — Cinemática inversa como QP para el LAK Obedience.

Resuelve en cada paso:
    min  w_com ||J_com dq - e_com||²
       + w_torso ||J_torso dq - e_torso||²
       + w_mf ||J_mf dq - e_mf||²
       + mu ||dq||²
    s.t. J_ff dq = e_ff    (pie de soporte fijo)

Dependencias: pinocchio, qpsolvers[osqp]
"""

import typing
from dataclasses import dataclass
from typing import Any

import numpy as np
import pinocchio as pin
from qpsolvers import solve_qp


@dataclass
class InvKinSolverParams:
    """
    Parámetros del QP de cinemática inversa.

    Para el LAK Obedience, valores iniciales recomendados:
        w_com    = 10.0   (seguir target del CoM)
        w_torso  = 5.0    (mantener tronco recto)
        w_mf     = 50.0   (seguir trayectoria del pie swing)
        mu       = 1e-4   (regularización/damping)
        locked_joints = []  (si usas modelo reducido, ya no hay joints de cabeza)

    Attributes
    ----------
    fixed_foot_frame : int
        Pinocchio frame id of the stance foot (hard equality).
    moving_foot_frame : int
        Pinocchio frame id of the swing foot (soft task).
    torso_frame : int
        Pinocchio frame id for torso orientation task (angular only).
    model : pin.Model
        Pinocchio kinematic model.
    data : Any
        Pinocchio data buffer associated with `model`.
    w_torso : float
        Weight of torso angular task in the QP cost.
    w_com : float
        Weight of CoM task in the QP cost.
    w_mf : float
        Weight of moving-foot 6D task in the QP cost.
    mu : float
        Damping on joint velocities in the QP (Tikhonov term).
    dt : float
        Time step used by caller; not used here directly but kept for API symmetry.
    locked_joints : list[int] | None
        Optional list of joints or velocity indices to lock.
        Each element can be:
          - a Pinocchio joint id j in [0, model.njoints), which locks its velocity span
          - or a direct velocity index v in [0, model.nv)
    """

    fixed_foot_frame: int
    moving_foot_frame: int
    torso_frame: int
    model: pin.Model
    data: Any
    w_torso: float
    w_com: float
    w_mf: float
    mu: float
    dt: float
    locked_joints: typing.Optional[typing.List[int]] = None


def _se3_task_error_and_jacobian(model, data, q, frame_id, M_des):
    """
    Calcula el error de pose 6D (log SE3) y el Jacobiano de tarea para un frame.
    
    Parameters
    ----------
    model : pin.Model
        Pinocchio model.
    data : pin.Data
        Pinocchio data (assumed up to date for frame placements if needed).
    q : ndarray, shape (nq,)
        Configuration vector.
    frame_id : int
        Target frame id in `model`.
    M_des : pin.SE3
        Desired frame pose in world.

    Returns
    -------
    e6 : ndarray, shape (6,)
        Right-invariant pose residual in the LOCAL frame.
        Order: angular (rx, ry, rz), linear (vx, vy, vz).
    Jtask : ndarray, shape (6, nv)
        Task Jacobian that maps generalized velocity `dq` to residual rate.

    Notes
    -----
    Uses LOCAL frame convention and Pinocchio's `Jlog6` to map spatial velocity
    to se(3) log-space.

    """
    # Pose del frame i en el world; convención del frame LOCAL (diferenciación por la derecha)
    oMi = data.oMf[frame_id]
    iMd = oMi.actInv(M_des)
    e6 = pin.log(iMd).vector

    # Calcula el Jacobiano del frame i en el frame de referencial local
    Jb = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)

    # Right Jacobian of the log map (Pinocchio’s Jlog6)
    Jl = pin.Jlog6(iMd)
    return e6, Jl @ Jb # minus sign per right-invariant residual


def _joint_vel_span(j, model):
    """
    Return the velocity-index span for joint `j`.

    Parameters
    ----------
    j : int
        Pinocchio joint id.
    model : pin.Model
        Model containing `idx_v` mapping.

    Returns
    -------
    range
        Range of velocity indices covered by joint `j`.
    """
    i = model.idx_v[j]
    nvj = (model.idx_v[j + 1] - i) if j + 1 < model.njoints else (model.nv - i)
    return range(i, i + nvj)


def solve_inverse_kinematics(q, com_target, oMf_fixed_foot,
                              oMf_moving_foot, oMf_torso,
                              params: InvKinSolverParams):
    """
    Un paso de IK via QP con restricción hard de pie de soporte.

    Parameters
    ----------
    q : ndarray, shape (nq,)
        configuración actual (nq,)
    com_target : ndarray, shape (3,)
        posición deseada del CoM (3,)
    oMf_fixed_foot: pin.SE3
        pose deseada pie de soporte (SE3)
    oMf_moving_foot: pin.SE3
        pose deseada pie swing (SE3)
    oMf_torso     : pin.SE3
        pose deseada torso -solo orientación- (SE3)
    params        : InvKinSolverParams
        Pesos, modelos/data, damping y articulaciones opcionales bloqueadas

    Returns
    -------
    q_next : ndarray, shape (nq,)
        nueva configuración, Configuración integrada `integrate(modelo, q, dq)`.
    dq : ndarray, shape (nv,)
        velocidades generalizadas solución (ceros para articulaciones bloqueadas)
    """
    model, data = params.model, params.data
    nv = model.nv

    # Índices de velocidad activos (excluye joints bloqueados)
    locked_v: set = set()
    if params.locked_joints:
        for j in params.locked_joints:
            if 0 <= j < model.njoints:
                i0 = model.idx_vs[j]
                i1 = model.idx_vs[j+1] if j+1 < model.njoints else nv
                locked_v.update(range(i0, i1))
            elif 0 <= j < nv:
                locked_v.add(j)
    active = np.array(sorted(set(range(nv)) - locked_v), dtype=int)
    nav = active.size

    def red(M):
        return M[:, active]

    # Tarea CoM
    pin.computeCentroidalMap(model, data, q)
    com   = pin.centerOfMass(model, data, q)
    Jcom  = pin.jacobianCenterOfMass(model, data, q)
    e_com = com_target - com

    # Pie fijo (restricción de igualdad)
    e_ff, J_ff = _se3_task_error_and_jacobian(
        model, data, q, params.fixed_foot_frame, oMf_fixed_foot)

    # Pie swing (soft)
    e_mf, J_mf = _se3_task_error_and_jacobian(
        model, data, q, params.moving_foot_frame, oMf_moving_foot)

    # Torso — solo parte angular (filas 3:6 en LOCAL)
    e6t, J6t = _se3_task_error_and_jacobian(
        model, data, q, params.torso_frame, oMf_torso)
    S = np.zeros((3, 6))
    S[0, 3] = S[1, 4] = S[2, 5] = 1.0
    e_torso  = S @ e6t
    J_torso  = S @ J6t

    # QP matrices
    Jc_r = red(Jcom);   Jff_r = red(J_ff)
    Jm_r = red(J_mf);   Jt_r  = red(J_torso)

    H = (Jc_r.T @ (np.eye(3)*params.w_com)   @ Jc_r
       + Jt_r.T  @ (np.eye(3)*params.w_torso) @ Jt_r
       + Jm_r.T  @ (np.eye(6)*params.w_mf)    @ Jm_r
       + np.eye(nav)*params.mu)
    g = (-Jc_r.T @ (np.eye(3)*params.w_com)   @ e_com
       + -Jt_r.T  @ (np.eye(3)*params.w_torso) @ e_torso
       + -Jm_r.T  @ (np.eye(6)*params.w_mf)    @ e_mf)

    H = 0.5 * (H + H.T)

    dq_r = solve_qp(P=H, q=g, A=Jff_r, b=e_ff, solver="osqp")

    dq = np.zeros(nv)
    dq[active] = dq_r
    q_next = pin.integrate(model, q, dq)

    return q_next, dq