"""
simulation.py — Interfaz de simulación con MuJoCo para el LAK Obedience.

Reemplaza la clase Simulator basada en PyBullet del repo original.
MuJoCo se elige porque:
  - Mejor física de contacto para robots ligeros (2.28 kg)
  - API Python simple y directa
  - Más rápido que Gazebo para iterar en el controlador

IMPORTANTE: El URDF se convierte a MJCF automáticamente por MuJoCo
al cargarlo. No necesitas hacer conversión manual.

Uso típico:
    sim = MujocoSimulator(urdf_path, lak_model)
    sim.reset(q_init)
    for k in range(N):
        sim.apply_position_control(q_des)
        sim.step()
        q = sim.get_q()
        rf_force, lf_force = sim.get_contact_forces()
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import numpy as np
import mujoco
import pinocchio as pin


def _patch_urdf_for_mujoco(urdf_path: str) -> str:
    """
    Prepara el URDF para MuJoCo:
    1. Corrige la geometry vacía de trunk_assembly_1
    2. Ajusta rutas de meshes a rutas absolutas (MuJoCo necesita paths absolutos)

    Returns ruta a URDF temporal corregido.
    """
    # urdf está en:  .../lak_description/urdf/lak_obedience.urdf
    # package root:  .../lak_description/
    pkg_root = str(Path(urdf_path).parent.parent)  # sube de urdf/ a lak_description/

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Corregir geometry vacía
    for link in root.findall("link"):
        if link.get("name") == "trunk_assembly_1":
            for coll in link.findall("collision"):
                geo = coll.find("geometry")
                if geo is not None and len(geo) == 0:
                    box = ET.SubElement(geo, "box")
                    box.set("size", "0.12 0.08 0.14")

    # Reemplazar package:// por rutas absolutas
    # package://lak_description/meshes/visual/foo.stl
    #   → <pkg_root>/meshes/visual/foo.stl
    for elem in root.iter():
        for attr in ["filename"]:
            val = elem.get(attr, "")
            if val.startswith("package://"):
                # quita el prefijo "package://<pkg_name>/"
                # sin importar cómo se llame el paquete
                parts = val[len("package://"):].split("/", 1)
                rel = parts[1] if len(parts) > 1 else ""
                abs_path = os.path.join(pkg_root, rel)
                elem.set(attr, abs_path)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="w", encoding="utf-8"
    )
    tree.write(tmp.name, encoding="unicode", xml_declaration=True)
    tmp.close()
    return tmp.name


class MujocoSimulator:
    """
    Interfaz de simulación MuJoCo para el robot LAK Obedience.

    Mantiene sincronización entre los índices de joints de Pinocchio
    y los de MuJoCo (que pueden tener orden diferente).

    Parameters
    ----------
    urdf_path : str | Path
        Ruta al lak_obedience.urdf
    lak_model : LakObedience
        Modelo Pinocchio ya inicializado (para mapeo de joints)
    dt : float
        Timestep del simulador (default 1/500 s)
    render : bool
        Si True, abre visor 3D interactivo de MuJoCo
    """

    def __init__(self, urdf_path: str | Path, lak_model,
                 dt: float = 1.0 / 500.0, render: bool = False):
        self.dt = dt
        self.lak = lak_model
        self.render = render

        # Prepara URDF y carga modelo MuJoCo
        patched = _patch_urdf_for_mujoco(str(urdf_path))
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(patched)
        finally:
            os.unlink(patched)

        self.mj_model.opt.timestep = dt
        self.mj_data = mujoco.MjData(self.mj_model)

        # Visor (opcional)
        self._viewer = None
        if render:
            self._init_viewer()

        # Construir mapa de joints Pinocchio → MuJoCo
        # MuJoCo usa qpos[7:] para joints (los primeros 7 son free-flyer)
        # Pinocchio también usa q[7:] pero puede tener orden diferente
        self._pin_to_mj_idx = self._build_joint_map()

        # Nombres de links de pies para detección de contacto
        self._lf_geom_ids = self._find_geom_ids(["foot_assembly_2"])
        self._rf_geom_ids = self._find_geom_ids(["foot_assembly_1"])
        self._ground_geom_id = 0  # el suelo es geom 0 en MuJoCo por defecto

    def _build_joint_map(self) -> dict:
        """
        Construye mapa {pin_q_idx: mj_qpos_idx} para joints actuados.
        Excluye el free-flyer (primeros 7 en ambos).
        """
        pin_model = self.lak.model
        mapping = {}

        for j in range(1, pin_model.njoints):
            jname = pin_model.names[j]
            if pin_model.joints[j].nq != 1:
                continue
            try:
                mj_jid = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname
                )
                if mj_jid >= 0:
                    pin_qidx = pin_model.joints[j].idx_q
                    mj_qidx  = self.mj_model.jnt_qposadr[mj_jid]
                    mapping[pin_qidx] = mj_qidx
            except Exception:
                pass

        return mapping

    def _find_geom_ids(self, link_names: list) -> list:
        """Encuentra los IDs de geoms de MuJoCo para una lista de link names."""
        ids = []
        for name in link_names:
            gid = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if gid >= 0:
                ids.append(gid)
            # también probar con sufijo _collision
            gid2 = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                name + "_collision"
            )
            if gid2 >= 0 and gid2 not in ids:
                ids.append(gid2)
        return ids

    def _init_viewer(self):
        """Inicializa el visor pasivo de MuJoCo."""
        try:
            import mujoco.viewer
            self._viewer_handle = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data
            )
        except Exception as e:
            print(f"[Simulator] Visor no disponible: {e}")
            self._viewer_handle = None

    def reset(self, q_pin: np.ndarray):
        """
        Resetea el simulador a una configuración Pinocchio q.

        Parameters
        ----------
        q_pin : ndarray (nq,)
            Configuración en formato Pinocchio:
            q[:3]=posición base, q[3:7]=quaternión xyzw, q[7:]=joints
        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # MuJoCo carga el URDF sin free-flyer (base fija en el mundo).
        # Solo copiamos los joints actuados desde el vector q de Pinocchio.
        # q_pin[:7] = base pose (ignorado aquí, MuJoCo fija la base)
        # q_pin[7:] = joints
        for pin_idx, mj_idx in self._pin_to_mj_idx.items():
            self.mj_data.qpos[mj_idx] = q_pin[pin_idx]

        # Velocidades a cero
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def apply_position_control(self, q_des_pin: np.ndarray,
                                kp: float = 50.0, kd: float = 2.0):
        """
        Control de posición articular mediante servo ideal (qpos directo).

        El URDF no tiene transmisiones/actuadores definidos, por lo que
        usamos escritura directa en qpos — equivalente a un servo ideal.
        Para hardware real, reemplazar por comandos a los servos vía ROS2.

        MuJoCo free-flyer layout:
          qpos[0:3]  = base position
          qpos[3:7]  = base quaternion (wxyz)  → 7 elementos
          qpos[7:]   = joints
          qvel[0:6]  = base velocity (6 dof)
          qvel[6:]   = joint velocities
        """
        for pin_idx, mj_idx in self._pin_to_mj_idx.items():
            q_actual = self.mj_data.qpos[mj_idx]
            q_target = q_des_pin[pin_idx]
            error    = q_target - q_actual

            # Servo ideal: escribir directamente la posición deseada
            self.mj_data.qpos[mj_idx] = q_target

            # En base fija: qpos[i] y qvel[i] tienen el mismo índice
            v_idx = mj_idx
            if 0 <= v_idx < self.mj_model.nv:
                self.mj_data.qvel[v_idx] = kp * error

    def step(self):
        """Avanza la simulación un timestep."""
        mujoco.mj_step(self.mj_model, self.mj_data)
        if self.render and hasattr(self, '_viewer_handle') and self._viewer_handle:
            self._viewer_handle.sync()

    def get_q(self) -> np.ndarray:
        """
        Retorna la configuración actual en formato Pinocchio (nq,).

        Convierte el free-flyer de MuJoCo (wxyz) a Pinocchio (xyzw).
        """
        # Devolvemos un vector q en formato Pinocchio (nq=17 con free-flyer).
        # Base se mantiene en neutral (MuJoCo fija la base).
        import pinocchio as pin
        q = pin.neutral(self.lak.model)

        # Copiar joints desde MuJoCo
        for pin_idx, mj_idx in self._pin_to_mj_idx.items():
            q[pin_idx] = self.mj_data.qpos[mj_idx]

        return q

    def get_contact_forces(self) -> Tuple[float, float]:
        """
        Retorna la fuerza de contacto vertical (Fz) de cada pie.

        Returns
        -------
        (rf_force, lf_force) : float, float
            Fuerza vertical en Newton. > 0 = pie en contacto con suelo.
        """
        rf_force = 0.0
        lf_force = 0.0

        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            # Extraer fuerza de contacto
            force = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
            fz = abs(force[2])

            if g1 in self._rf_geom_ids or g2 in self._rf_geom_ids:
                rf_force += fz
            if g1 in self._lf_geom_ids or g2 in self._lf_geom_ids:
                lf_force += fz

        return rf_force, lf_force

    def get_com_position(self) -> np.ndarray:
        """
        Retorna la posición del CoM calculada por MuJoCo (subtree_com).
        """
        # subtree_com[0] = CoM del sistema completo
        return self.mj_data.subtree_com[0].copy()

    def get_zmp(self) -> np.ndarray:
        """
        Estima el ZMP desde los contactos activos.

        Returns
        -------
        ndarray (3,) o None si no hay contacto.
        """
        F_total = np.zeros(3)
        M_total = np.zeros(3)

        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, force)
            f = force[:3]
            pos = contact.pos
            F_total += f
            M_total += np.cross(pos, f)

        if abs(F_total[2]) < 1e-6:
            return None

        px = -M_total[1] / F_total[2]
        py =  M_total[0] / F_total[2]
        return np.array([px, py, 0.0])