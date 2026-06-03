"""
model.py — Modelo cinemático del robot LAK Obedience usando Pinocchio.

Reemplaza la clase Talos del repo original. Mantiene la misma interfaz
para que inverse_kinematic.py y el script principal funcionen sin cambios.

Datos extraídos del URDF (pose neutral):
  CoM z=0.247 m  -> zc=0.22 m en marcha (rodillas flexionadas)
  Pie izq: foot_assembly_2  |  Pie der: foot_assembly_1
  Masa total: ~2.28 kg  |  nq=23, nv=22 (modelo completo con free-flyer)

Joints de locomoción activos:
  dof_left_hip_yaw/roll/pitch, dof_left_knee, dof_left_ankle
  dof_right_hip_yaw/roll/pitch, dof_right_knee, dof_right_ankle

Joints bloqueados (no participan en marcha):
  dof_neck_pitch, dof_head_pitch, dof_head_yaw, dof_head_roll
  fix_left_antenna, fix_right_antenna
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin


# IDs de joints a bloquear al construir modelo reducido
# (cabeza, cuello, antenas)
_JOINTS_TO_LOCK_NAMES = [
    "dof_neck_pitch",
    "dof_head_pitch",
    "dof_head_yaw",
    "dof_head_roll",
    "fix_left_antenna",
    "fix_right_antenna",
]


def _patch_urdf(urdf_path: str) -> str:
    """
    Corrige el URDF: trunk_assembly_1 tiene <geometry> vacía en collision,
    lo que hace fallar a urdfdom. Devuelve ruta a URDF temporal corregido.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        if link.get("name") == "trunk_assembly_1":
            for coll in link.findall("collision"):
                geo = coll.find("geometry")
                if geo is not None and len(geo) == 0:
                    box = ET.SubElement(geo, "box")
                    box.set("size", "0.12 0.08 0.14")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="w", encoding="utf-8"
    )
    tree.write(tmp.name, encoding="unicode", xml_declaration=True)
    tmp.close()
    return tmp.name


def set_joint(q: np.ndarray, model: pin.Model, joint_name: str, val: float):
    """Asigna un valor a una articulación por nombre en el vector q."""
    jid = model.getJointId(joint_name)
    if jid < model.njoints and model.joints[jid].nq == 1:
        q[model.joints[jid].idx_q] = val


def q_from_base_and_joints(q: np.ndarray, oMb: pin.SE3) -> np.ndarray:
    """
    Ensambla el vector de configuración completo a partir de
    la pose del base link (SE3) y los valores de joints actuales en q.
    """
    quat = pin.Quaternion(oMb.rotation)
    q_out = q.copy()
    q_out[:3] = oMb.translation
    q_out[3:7] = np.array([quat.x, quat.y, quat.z, quat.w])
    return q_out


class LakObedience:
    """
    Modelo cinemático del robot LAK Obedience.

    Interfaz idéntica a la clase Talos del repo original.

    Parameters
    ----------
    urdf_path : str | Path
        Ruta al archivo lak_obedience.urdf
    reduced : bool
        Si True, construye modelo reducido bloqueando joints de cabeza/antenas.
        Recomendado para control de marcha (reduce nq y simplifica IK).
    """

    def __init__(self, urdf_path: str | Path, reduced: bool = True):
        urdf_path = str(urdf_path)
        patched = _patch_urdf(urdf_path)

        try:
            full_model = pin.buildModelFromUrdf(patched, pin.JointModelFreeFlyer())
        finally:
            os.unlink(patched)

        # Configuración de referencia para joints bloqueados
        q_ref = pin.neutral(full_model)
        for name in _JOINTS_TO_LOCK_NAMES:
            set_joint(q_ref, full_model, name, 0.0)

        self.reduced = reduced

        if reduced:
            lock_ids = [
                full_model.getJointId(n)
                for n in _JOINTS_TO_LOCK_NAMES
                if full_model.getJointId(n) < full_model.njoints
            ]
            self.model = pin.buildReducedModel(full_model, lock_ids, q_ref)
        else:
            self.model = full_model

        self.data = self.model.createData()

        # Frame IDs — verificados contra el URDF real
        self.left_foot_id  = self.model.getFrameId("foot_assembly_2")
        self.right_foot_id = self.model.getFrameId("foot_assembly_1")
        self.torso_id      = self.model.getFrameId("trunk_assembly_1")

        assert self.left_foot_id  < len(self.model.frames), "frame foot_assembly_2 no encontrado"
        assert self.right_foot_id < len(self.model.frames), "frame foot_assembly_1 no encontrado"
        assert self.torso_id      < len(self.model.frames), "frame trunk_assembly_1 no encontrado"

    def set_and_get_default_pose(self) -> np.ndarray:
        """
        Postura inicial con rodillas ligeramente flexionadas.

        Calibrada para que el CoM quede en z≈0.22 m con los pies
        aproximadamente en z=0 (base_link en el suelo).

        Returns
        -------
        q : ndarray, shape (nq,)
        """
        q = pin.neutral(self.model)

        # Pierna izquierda
        set_joint(q, self.model, "dof_left_hip_pitch", -0.3)
        set_joint(q, self.model, "dof_left_knee",       0.6)
        set_joint(q, self.model, "dof_left_ankle",     -0.3)
        set_joint(q, self.model, "dof_left_hip_roll",   0.0)
        set_joint(q, self.model, "dof_left_hip_yaw",    0.0)

        # Pierna derecha (hip_pitch tiene signo opuesto por convención URDF)
        set_joint(q, self.model, "dof_right_hip_pitch",  0.3)
        set_joint(q, self.model, "dof_right_knee",       0.6)
        set_joint(q, self.model, "dof_right_ankle",     -0.3)
        set_joint(q, self.model, "dof_right_hip_roll",   0.0)
        set_joint(q, self.model, "dof_right_hip_yaw",    0.0)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return q

    def get_com_height(self) -> float:
        """Retorna la altura real del CoM en la postura default. Útil para calibrar zc."""
        q = self.set_and_get_default_pose()
        com = pin.centerOfMass(self.model, self.data, q)
        return float(com[2])

    def get_locked_joints_for_ik(self) -> list:
        """
        Lista de joint IDs a bloquear en la IK.
        Si el modelo es reducido, la cabeza ya fue eliminada → lista vacía.
        """
        if self.reduced:
            return []
        return [
            self.model.getJointId(n)
            for n in _JOINTS_TO_LOCK_NAMES
            if self.model.getJointId(n) < self.model.njoints
        ]

    @staticmethod
    def get_lf_link_name() -> str:
        return "foot_assembly_2"

    @staticmethod
    def get_rf_link_name() -> str:
        return "foot_assembly_1"

    def print_info(self):
        """Diagnóstico rápido del modelo cargado."""
        q = self.set_and_get_default_pose()
        com = pin.centerOfMass(self.model, self.data, q)
        lf  = self.data.oMf[self.left_foot_id].translation
        rf  = self.data.oMf[self.right_foot_id].translation
        print("─" * 55)
        print(f"  LAK Obedience ({'reducido' if self.reduced else 'completo'})")
        print(f"  nq={self.model.nq}  nv={self.model.nv}  nframes={len(self.model.frames)}")
        print(f"  left_foot  = frame {self.left_foot_id} ({self.model.frames[self.left_foot_id].name})")
        print(f"  right_foot = frame {self.right_foot_id} ({self.model.frames[self.right_foot_id].name})")
        print(f"  torso      = frame {self.torso_id} ({self.model.frames[self.torso_id].name})")
        print(f"  CoM postura default: z={com[2]:.4f} m  (usar zc={com[2]:.2f})")
        print(f"  Pie izq: y={lf[1]:.4f}  z={lf[2]:.4f}")
        print(f"  Pie der: y={rf[1]:.4f}  z={rf[2]:.4f}")
        print(f"  Separación lateral: {abs(lf[1]-rf[1]):.4f} m")
        print("─" * 55)