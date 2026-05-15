import pinocchio as pin
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

pkg_path = get_package_share_directory("lak_description")

def validar_robot(urdf_path):
    # 1. Cargar el modelo URDF en Pinocchio
    # Esta función construye el mapa matemático de juntas y eslabones [1]
    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        print(f"✅ Modelo cargado exitosamente. Juntas: {model.njoints}, Frames: {model.nframes}")
    except Exception as e:
        print(f"❌ Error al cargar el URDF: {e}")
        return

    # 2. Nombres de frames requeridos por el InvKinSolverParams [2]
    # DEBES REEMPLAZAR ESTOS NOMBRES con los exactos de tu URDF en lak_obedience
    frames_a_validar = {
        "torso": "trunk_assembly_1",           # Frame para la orientación del torso [4]
        "pie_izquierdo": "foot_assembly_2",    # Frame para seguimiento de trayectoria del pie [5]
        "pie_derecho": "foot_assembly_1"      # Frame para la restricción de pie fijo [5]
    }

    print("\n--- Validación de Frames ---")
    ids_encontrados = {}
    for clave, nombre in frames_a_validar.items():
        if model.existFrame(nombre):
            f_id = model.getFrameId(nombre)
            ids_encontrados[clave] = f_id
            print(f"✔ Frame '{clave}' ({nombre}) encontrado. ID: {f_id}")
        else:
            print(f"✖ Frame '{clave}' ({nombre}) NO EXISTE en el URDF.")

    # 3. Validación de la posición del Centro de Masa (CoM)
    # El modelo LIPM depende de una altura de CoM (zc) constante [6, 7]
    q_neutral = pin.neutral(model) # Postura inicial de los motores
    pin.forwardKinematics(model, data, q_neutral)
    pin.updateFramePlacements(model, data)
    
    com_pos = pin.centerOfMass(model, data, q_neutral)
    print(f"\n--- Análisis de Masa ---")
    print(f"📍 Posición inicial del CoM: {com_pos}")
    print(f"💡 Tu parámetro 'zc' (altura de CoM) debería ser cercano a: {com_pos[2]:.4f} m")

    # 4. Verificación de la altura de los pies
    if "pie_izquierdo" in ids_encontrados:
        pose_pie = data.oMf[ids_encontrados["pie_izquierdo"]]
        print(f"🦶 Altura del pie izq. en pose neutral: {pose_pie.translation[2]:.4f} m")

# Ruta al URDF de tu repositorio lak_obedience
# (Asegúrate de que la ruta sea correcta en tu entorno)
urdf_file = pkg_path + "/urdf/lak_obedience.urdf"


def main():
    validar_robot(urdf_file)

if __name__ == "__main__":
    main()