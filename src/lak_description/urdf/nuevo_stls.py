import xml.etree.ElementTree as ET
import trimesh
import trimesh.transformations as tf
import numpy as np
import os

def procesar_urdf(urdf_path, output_dir, procesar_tipo="visual"):
    """
    Lee un URDF y unifica los STL de cada link en un solo archivo.
    procesar_tipo puede ser "visual" o "collision".
    """
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    for link in root.findall('link'):
        link_name = link.get('name')
        meshes_to_combine = []

        # Buscar todos los elementos visuales (o de colisión) del link
        for elemento in link.findall(procesar_tipo):
            # Obtener el origen (transformación)
            origin = elemento.find('origin')
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            
            if origin is not None:
                if origin.get('xyz'):
                    xyz = [float(x) for x in origin.get('xyz').split()]
                if origin.get('rpy'):
                    rpy = [float(x) for x in origin.get('rpy').split()]

            # Obtener la geometría (STL)
            geometry = elemento.find('geometry')
            if geometry is not None:
                mesh_tag = geometry.find('mesh')
                if mesh_tag is not None:
                    filename = mesh_tag.get('filename')



                    # Manejo del prefijo package://
                    if filename.startswith("package://"):
                        # 1. Quitamos el prefijo
                        ruta_sin_prefijo = filename.replace("package://", "")
                        
                        # 2. Separamos el nombre del paquete del resto de la ruta
                        # Ejemplo: "mi_robot_pkg/meshes/pieza.stl" -> ["mi_robot_pkg", "meshes/pieza.stl"]
                        partes = ruta_sin_prefijo.split("/", 1)
                        
                        if len(partes) == 2:
                            nombre_paquete, ruta_relativa = partes
                            
                            # --- ¡MODIFICA ESTA LÍNEA! ---
                            # Pon la ruta absoluta de la carpeta src de tu workspace de ROS
                            ruta_directorio_src = "/home/eliascm/ros2/lak_obedience_ws/src"
                            
                            # Construimos la ruta final: /home/.../src/mi_robot_pkg/meshes/pieza.stl
                            filepath = os.path.join(ruta_directorio_src, nombre_paquete, ruta_relativa)
                        else:
                            print(f"Error parseando la ruta del paquete: {filename}")
                            continue
                    else:
                        # Si la ruta no tiene package://, asumimos que es relativa al URDF
                        filepath = os.path.join(urdf_dir, filename)
                    
                    # Ahora sí, comprobar si existe y cargar
                    if os.path.exists(filepath):
                        # Cargar el STL
                        malla = trimesh.load_mesh(filepath)
                        
                        # 1. Crear matriz de rotación a partir de RPY (Roll-Pitch-Yaw)
                        # URDF usa ángulos de Euler estáticos XYZ ('sxyz')
                        matrix = tf.euler_matrix(rpy[0], rpy[1], rpy[2], axes='sxyz')
                        
                        # 2. Añadir la traslación a la matriz
                        matrix[0, 3] = xyz[0]
                        matrix[1, 3] = xyz[1]
                        matrix[2, 3] = xyz[2]
                        
                        # 3. Aplicar la transformación a los vértices de la malla
                        malla.apply_transform(matrix)
                        
                        meshes_to_combine.append(malla)
                    else:
                        print(f"Error: No se encontró el archivo {filepath}")

        # Si hay mallas para combinar en este link, unificarlas
        if meshes_to_combine:
            # Concatenar todas las mallas transformadas
            malla_unificada = trimesh.util.concatenate(meshes_to_combine)
            
            # Exportar el nuevo STL
            output_filename = os.path.join(output_dir, f"{link_name}_combined.stl")
            malla_unificada.export(output_filename)
            print(f"Éxito: Se ha creado el STL unificado para '{link_name}' en {output_filename}")

# --- Configuración y Ejecución ---
if __name__ == "__main__":
    # Ruta a tu archivo URDF actual
    ruta_urdf = "lak_obedience_v2 copy.urdf" 
    
    # Carpeta donde se guardarán los nuevos STL unificados
    carpeta_salida ="../meshes_v3/visual_2"  #"../meshes_v3/visual"
    
    # print("--- Procesando Visuales ---")
    # procesar_urdf(ruta_urdf, carpeta_salida, procesar_tipo="visual")
    
    # Si también quieres unificar las colisiones, descomenta la siguiente línea:
    print("\n--- Procesando Colisiones ---")
    procesar_urdf(ruta_urdf, carpeta_salida, procesar_tipo="collision")