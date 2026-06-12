import os
import pymeshlab as ml

# Configuración de rutas
FOLDER_INPUT = "./visual_2"
FOLDER_OUTPUT = "./collision_2"
os.makedirs(FOLDER_OUTPUT, exist_ok=True)

def procesar_eslabon_simple(input_path, output_path):
    """Genera un único Convex Hull usando PyMeshLab"""
    ms = ml.MeshSet()
    ms.load_new_mesh(input_path)
    
    # Aplica el filtro de convex hull clásico
    ms.apply_filter('generate_convex_hull')
    
    ms.save_current_mesh(output_path)
    print(file=None) # Línea limpia para consola
    print(f"✅ [Simple] Procesado con PyMeshLab: {os.path.basename(output_path)}")



# --- PIPELINE PRINCIPAL ---
def main():
    for filename in os.listdir(FOLDER_INPUT):
        if not filename.lower().endswith(('.stl', '.obj')):
            continue
            
        input_path = os.path.join(FOLDER_INPUT, filename)
        output_path = os.path.join(FOLDER_OUTPUT, filename)
        
        procesar_eslabon_simple(input_path, output_path)

if __name__ == "__main__":
    main()