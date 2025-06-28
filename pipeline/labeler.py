import os
import json
import argparse

def label_pipeline(clips_dir: str, output_json: str) -> None:
    """
    Etiqueta interactiva de clips (.avi) y features (.npy) en `clips_dir`, guardando el JSON en `output_json`.
    """
    # Cargar etiquetas previas si existen
    existing = {}
    if os.path.isfile(output_json):
        with open(output_json, 'r', encoding='utf-8') as jf:
            existing = json.load(jf)
    # Listar archivos nuevos ignorando carpeta 'discarded' y ya etiquetados
    items = sorted([
        f for f in os.listdir(clips_dir)
        if f.lower().endswith(('.avi', '.npy')) and f not in existing
    ])
    # Carpeta para clips descartados
    discard_dir = os.path.join(clips_dir, 'discarded')
    os.makedirs(discard_dir, exist_ok=True)
    labels = existing.copy()
    
    for item in items:
        print(f"Procesando: {item}")
        label = input("Introduce etiqueta (A-Z o palabra) o 'd' para descartar: ").strip()
        if label.lower() == 'd':
            # Mover clip a carpeta descartados
            src = os.path.join(clips_dir, item)
            dst = os.path.join(discard_dir, item)
            os.rename(src, dst)
            print(f"Descartado: {item}")
            continue
        labels[item] = label

    # Asegurar que existe la carpeta de salida
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    # Guardar etiquetas combinadas (previas + nuevas)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Etiquetas guardadas en {output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline de etiquetado')
    parser.add_argument('--in_dir', default='data/clips', help='Directorio con clips brutos (.avi)')
    parser.add_argument('--out_json', default='data/labels.json', help='Archivo JSON de etiquetas')
    args = parser.parse_args()
    # Convertir rutas relativas a absolutas según la raíz del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_dir = args.in_dir if os.path.isabs(args.in_dir) else os.path.join(project_root, args.in_dir)
    output_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(project_root, args.out_json)
    label_pipeline(input_dir, output_json)