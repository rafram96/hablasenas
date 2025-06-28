import os
import json
import argparse

def label_clips(
    clips_dir: str,
    output_json: str
) -> None:
    """
    Recorre los archivos .avi o .npy en clips_dir e interactúa con el usuario para asignar una etiqueta.
    Guarda un JSON con el mapeo {"archivo": "etiqueta"} en output_json.
    """
    # Obtener lista de ficheros a etiquetar
    items = sorted([f for f in os.listdir(clips_dir)
                    if f.lower().endswith(('.avi', '.npy'))])
    labels = {}

    for item in items:
        path = os.path.join(clips_dir, item)
        print(f"Archivos para etiquetar: {item}")
        label = input("Introduce la etiqueta (A-Z o palabra): ").strip()
        labels[item] = label
        print(f"Etiquetado: {item} -> {label}\n")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"Etiquetas guardadas en {output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Etiquetador de clips/features')
    parser.add_argument('clips_dir', help='Directorio con clips (.avi) o features (.npy)')
    parser.add_argument('out_json', help='Ruta de salida para el JSON de etiquetas')
    args = parser.parse_args()
    label_clips(args.clips_dir, args.out_json)
