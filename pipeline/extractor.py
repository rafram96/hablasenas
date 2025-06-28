import os
import time
import sys
import cv2
import numpy as np
import json
from src.capture import KeypointExtractor

# El directorio raíz del proyecto es el padre directo de pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)




def main(
    max_samples: int = 100,
    threshold: float = 0.2
):
    # Pedir etiqueta para las muestras
    label = input("Ingrese etiqueta (letra/palabra) o ENTER para salir: ")
    # Normalizar etiqueta a minúsculas
    label = label.lower()
    if not label:
        print("Etiqueta no ingresada, cancelando extracción.")
        return
    # Crear carpeta por etiqueta en data/features
    label_dir = os.path.join(project_root, 'data', 'features', label)
    os.makedirs(label_dir, exist_ok=True)
    # Timestamp para nombres de archivo
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    cap = cv2.VideoCapture(0)
    extractor = KeypointExtractor(mode=True, maxHands=2, detectionCon=0.2, trackCon=0.2)
    samples = []
    print(f"Buscando {max_samples} frames con más de {threshold*100:.0f}% de detección...")

    while len(samples) < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extractor.extract(frame)
        ratio = np.count_nonzero(kp) / kp.size
        cv2.putText(frame, f"Ratio: {ratio:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Test Pipeline', frame)
        key = cv2.waitKey(1) & 0xFF
        if ratio >= threshold:
            samples.append(kp)
            print(f"Frame {len(samples)}/{max_samples} guardado (ratio={ratio:.2f})")
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("No se capturaron frames válidos.")
        return

    arr = np.stack(samples, axis=0)
    # Confirmar si se desea descartar la toma antes de guardar
    choice = input("¿Desea descartar esta toma? Presione 'q' para descartar o ENTER para guardar: ")
    if choice.lower() == 'q':
        print("Toma descartada, no se guardará ningun archivo.")
        return
    # Guardar archivos en carpeta de la etiqueta
    npy_name = f"{label}_{timestamp}.npy"
    json_name = f"{label}_{timestamp}_summary.json"
    path_npy = os.path.join(label_dir, npy_name)
    np.save(path_npy, arr)
    print(f"Guardados {arr.shape[0]} vectores en {path_npy}, shape={arr.shape}")

    nz = int(np.count_nonzero(arr))
    total = int(arr.size)
    global_ratio = nz / total
    # Diccionario de resumen con claves en español
    summary = {
        'nombre_archivo': npy_name,
        'num_muestras': arr.shape[0],
        'forma_caracteristicas': list(arr.shape),
        'no_ceros_global': [nz, total],
        'ratio_no_ceros_global': round(global_ratio, 4)*100,
        'ratio_no_ceros_por_muestra': [round(np.count_nonzero(sample)/sample.size, 4)*100 for sample in samples]
    }
    json_path = os.path.join(label_dir, json_name)
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2)
    print(f"Guardado resumen en: {json_path}")
    # Actualizar labels.json automáticamente
    features_root = os.path.join(project_root, 'data', 'features')
    labels_file = os.path.join(features_root, 'labels.json')
    labels_list = []
    if os.path.isfile(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as lf:
            try:
                labels_list = json.load(lf)
            except json.JSONDecodeError:
                labels_list = []
    # Agregar nueva entrada
    rel_path = os.path.relpath(path_npy, project_root)
    labels_list.append({'filename': rel_path, 'label': label})
    with open(labels_file, 'w', encoding='utf-8') as lf:
        json.dump(labels_list, lf, indent=2, ensure_ascii=False)
    print(f"Labels.json actualizado con: {rel_path} -> {label}")

if __name__ == '__main__':
    main()
