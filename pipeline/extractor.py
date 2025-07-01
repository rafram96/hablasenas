import os
import time
import sys
import cv2
import numpy as np
import json
from src.capture import KeypointExtractor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main(
    max_samples: int = 50,
    threshold: float = 0.2
):
    while True:
        label = input("Ingrese etiqueta (letra/palabra) o ENTER para salir: ").strip().lower()
        if not label:
            print("Extracción finalizada.")
            break

        label_dir = os.path.join(project_root, 'data', 'features', label)
        os.makedirs(label_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        cap = cv2.VideoCapture(0)
        extractor = KeypointExtractor(mode=True, maxHands=2, detectionCon=0.75, trackCon=0.65)
        samples = []
        print(f"Buscando hasta {max_samples} frames con ≥{threshold*100:.0f}% de detección...")

        hand_dims = 2 * 21 * 4
        while len(samples) < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            kp = extractor.extract(frame)
            kp_hand = kp[:hand_dims]
            ratio = np.count_nonzero(kp_hand) / kp_hand.size

            cv2.putText(frame, f"Ratio: {ratio:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Extraccion de Datos', frame)
            key = cv2.waitKey(1) & 0xFF

            if ratio >= threshold:
                samples.append(kp_hand)
                print(f"Frame {len(samples)}/{max_samples} guardado (ratio={ratio:.2f})")
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if not samples:
            print("No se capturaron frames válidos, salteando esta etiqueta.")
            continue

        arr = np.stack(samples, axis=0)

        choice = input("¿Desea descartar esta toma? Presione 'q' para descartar o ENTER para guardar: ")
        if choice.lower() == 'q':
            print("Toma descartada.")
            continue

        npy_name  = f"{label}_{timestamp}.npy"
        json_name = f"{label}_{timestamp}_summary.json"
        path_npy  = os.path.join(label_dir, npy_name)
        np.save(path_npy, arr)
        print(f"Guardados {arr.shape[0]} vectores en {path_npy}")

        nz = int(np.count_nonzero(arr))
        total = int(arr.size)
        global_ratio = nz / total
        summary = {
            'nombre_archivo': npy_name,
            'num_muestras': arr.shape[0],
            'forma_caracteristicas': list(arr.shape),
            'no_ceros_global': [nz, total],
            'ratio_no_ceros_global': round(global_ratio,4)*100,
            'ratio_no_ceros_por_muestra': [
                round(np.count_nonzero(s)/s.size,4)*100
                for s in samples
            ]
        }
        json_path = os.path.join(label_dir, json_name)
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(summary, jf, indent=2)
        print(f"Resumen guardado en: {json_path}")


        features_root = os.path.join(project_root, 'data', 'features')
        labels_file   = os.path.join(features_root, 'labels.json')
        labels_list   = []
        if os.path.isfile(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as lf:
                try:
                    labels_list = json.load(lf)
                except json.JSONDecodeError:
                    labels_list = []
        rel_path = os.path.relpath(path_npy, project_root)
        labels_list.append({'filename': rel_path, 'label': label})
        with open(labels_file, 'w', encoding='utf-8') as lf:
            json.dump(labels_list, lf, indent=2, ensure_ascii=False)
        print(f"labels.json actualizado con: {rel_path} -> {label}\n")

if __name__ == '__main__':
    main()