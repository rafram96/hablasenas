"""
interactive_data_collector.py

Script interactivo para captura de muestras de LSP: solicita una etiqueta,
recoge vectores de keypoints cuando superan un ratio de detección,
guarda los features en carpetas por etiqueta para entrenar el modelo.
"""
import os
import sys
import time
import cv2
import numpy as np
from src.capture import KeypointExtractor

# Configuración por defecto
default_max_samples = 50
default_threshold = 0.2  # ratio mínimo de landmarks detectados


def collect_samples(label: str, output_root: str, max_samples: int, threshold: float):
    """Captura max_samples vectores de características para la etiqueta dada."""
    extractor = KeypointExtractor(mode=True, maxHands=2)
    cap = cv2.VideoCapture(0)
    saved = []
    print(f"\nIniciando captura para etiqueta '{label}' (objetivo: {max_samples} muestras, umbral={threshold})")
    while len(saved) < max_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame de cámara.")
            break
        kp = extractor.extract(frame)
        ratio = np.count_nonzero(kp) / kp.size
        cv2.putText(frame, f"Ratio: {ratio:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Collector', frame)
        key = cv2.waitKey(1) & 0xFF
        if ratio >= threshold:
            saved.append(kp)
            print(f"  Muestra {len(saved)}/{max_samples} capturada (ratio={ratio:.2f})")
        if key == ord('q'):
            print("Captura interrumpida por usuario.")
            break
    cap.release()
    cv2.destroyAllWindows()

    if not saved:
        print("No se capturaron muestras para esta etiqueta.")
        return None

    # Guardar vectores en .npy dentro de carpeta por etiqueta
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    label_dir = os.path.join(output_root, label)
    os.makedirs(label_dir, exist_ok=True)
    filename = f"{label}_{timestamp}.npy"
    path_npy = os.path.join(label_dir, filename)
    arr = np.stack(saved, axis=0)
    np.save(path_npy, arr)
    print(f"Guardadas {arr.shape[0]} muestras en: {path_npy}")
    return path_npy


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_root = os.path.join(project_root, 'data', 'features')
    os.makedirs(output_root, exist_ok=True)

    print("=== Pipeline Interactivo de Captura de Muestras ===")
    while True:
        label = input("Ingrese etiqueta (letra/palabra) o ENTER para salir: ")
        if not label:
            print("Saliendo del pipeline interactivo.")
            break
        # Opcionales: ajustar parámetros
        try:
            ms = int(input(f"Número de muestras (por defecto {default_max_samples}): ") or default_max_samples)
        except ValueError:
            ms = default_max_samples
        try:
            th = float(input(f"Umbral ratio (por defecto {default_threshold}): ") or default_threshold)
        except ValueError:
            th = default_threshold
        collect_samples(label, output_root, ms, th)

if __name__ == '__main__':
    main()
