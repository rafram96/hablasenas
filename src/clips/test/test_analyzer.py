import os
import sys
import numpy as np
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from pipeline.lector import get_hand_landmarks, get_face_landmarks

def analyze_frame(arr: np.ndarray, frame_idx: int):
    frame = arr[frame_idx]
    print("\n" + "="*50)
    print(f"   📊 Análisis del Frame #{frame_idx}")
    print("="*50)
    # Manos
    hands = get_hand_landmarks(frame, maxHands=2)
    print("\n▶ Hands Landmarks:")
    for h_idx, hand in enumerate(hands):
        print(f"  Mano {h_idx+1}:")
        for idx, (x, y, z, p) in enumerate(hand):
            presence = '✔' if p else '✖'
            print(f"    [{idx:02d}] x={x:.3f}, y={y:.3f}, z={z:.3f} {presence}")
    # Cara
    face = get_face_landmarks(frame, maxHands=2)
    print("\n▶ Face Landmarks (índices clave):")
    # índices ampliados: puente nariz, punta nariz, frente, ojos internos, cejas, pómulos, boca, mentón
    important_idxs = [1, 4, 2, 5, 10, 33, 133, 55, 65, 93, 199, 61, 291, 285, 295, 323]
    for idx in important_idxs:
        x, y, z = face[idx]
        print(f"    [idx {idx:03d}] x={x:.3f}, y={y:.3f}, z={z:.3f}")
    print("="*50 + "\n")

def analyze_specific_frame(npy_path: str, frame_idx: int):
    print(f"\n>>> Análisis específico del frame {frame_idx} <<<")
    arr = np.load(npy_path)
    analyze_frame(arr, frame_idx)

def print_npy(npy_path: str):
    if not os.path.isfile(npy_path):
        print(f"Archivo no encontrado: {npy_path}")
        return
    arr = np.load(npy_path)
    print(f"\n=== Contenido de {npy_path} ===")
    print(arr)

def main():
    print("\n*** Generando Reporte Completo ***")
    # Configuración de rutas
    npy_file = 'test_output/samples.npy'
    json_report = os.path.join(project_root, 'test_output', 'analysis_report.json')
    npy_path = os.path.join(project_root, npy_file)
    if not os.path.isfile(npy_path):
        print(f"❌ Archivo .npy no encontrado: {npy_path}")
        return
    arr = np.load(npy_path)
    report = {
        'npy_file': npy_file,
        'shape': arr.shape,
        'global_non_zero_ratio': round(float(np.count_nonzero(arr) / arr.size),4)
    }
    os.makedirs(os.path.dirname(json_report), exist_ok=True)
    with open(json_report, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2)
    print(f"✅ Reporte guardado en: {json_report}\n")


if __name__ == '__main__':
    main()
    npy_path = os.path.join(project_root, 'test_output', 'samples.npy')
    analyze_specific_frame(npy_path, 60)
    # para ver todo el contenido del .npy
    # print_npy(npy_path)
