import os
import numpy as np
import json


# Funciones inyectadas de lector.py original:
def get_hand_landmarks(vector: np.ndarray, maxHands: int = 2) -> np.ndarray:
    """
    Dado un vector plano de (3*21*maxHands + 3*468), devuelve un array de forma
    (maxHands, 21, 3) con las coordenadas (x,y,z) de cada landmark de las manos.
    """
    if isinstance(vector, str):
        archivo = vector
        carpeta_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        ruta = os.path.join(carpeta_output, archivo)
        arr = np.load(ruta)
        vector = arr[0]
    per_hand = 21 * 4
    hands = []
    for i in range(maxHands):
        start = i * per_hand
        hand_vec = vector[start:start + per_hand]
        hands.append(hand_vec.reshape(21, 4))
    return np.stack(hands, axis=0)

def get_face_landmarks(vector: np.ndarray, maxHands: int = 2) -> np.ndarray:
    """
    Dado un vector plano de (3*21*maxHands + 3*468), devuelve un array de forma
    (468, 3) con las coordenadas (x,y,z) de cada landmark facial.
    """
    if isinstance(vector, str):
        archivo = vector
        carpeta_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        ruta = os.path.join(carpeta_output, archivo)
        arr = np.load(ruta)
        vector = arr[0]
    per_hand = 21 * 4
    face_vec = vector[maxHands * per_hand:]
    return face_vec.reshape(468, 3)

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

def analyze_npy(npy_path: str, report_json: bool = True):
    """
    Analiza un archivo .npy específico, genera un reporte JSON con estadísticas globales
    y muestra el análisis del último frame en la consola.
    """
    arr = np.load(npy_path)
    # Reporte global
    report = {
        'npy_file': os.path.basename(npy_path),
        'shape': arr.shape,
        'global_non_zero_ratio': round(float(np.count_nonzero(arr) / arr.size), 4)
    }
    if report_json:
        # Determinar etiqueta y ruta de reportes al mismo nivel de 'features'
        label = os.path.basename(os.path.dirname(npy_path))
        # Carpeta 'data/reports/<label>'
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        reports_dir = os.path.join(project_root, 'data', 'reports', label)
        os.makedirs(reports_dir, exist_ok=True)
        # Nombre del reporte
        base = os.path.splitext(os.path.basename(npy_path))[0]
        report_path = os.path.join(reports_dir, f"{base}_report.json")
        with open(report_path, 'w', encoding='utf-8') as jf:
            json.dump(report, jf, indent=2)
        print(f"✅ Reporte global guardado en: {report_path}")
    # Análisis del último frame
    analyze_frame(arr, -1)

def analyze_all_features(root_dir: str):
    """
    Analiza todos los archivos .npy en subcarpetas de root_dir y genera reportes JSON.
    """
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        print(f"\n=== Analizando features para etiqueta: {label} ===")
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith('.npy'):
                continue
            npy_path = os.path.join(label_dir, fname)
            print(f"--> {fname}")
            analyze_npy(npy_path, report_json=True)

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    features_dir = os.path.join(project_root, 'data', 'features')
    if not os.path.isdir(features_dir):
        print(f"No se encontró carpeta de features: {features_dir}")
        exit(1)
    analyze_all_features(features_dir)
