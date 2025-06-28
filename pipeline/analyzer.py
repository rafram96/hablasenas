import os
import numpy as np
import json


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

def compute_fluidity(arr: np.ndarray, maxHands: int = 2) -> list:
    """
    Calcula la fluidez media de las manos entre frames consecutivos.
    """
    fluid = []
    for i in range(1, arr.shape[0]):
        prev = get_hand_landmarks(arr[i-1], maxHands)[..., :3]
        curr = get_hand_landmarks(arr[i],   maxHands)[..., :3]
        # media de distancias euclidianas por mano
        dists = [np.mean(np.linalg.norm(curr[h] - prev[h], axis=1)) for h in range(maxHands)]
        fluid.append(float(np.mean(dists)))
    return fluid

def analyze_npy(npy_path: str, report_json: bool = True):
    """
    Analiza un archivo .npy específico, genera un reporte JSON con estadísticas globales
    y muestra el análisis del último frame en la consola.
    """
    arr = np.load(npy_path)
    # Reporte global en español
    report = {
        'archivo_npy': os.path.basename(npy_path),
        'forma': arr.shape,
        'ratio_no_ceros_global': round(float(np.count_nonzero(arr) / arr.size), 4)*100,  # en porcentaje
    }
    # Calcular métricas de fluidez y añadir al reporte
    fluid = compute_fluidity(arr)
    if fluid:
        report['fluidez_media'] = round(float(np.mean(fluid)), 4)*100  # en porcentaje
        report['fluidez_desviacion']  = round(float(np.std(fluid)),  4)*100  # en porcentaje
    else:
        report['fluidez_media'] = 0.0
        report['fluidez_desviacion'] = 0.0
    if report_json:
        # Determinar etiqueta y ruta de reportes en data/reports/<label>
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
        # Mostrar métricas de fluidez en consola
        print(f"▶ Fluidez media: {report['fluidez_media']:.4f}, desviación estándar: {report['fluidez_desviacion']:.4f}")
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
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    features_dir = os.path.join(project_root, 'data', 'features')
    if not os.path.isdir(features_dir):
        print(f"No se encontró carpeta de features: {features_dir}")
        sys.exit(1)
    # Solicitar ruta específica o analizar último archivo
    choice = input("Ingrese ruta de archivo .npy para análisis o presione ENTER para usar el último creado: ")
    if choice:
        npy_path = choice.strip()
        if not os.path.isfile(npy_path):
            print(f"No se encontró el archivo especificado: {npy_path}")
            sys.exit(1)
        analyze_npy(npy_path, report_json=True)
    else:
        # Encontrar el último .npy creado
        latest_file = None
        latest_mtime = 0
        for root, _, files in os.walk(features_dir):
            for fname in files:
                if not fname.endswith('.npy'):
                    continue
                path = os.path.join(root, fname)
                mtime = os.path.getmtime(path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = path
        if not latest_file:
            print(f"No se encontraron archivos .npy en {features_dir}")
            sys.exit(1)
        print(f"Analizando último archivo creado: {latest_file}")
        analyze_npy(latest_file, report_json=True)
