import os
import numpy as np

def _analyze_vector(vector: np.ndarray, maxHands: int = 2):
    """
    Interpreta un vector plano de features en coordenadas de manos y cara.
    """
    per_hand = 21 * 4  # 4 valores por landmark: x,y,z,presencia
    print("\n-- Análisis del primer frame --")
    for h in range(maxHands):
        start = h * per_hand
        hand = vector[start:start+per_hand].reshape(21, 4)
        print(f"Mano {h+1}, Landmark[0]: x={hand[0,0]:.3f}, y={hand[0,1]:.3f}, z={hand[0,2]:.3f}, p={hand[0,3]:.0f}")
    face = vector[maxHands*per_hand:].reshape(468, 3)
    print(f"Cara, Landmark[0]: x={face[0,0]:.3f}, y={face[0,1]:.3f}, z={face[0,2]:.3f}")

def leer_todos_los_npy():
    # Obtener la ruta absoluta a la carpeta 'output' (una carpeta arriba de 'pipeline')
    carpeta_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

    # Listar solo los archivos que terminan en .npy
    archivos = [f for f in os.listdir(carpeta_output) if f.endswith('.npy')]

    if not archivos:
        print("No se encontraron archivos .npy en la carpeta 'output'.")
        return

    for archivo in archivos:
        ruta_completa = os.path.join(carpeta_output, archivo)
        try:
            datos = np.load(ruta_completa, allow_pickle=True)
            print(f"\n📄 Archivo: {archivo}")
            print(f"   Forma (shape): {datos.shape if hasattr(datos, 'shape') else 'No aplica'}")
            print("   Contenido:")
            # Estadísticas de elementos no cero
            non_zero = np.count_nonzero(datos)
            total = datos.size if hasattr(datos, 'size') else 0
            if total:
                print(f"   No ceros: {non_zero}/{total} ({non_zero/total*100:.2f}%)")
                # Mostrar vector plano o ejemplo de frame
                if datos.shape[1] > 10:
                    print(f"Vector plano (primer frame, recortado): {datos[0,:200]} ...")
                    _analyze_vector(datos[0])
                    # Estadísticas de non-zeros por frame
                    nz_frames = np.count_nonzero(datos, axis=1)
                    total_feats = datos.shape[1]
                    ratios = nz_frames / total_feats
                    print(f"   Ratio non-cero por frame: min {ratios.min():.2f}, max {ratios.max():.2f}, avg {ratios.mean():.2f}")
                else:
                    print(datos)
        except Exception as e:
            print(f"⚠️  Error al leer '{archivo}': {e}")

def leer_npy_especifico(nombre_archivo):
    carpeta_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    ruta = os.path.join(carpeta_output, nombre_archivo)

    if not os.path.isfile(ruta):
        print(f"❌ El archivo '{nombre_archivo}' no existe en la carpeta 'output'.")
        return

    try:
        datos = np.load(ruta, allow_pickle=True)
        print(f"\n📄 Archivo: {nombre_archivo}")
        print(f"   Forma (shape): {datos.shape if hasattr(datos, 'shape') else 'No aplica'}")
        print("   Contenido:")
        # Estadísticas de elementos no cero
        non_zero = np.count_nonzero(datos)
        total = datos.size if hasattr(datos, 'size') else 0
        if total:
            print(f"   No ceros: {non_zero}/{total} ({non_zero/total*100:.2f}%)")
            print(f"Vector plano (primer frame, recortado): {datos[0,:30]} ...")
            _analyze_vector(datos[0])
            # Estadísticas de non-zeros por frame
            nz_frames = np.count_nonzero(datos, axis=1)
            total_feats = datos.shape[1]
            ratios = nz_frames / total_feats
            print(f"   Ratio non-cero por frame: min {ratios.min():.2f}, max {ratios.max():.2f}, avg {ratios.mean():.2f}")
        else:
            print(datos)
    except Exception as e:
        print(f"⚠️  Error al leer '{nombre_archivo}': {e}")

def get_hand_landmarks(vector: np.ndarray, maxHands: int = 2) -> np.ndarray:
    """
    Dado un vector plano de (3*21*maxHands + 3*468), devuelve un array de forma
    (maxHands, 21, 3) con las coordenadas (x,y,z) de cada landmark de las manos.
    """
    # Si se pasa un string, interpretarlo como nombre de .npy en carpeta 'output'
    if isinstance(vector, str):
        archivo = vector
        carpeta_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        ruta = os.path.join(carpeta_output, archivo)
        arr = np.load(ruta)
        vector = arr[0]  # primer frame por defecto
    per_hand = 21 * 4  # x,y,z,p per landmark
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
    # Permitir filename como input
    if isinstance(vector, str):
        archivo = vector
        carpeta_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        ruta = os.path.join(carpeta_output, archivo)
        arr = np.load(ruta)
        vector = arr[0]
    per_hand = 21 * 4
    face_vec = vector[maxHands * per_hand:]
    return face_vec.reshape(468, 3)




# Ejecutar si se corre directamente
if __name__ == "__main__":
    num = "20250618_001333"
    clip = "clip_" + num + ".npy"
    # Prueba directa con el clip de ejemplo compartido
    leer_npy_especifico(clip)
    print("\n--- Análisis de landmarks ---")
    hands = get_hand_landmarks(clip, maxHands=2)
    face = get_face_landmarks(clip, maxHands=2)
    print("Landmarks de manos:")
    print(hands)
    print("---------------------------")
    print("Landmarks de cara:")
    print(face)