"""
__main__.py

Script orquestador del flujo completo:
1. Captura de muestras interactivas
2. Etiquetado interactivo de archivos .npy
3. Análisis de todos los features generados
"""
import os
from .extractor import extract_samples
from .labeler import label_features
from .analyzer import analyze_all_features


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    features_root = os.path.join(project_root, 'data', 'features')
    os.makedirs(features_root, exist_ok=True)

    print("=== RUN PIPELINE: Flujo completo de captura, etiquetado y análisis ===")
    # Paso 1: Captura
    label = input("1) Ingrese etiqueta (letra/palabra) o ENTER para saltar captura: ")
    if label:
        try:
            ms = int(input("   Número de muestras (por defecto 50): ") or 50)
        except ValueError:
            ms = 50
        try:
            th = float(input("   Umbral ratio (por defecto 0.2): ") or 0.2)
        except ValueError:
            th = 0.2
        extract_samples(label, features_root, max_samples=ms, threshold=th)

    # Paso 2: Etiquetado
    print("\n2) Etiquetado de archivos .npy en data/features")
    label_features(features_root)

    # Paso 3: Análisis
    print("\n3) Análisis de todas las características generadas")
    analyze_all_features(features_root)


if __name__ == '__main__':
    main()
