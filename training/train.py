"""
train.py

Script para entrenar el modelo de clasificación de gestos:
- Carga features (.npy) y etiquetas (labels.json) generados por el pipeline.
- Construye matrices X (samples × features) e y (labels).
- Entrena un RandomForest y guarda el modelo en `model.joblib`.
"""
import os
import json
import numpy as np
from src.model import GestureClassifier


def load_dataset(features_dir: str):
    """
    Carga todos los archivos npy y su etiqueta definida en labels.json.
    Devuelve X (ndarray) e y (lista).
    """
    labels_path = os.path.join(features_dir, 'labels.json')
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"No se encontró labels.json en {features_dir}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    X_list, y_list = [], []
    for entry in dataset:
        npy_path = entry['filename']
        label = entry['label']
        if not os.path.isfile(npy_path):
            print(f"Advertencia: no existe {npy_path}, se omite.")
            continue
        arr = np.load(npy_path)
        # arr.shape = (num_samples, num_features)
        for sample in arr:
            X_list.append(sample)
            y_list.append(label)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    features_dir = os.path.join(project_root, 'data', 'features')
    print(f"Cargando dataset de: {features_dir}")
    X, y = load_dataset(features_dir)
    print(f"Dataset cargado: X.shape={X.shape}, y.len={len(y)}")

    clf = GestureClassifier(model_path=os.path.join(project_root, 'model.joblib'))
    clf.train(X, y)

    print("Entrenamiento completado.")

if __name__ == '__main__':
    main()
