"""
model.py

Clase GestureClassifier para entrenamiento e inferencia de gestos LSP.
"""
import joblib
from sklearn.ensemble import RandomForestClassifier

class GestureClassifier:
    def __init__(self, model_path="model.joblib"):
        self.model_path = model_path
        self.model = None

    def train(self, X, y):
        """
        Entrena el clasificador con X (features) e y (labels) y guarda el modelo.
        """
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, self.model_path)
        self.model = clf
        print(f"Modelo entrenado y guardado en {self.model_path}")

    def load(self):
        """
        Carga el modelo desde el archivo.
        """
        self.model = joblib.load(self.model_path)
        print(f"Modelo cargado desde {self.model_path}")

    def predict(self, features):
        """
        Devuelve la predicción (etiqueta) para un solo vector de características.
        """
        if self.model is None:
            self.load()
        return self.model.predict([features])[0]
