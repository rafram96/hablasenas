import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

class GestureClassifier:
    """
    Envuelve un RandomForest con escalado, entrenamiento, evaluación, guardado y carga.
    """
    def __init__(self,
                 model_path: str = None,
                 n_estimators: int = 400,
                 random_state: int = 96):
        if model_path is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..'))
            model_path = os.path.join(project_root, 'training', 'model.joblib')
        self.model_path = model_path

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state))
        ])




    def train(self, X, y, test_size: float = 0.2, cv: int = 5, save: bool = True):
        """
        Entrena el modelo usando X, y; separa un test interno y reporta métricas.
        Si cv>1, hace validación cruzada en vez de split.
        Devuelve dict con accuracy y report.
        """
        if cv > 1:
            scores = cross_val_score(self.pipeline, X, y, cv=cv)
            metrics = {
                'cv_mean_accuracy': float(scores.mean()),
                'cv_std_accuracy': float(scores.std())
            }
            self.pipeline.fit(X, y)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)
            self.pipeline.fit(X_tr, y_tr)
            y_pred = self.pipeline.predict(X_te)
            metrics = {
                'test_accuracy': float(accuracy_score(y_te, y_pred)),
                'classification_report': classification_report(
                    y_te, y_pred, output_dict=True)
            }
        if save:
            self.save()
        return metrics



    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"✅ Modelo guardado en {self.model_path}")


    def load(self):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"No existe el modelo: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        print(f"✅ Modelo cargado desde {self.model_path}")

    def predict(self, X):
        """
        Predice etiquetas para un array 2D X (muestras × features).
        Retorna lista de predicciones.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Devuelve probabilidades de cada clase para X (útil para confianza).
        """
        return self.pipeline.predict_proba(X)

    def evaluate(self, X, y):
        """
        Evalúa el modelo en un conjunto dado X, y.
        Retorna dict con accuracy y classification_report.
        """
        y_pred = self.pipeline.predict(X)
        return {
            'accuracy': float(accuracy_score(y, y_pred)),
            'classification_report': classification_report(
                y, y_pred, output_dict=True)
        }
