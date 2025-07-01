# Entrenamiento y Modelo de Clasificación de Gestos

Este documento describe el flujo de entrenamiento del modelo de clasificación de gestos y la arquitectura del pipeline implementado.

---

## 1. Estructura de datos

1. **Features**: archivos NumPy (`.npy`) con forma `(n_frames, n_features)` capturados por `pipeline/extractor.py`.  
   - **Solo manos**: 2 manos × 21 landmarks × 4 valores = **168 features**.
   - El x4 es por los valores que recoge cada landmark:
     - X: eje horizontal
     - Y: eje vertical
     - Z: profundidad
     - Visibilidad (presencia en el frame)
2. **Etiquetas**: `data/features/labels.json`, un array de objetos `{"filename": "...", "label": "a"}`.

---

## 2. Carga de datos (`training/train.py`)

La función `load_dataset(features_dir)` realiza:

1. Leer `labels.json`.  
2. Para cada entrada:
   - Resolver ruta absoluta de `.npy`.  
   - Cargar array: `arr = np.load(npy_path)` → forma `(n_samples, 168)`.  
   - Extraer cada fila `sample` y agregar a `X_list`, con su `label` en `y_list`.
3. Convertir `X_list` y `y_list` en matrices NumPy:
   ```python
   X = np.stack(X_list, axis=0)   # (total_samples, 168)
   y = np.array(y_list)           # (total_samples,)
   ```

Al finalizar, imprime `X.shape` y `len(y)`.

---

## 3. Arquitectura del modelo (`training/model.py`)

Se emplea `scikit-learn` con un pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

self.pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
         n_estimators=100,
         random_state=42
    ))
])
```

- **StandardScaler**: normaliza cada feature a media 0, varianza 1.
- **RandomForestClassifier**: ensemble de árboles con predicción robusta.

Métodos principales:
- `train(X, y)`: ajusta el pipeline y guarda con `joblib.dump` en `training/model.joblib`.
- `load()`: carga el pipeline guardado.
- `predict(X)`, `predict_proba(X)`: inferencia de etiquetas y probabilidades.
- `evaluate(X, y)`: calcula accuracy y reporte de clasificación.

---

## 4. Flujo de entrenamiento

Desde la raíz del proyecto:

```bash
python training/train.py
```

1. Se carga `data/features/labels.json` y todos los `.npy`.  
2. Se construye el dataset `X, y`.  
3. Se entrena el `GestureClassifier`:
   - Si `cv>1`, aplica validación cruzada (`cross_val_score`).  
   - Si `cv=1`, split train/test interno (80/20).
4. Imprime métricas (accuracy o media y desviación de CV).  
5. Guarda el pipeline completo en `training/model.joblib`.

---

## 5. Personalización y ajustes

- **Hiperparámetros**: `n_estimators`, `max_depth`, `random_state`.  
- **Validación**: cambiar `cv` o `test_size` en `train()`.  
- **Selección de features**: solo manos (168 dims) o incluir puntos faciales clave.  
- **Otros modelos**: probar `LogisticRegression`, SVM, redes neuronales.

---

## 6. Uso en la aplicación (`src/main.py`)

Una vez generado `training/model.joblib`, la aplicación en `src/main.py` carga el modelo y realiza inferencia en tiempo real:

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))
import cv2
from src.capture import KeypointExtractor
from src.recorder import ClipRecorder
from training.model import GestureClassifier

def main():
    # Rutas
    project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
    clips_dir    = os.path.join(project_root, 'data', 'clips')
    model_path   = os.path.join(project_root, 'training', 'model.joblib')

    # Inicializar captura y modelo
    cap       = cv2.VideoCapture(0)
    extractor = KeypointExtractor(maxHands=2)
    recorder  = ClipRecorder(output_dir=clips_dir)
    clf       = GestureClassifier(model_path=model_path)
    clf.load()
    mode = 'translation'

    while True:
        ret, frame = cap.read()
        if not ret: break
        key = cv2.waitKey(1) & 0xFF
        # Cambiar modo: 't' traducción, 'r' grabar, 's' guardar clip, 'q' salir
        if key == ord('t'): mode = 'translation'
        elif key == ord('r'): mode = 'record'; recorder.frames=[]
        elif key == ord('s') and mode=='record': recorder.save_clip()
        elif key == ord('q'): break

        kp = extractor.extract(frame)
        # Solo manos: primeros 168 dims
        feat = kp[:168].reshape(1, -1)
        if mode == 'translation':
            letter = clf.predict(feat)[0]
            cv2.putText(frame, letter, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        else:
            recorder.add_frame(frame)
            cv2.putText(frame, f"Grabando {len(recorder.frames)}/{recorder.max_frames}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),1)
        cv2.imshow('LSP Translator', frame)
    cap.release(); cv2.destroyAllWindows()
```

Este código:
- Carga el modelo entrenado (`model.joblib`).
- Captura frames con `KeypointExtractor`.
- En modo "translation" predice la letra y la muestra en pantalla.
- En modo "record" acumula frames y permite guardar clips de prueba.

---

> _Nota: cada vez que actualices datos o labels, vuelve a ejecutar este script para regenerar `model.joblib`._
