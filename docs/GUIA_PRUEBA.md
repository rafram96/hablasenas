# Guía de Prueba del Pipeline LSP-Traductor

Esta guía explica paso a paso cómo configurar y ejecutar la prueba de extracción, etiquetado y generación de features para tu proyecto de traducción LSP.

---
## 1. Preparación del Entorno

1. Abre una terminal PowerShell en la raíz del proyecto (`LSP-Traductor`).
2. Crea un entorno virtual (si no lo tienes):  
   ```pwsh
   python -m venv venv
   ```
3. Activa el entorno:
   ```pwsh
   venv\Scripts\Activate.ps1
   ```
4. Instala dependencias:
   ```pwsh
   pip install -r requirements.txt
   ```

---
## 2. Captura de Clips

1. Ejecuta la aplicación de captura en tiempo real:
   ```pwsh
   python src\main.py
   ```
2. En la ventana de vídeo:
   - Presiona **`r`** para comenzar a grabar un clip (hasta 50 frames por defecto).
   - Presiona **`s`** para guardar el clip en `data/clips`.
   - Presiona **`q`** para salir.
3. Verifica que en `data/clips` aparezcan archivos `.avi` nombrados como `clip_YYYYMMDD_HHMMSS.avi`.

---
## 3. Procesado de Clips

1. Lanza el script del pipeline (sin argumentos):
   ```pwsh
   python pipeline\process_clip.py
   ```
2. El script:
   - Lee todos los `.avi` de `data/clips`.
   - Extrae landmarks de manos y cara por frame.
   - Guarda un archivo `.npy` con la matriz **(num_frames, num_features)** en `output/`.
   - Genera un vídeo anotado con mallas en `output/annotated/`.
3. Al finalizar verás en la consola rutas de los archivos `.npy` y vídeos anotados.

---
## 4. Etiquetado de Clips o Features

1. Ejecuta el etiquetador:
   ```pwsh
   python pipeline\labeler.py --clips_dir output --out_json data/labels.json
   ```
2. El script mostrará cada archivo (`.avi` o `.npy`) encontrado en `output/` y pedirá la etiqueta (A–Z).
3. Tras completar, se generará `data/labels.json` con el mapeo:
   ```json
   {
     "clip_20250615_201305.npy": "A",
     ...
   }
   ```

---
## 5. Generación del Dataset para Entrenamiento

1. Importa los paquetes necesarios:
   ```python
   import os, json
   import numpy as np
   from src.model import GestureClassifier
   ```
2. Carga las etiquetas:
   ```python
   labels = json.load(open('data/labels.json', 'r', encoding='utf-8'))
   ```
3. Recorre cada `.npy` en `output/`:
   ```python
   X, y = [], []
   for fname, label in labels.items():
       arr = np.load(os.path.join('output', fname))  # shape (T, F)
       # Ejemplo: plano por frame
       for frame_feat in arr:
           X.append(frame_feat)
           y.append(label)
   X = np.stack(X)
   y = np.array(y)
   ```
4. Guarda `X` e `y` si deseas:
   ```python
   np.save('data/X.npy', X)
   np.save('data/y.npy', y)
   ```

---
## 6. Entrenamiento del Modelo

1. En un nuevo script `train.py` (o REPL), crea y entrena el clasificador:
   ```python
   from src.model import GestureClassifier
   import numpy as np

   X = np.load('data/X.npy')
   y = np.load('data/y.npy')
   clf = GestureClassifier(model_path='model.joblib')
   clf.train(X, y)
   ```
2. El modelo entrenado se guardará en `model.joblib`.

---
## 7. Prueba en Tiempo Real

1. Ejecuta la aplicación principal:
   ```pwsh
   python src\main.py
   ```
2. Asegúrate de que `model.joblib` existe al ejecutar.
3. En modo traducción (`t`), la letra predicha se mostrará sobre la imagen.

---
**¡Listo!** Con estos pasos has recorrido todo el pipeline: captura → extracción → etiquetado → entrenamiento → prueba.   
Revisa los archivos generados en `output/` y `data/`, y ajusta hiperparámetros o etiquetas según necesites.  
