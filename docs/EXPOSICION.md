# Exposición del Proyecto: LSP-Traductor

## 1. Objetivo y Alcance
El objetivo de *LSP-Traductor* es desarrollar un sistema modular capaz de capturar video en tiempo real de la Lengua de Señas Peruana (LSP), extraer landmarks de manos y rostro, procesar los vectores resultantes y clasificarlos en letras o palabras. Se prioriza la claridad diagnóstica y la explicabilidad del modelo.

## 2. Estructura del Proyecto
```
├── src/               # Código principal (captura, modelo, visualización)
│   ├── capture.py     # Extracción de keypoints con MediaPipe
│   ├── main.py        # Interfaz y bucle en tiempo real
│   ├── model.py       # Definición y entrenamiento del clasificador
│   ├── visualize_face_mesh.py  # Dibujo de malla facial y puntos clave
│   └── clips/         # Grabación y pruebas unitarias
├── pipeline/          # Procesamiento batch y etiquetado
│   ├── process_clip.py
│   ├── labeler.py
│   └── lector.py      # Análisis y descomposición de vectores
├── data/              # Clips de video originales y etiquetas JSON
├── test_output/       # Salidas de prueba (.npy, reportes)
├── docs/              # Documentación y esta exposición
│   ├── GUIDE.md
│   ├── MODELOS.md
│   ├── LANDMARKS_RESUMEN.md
│   └── EXPOSICION.md  # (este archivo)
└── requirements.txt   # Dependencias (MediaPipe, OpenCV, numpy, scikit-learn)
```

## 3. Flujo Principal
1. **Captura**: `capture.py` usa OpenCV y MediaPipe para extraer landmarks de manos y cara.
2. **Grabación de Clips**: `clips/recorder.py` captura secuencias y guarda frames.
3. **Procesamiento Batch**: `pipeline/process_clip.py` convierte clips en vectores `.npy`.
4. **Etiquetado**: `pipeline/labeler.py` guía al usuario para asignar clases a cada clip.
5. **Entrenamiento y Clasificación**: `model.py` entrena un clasificador y `main.py` lo usa en tiempo real.
6. **Visualización y Diagnóstico**: scripts de prueba (`test_pipeline.py`, `test_extractor.py`, `test_analyzer.py`) generan reportes y ayudan a ajustar umbrales.

## 4. Retos y Soluciones
- **Detección Inconsistente**: Variabilidad en iluminación y ángulos provocaba falsos negativos.  
  *Solución*: Ajuste de umbrales de MediaPipe y modos de refinamiento.

- **Dimensionalidad Alta**: 468 landmarks faciales + 42 de manos resultaban en vectores muy grandes.  
  *Solución*: Selección de un subconjunto clave (15–20 puntos) y uso de bit de presencia.

- **Ruido y Oclusiones**: Puntos faltantes o con coordenadas erráticas.  
  *Solución*: Bit de presencia y padding fijo para mantener la estructura del vector.

## 5. Derivación del Apartado de Tests
Para garantizar robustez y facilitar diagnóstico, se construyó un conjunto de pruebas:  

- **test_pipeline.py**: Verifica extracción correcta de frames y guardado de `.npy`.  
- **test_extractor.py**: Asegura el formato y estadísticas básicas de los vectores.  
- **test_analyzer.py**: Permite inspeccionar landmarks por frame, resaltar índices clave y generar reportes JSON.

### 5.1 Diferencias en Captura y Extracción: Tests vs Pipeline Principal

**Pipeline Principal**:
- Procesamiento batch de clips completos (`pipeline/process_clip.py`).
- Extrae landmarks en modo no interactivo, optimizado para volumen de datos.
- Aplica normalización global y padding fijo antes de guardar vectores en disco (.npy).
- Incluye bit de presencia para indicar detección y permite etiquetado masivo.
- Diseñado para producción: eficiencia y consistencia en flujo de datos.

**Tests**:
- Ejecutan extracción en fragmentos o frames específicos para diagnóstico.
- Usan funciones de análisis (`test_analyzer.py`) para imprimir landmarks y ratios de non-zeros.
- No guardan vectores normalizados globalmente, sino que muestran valores crudos en consola.
- Permiten iterar manualmente sobre indices clave y evaluar calidad de detección en tiempo real.
- Enfocados en validación, depuración y ajuste de umbrales, no en volumen de procesamiento.

### 5.2 ¿Por qué el Pipeline Principal da ceros y el Test no?

- **Filtrado de frames**: `test_extractor.py` sólo guarda muestras con un ratio de detección por encima de un umbral, garantizando que cada vector tenga landmarks no nulos. El pipeline principal no filtra y guarda todos los frames, incluyendo aquellos sin detecciones.

- **Umbrales de detección**: En pruebas en vivo (test) el flujo de la cámara suele estar bien iluminado y frontal, facilitando detección de manos y rostro. En los clips de `data/clips`, variables como iluminación, ángulo o calidad del vídeo pueden impedir que MediaPipe supere el `min_detection_confidence` (0.5 en cara), resultando en ausencia de landmarks.

- **Static Image Mode**: Se procesa cada frame como imagen fija, sin aprovechamiento de tracking entre frames. Esto penaliza la detección en secuencias grabadas con variaciones, provocando ceros constantes.

- **Ausencia de normalización intermedia**: Sin una fase de prefiltrado o reprocesado (e.g., ajustar brillo/contraste) el extractor devuelve ceros cuando no detecta, y el pipeline guarda ese vector sin revisión.

Para corregirlo, se recomienda:
1. Filtrar o descartar frames con `np.count_nonzero(vector)==0` antes de guardar.
2. Ajustar `min_detection_confidence` y explorar `refine_landmarks=True` para FaceMesh.
3. Preprocesar vídeo (iluminación, recorte de región de interés) para mejorar detección.

## 6. Resultados y Demonstración
- Generación de archivos `.npy` con dimensiones controladas (p.ej., 64 dimensiones para 16 landmarks con bit de presencia).  
- Informes en `test_output/analysis_report.json` con estadísticas agregadas.  
- Visualización interactiva de la malla facial y puntos clave (script `visualize_face_mesh.py`).

## 7. Conclusiones y Futuro
El sistema ofrece un pipeline completo desde la captura hasta la clasificación y diagnóstico.  
**Próximos pasos**:  
- Integrar análisis visual de los vectores en la UI.  
- Ampliar conjunto de landmarks dinámicos según gestos complejos.  
- Evaluar modelos avanzados (deep learning) para mejorar precisión.

---

*Fin de la exposición del proyecto LSP-Traductor.*
