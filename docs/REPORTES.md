# Guía de Reportes Generados por `analyzer.py`

Este documento explica el contenido, la estructura y la interpretación de los informes JSON que se generan al analizar archivos `.npy` con el módulo `pipeline/analyzer.py`.

---

## 1. Ubicación de los reportes

Cada reporte se guarda en la carpeta:

```
data/reports/<etiqueta>/
```

- `<etiqueta>` corresponde a la etiqueta asociada al gesto o palabra en LSP.
- Cada archivo se llama `<base>_report.json`, donde `<base>` es el nombre original del archivo `.npy` sin extensión.


## 2. Estructura del JSON

Un ejemplo de un reporte global:

```json
{
  "archivo_npy": "a_20250628_010141.npy",
  "forma": [30, 1380],
  "ratio_no_ceros_global": 0.1234,
  "fluidez_media": 0.5678,
  "fluidez_desviacion": 0.9101
}
```

### Campos principales

- **archivo_npy**: Nombre del archivo `.npy` analizado.
- **forma**: Dimensiones del arreglo NumPy resultante (`[n_frames, n_características]`).
- **ratio_no_ceros_global**: Proporción de valores distintos de cero en todo el arreglo (`count_nonzero / total_size`). Indica cuántos frames o landmarks están activos.
- **fluidez_media**: Media de distancias euclidianas entre landmarks de manos en frames consecutivos. Mide qué tan suaves son los movimientos.
- **fluidez_desviacion**: Desviación estándar de las mismas distancias. Sirve para detectar variaciones o picos bruscos en la fluidez.


## 3. Interpretación de las métricas

- Un **ratio_no_ceros_global** muy bajo sugiere que la captura tiene muchos valores en cero (posibles pérdidas de detección).
- Una **fluidez_media** alta implica movimientos de manos más amplios o rápidos entre frames.
- Una **fluidez_desviacion** alta indica variabilidad en la suavidad de los movimientos (picos o interrupciones).

**Uso práctico**:
- **Depuración de datos**: detectar gestos mal capturados o con pérdida de landmarks.
- **Calidad de grabación**: asegurar que los movimientos sean suaves y consistentes.
- **Filtro previo al entrenamiento**: descartar muestras con fluidez excesivamente baja o muy errática.


## 4. Análisis de un frame específico

Después de generar el JSON, `analyzer.py` muestra en consola un desglose detallado del último frame (o uno específico).
- **Landmarks de manos**: coordenadas `(x, y, z)` y presencia
- **Landmarks faciales clave**: índices de nariz, ojos, labios, etc.

Estos datos permiten inspeccionar visualmente el patrón de landmarks en un instante.


## 5. Buenas prácticas

1. Verifica que la carpeta `data/reports/<etiqueta>/` existe tras la ejecución.
2. Revisa el campo `ratio_no_ceros_global` para identificar capturas con datos faltantes.
3. Examina la fluidez para filtrar movimientos atípicos antes de entrenar.

---

*Última actualización: 28 de junio de 2025*
