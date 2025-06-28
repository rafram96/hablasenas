# Resumen de Landmarks Faciales Clave y Justificación Teórica

## 1. Introducción
Este documento explica la selección ampliada de puntos de referencia (landmarks) faciales y su función en el sistema de traducción de LSP a texto. Presenta una visión teórica y de alto nivel sobre cómo estos landmarks soportan la estimación de orientación, expresiones y la generación de vectores de características.

## 1.1 Interpretación de Coordenadas (X, Y, Z)

Los landmarks devueltos por MediaPipe incluyen tres valores:
- **X**: posición horizontal normalizada en el rango [0,1], con 0 en el borde izquierdo del frame y 1 en el borde derecho.
- **Y**: posición vertical normalizada en el rango [0,1], con 0 en el borde superior del frame y 1 en el borde inferior.
- **Z**: profundidad relativa al plano de la cámara, en unidades normalizadas (negativo hacia dentro de la imagen). Valores cercanos a 0 indican que el punto está cerca del plano de la cámara; valores más negativos indican mayor distancia.

Estas coordenadas permiten reconstruir la posición 3D aproximada de cada landmark respecto al sensor de la cámara.

---

## 2. Grupos de Landmarks y sus Roles

1. **Puente y punta de la nariz**
   - Índices 1, 4 (puente) y 2, 5 (punta)
   - Estiman el **pitch** (inclinación hacia arriba/abajo) y el centro geométrico del rostro.
   - Sirven de ancla para normalizar el resto de puntos.

2. **Frente**
   - Índice 10
   - Detecta movimientos de la cabeza hacia atrás o adelante.

3. **Ojos internos**
   - Índices 33, 133
   - Definen el ancho interpupilar para estimar el **yaw** (giro horizontal) y el **roll** (inclinación lateral).

4. **Cejas**
   - Índices 55, 65 (superiores) y 285, 295 (inferiores)
   - Capturan gestos de ceja, útiles para distinguir emociones o acentos en señas expresivas.

5. **Pómulos**
   - Índices 93, 323
   - Mejoran la estimación de rotación (roll) y añaden contexto de forma facial amplia.

6. **Boca**
   - Índices 61, 291
   - Reflejan apertura y forma de la boca, clave para signos que involucran movimientos labiales.

7. **Mentón**
   - Índice 199
   - Ajusta la estimación de longitud y ángulo inferior del rostro, afinando la posición vertical.

---

## 3. Funcionamiento de Alto Nivel

1. **Captura de Frames**: Con OpenCV y MediaPipe se extraen los landmarks de cada frame.
2. **Construcción del Vector de Características**: Se organizan los valores `(x, y, z)` de cada punto seleccionado en un vector de dimensión reducida.
3. **Normalización y Padding**: Se aplican transformaciones para uniformizar escala y posición. Puntos no detectados llevan un bit de presencia.
4. **Clasificación / Modelado**: Un clasificador (p.ej., SVM, RandomForest) usa estos vectores para reconocer letras o palabras.
5. **Análisis y Diagnóstico**: Herramientas de test generan reportes de non-zeros, estadísticas y permiten inspeccionar frames específicos.

---

## 3.1 Dimensionalidad del Vector de Características

Cada punto facial seleccionado aporta tres valores `(x, y, z)` al vector de características. Con `M` landmarks:
- **Sin bit de presencia**: el tamaño del vector es `3 * M`.
- **Con bit de presencia**: se añade un cuarto valor por punto (indicador de detección), resultando en `4 * M` dimensiones.

Por ejemplo, para `M = 16` puntos:
- `3 * 16 = 48` dimensiones (coordenadas pura)
- `4 * 16 = 64` dimensiones (incluyendo bit de presencia)

Este esquema garantiza que el clasificador reciba información tanto de la posición espacial como de la fiabilidad de detección de cada landmark.

---

## 4. Beneficios de la Selección Ampliada
- **Reducción de ruido**: Se elige un subconjunto informativo, reduciendo la dimensionalidad.
- **Robustez ante oclusiones**: Diversos puntos cubren distintas zonas faciales.
- **Mejor precisión en orientación y expresividad**: Combinar puentes nasales, ojos, cejas y boca refina el modelo.

---

Con esta estructura de landmarks, el sistema equilibra entre claridad diagnóstica y rendimiento del clasificador, facilitando la explicación pedagógica y la capacidad de ajuste en condiciones reales de captura.
