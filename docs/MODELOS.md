# Comparativa de Modelos: RandomForest vs CNN + LSTM

## RandomForest

**Descripción**
- Ensamble de árboles de decisión entrenados con bagging y selección aleatoria de features.

**Funcionamiento**
1. Cada árbol aprende reglas de partición sobre los vectores de keypoints (x, y, z).
2. Para predecir, cada árbol vota y la clase mayoritaria es la salida.

**Pros**
- Fácil de implementar y entrenar.
- Eficaz en datos de baja dimensión.
- Robusto al sobreajuste (reducción de varianza).
- Interpretable (importancia de features).
- Pocos hiperparámetros.

**Contras**
- No modela relaciones espaciales explícitas.
- No captura dependencias temporales.
- Puede volverse pesado con muchos árboles.

---

## CNN + LSTM

**Descripción**
- Arquitectura híbrida que combina:
  - CNN para extraer patrones espaciales por frame.
  - LSTM para modelar la evolución temporal de dichos patrones.

**Funcionamiento**
1. CNN (CONV→ReLU→Pooling) genera un vector de features espaciales por frame.
2. Secuencia de vectores entra en LSTM, que aprende dependencias temporales.
3. Capa fully-connected + softmax para clasificación final.

**Pros**
- Captura relaciones espaciales y locales.
- Modela dinámicas temporales (útil si hay movimiento).
- Escalable a secuencias largas (palabras/frases).
- Suele ser más preciso que clasificadores estáticos en vídeo.

**Contras**
- Más complejo de implementar y entrenar.
- Requiere muchos datos etiquetados.
- Computacionalmente más costoso (ideal GPU).
- Hiperparámetros sensibles.

---

## Diferencias Clave

| Aspecto                | RandomForest           | CNN + LSTM                          |
|------------------------|------------------------|-------------------------------------|
| Entrada                | Vector plano por frame | Tensor espacial + secuencia temporal|
| Modelado temporal      | Ninguno                | Sí (LSTM)                           |
| Datos necesarios       | Pocos                  | Muchos                              |
| Velocidad de inferencia| Alta (CPU)             | Media-Baja (GPU recomendable)       |


**¿Cuándo elegir?**
- **RandomForest**: prototipo rápido y abecedario estático.
- **CNN + LSTM**: señales con movimiento, palabras o frases.
