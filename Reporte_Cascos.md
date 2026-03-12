# Detección de uso de cascos de seguridad en entornos laborales  
## Clasificación binaria con Red Neuronal Convolucional (CNN)

**Curso:** Deep Learning  
**Fecha:** 26 de febrero de 2026  

---

## 1. Introducción

### Problema y motivación

Según la Organización Internacional del Trabajo (OIT), al menos 60.000 trabajadores mueren cada año en accidentes en obras de construcción en todo el mundo. En Uruguay, según datos de la Inspección General del Trabajo y de la Seguridad Social (IGTSS), se registran aproximadamente 40.000 accidentes laborales por año, de los cuales 6.000 corresponden al sector de la construcción. En 2025, el 33% de los fallecimientos laborales ocurrió en dicho sector, siendo las caídas desde altura la principal causa de muerte. Un número significativo de estos accidentes se vincula al no uso del Equipo de Protección Personal (EPP) requerido, particularmente cascos de seguridad.

Este proyecto desarrolla un sistema basado en deep learning capaz de **clasificar automáticamente si los trabajadores en obras de construcción están usando cascos de seguridad**. Se entrena una **Red Neuronal Convolucional (CNN)** para clasificación binaria: *con casco* vs *sin casco*. Dicho sistema podría integrarse con cámaras de vigilancia existentes para proveer monitoreo y alertas en tiempo real, reduciendo el riesgo de accidentes fatales.

---

## 2. Solución propuesta

La solución consiste en una CNN de clasificación binaria que recibe como entrada recortes de personas/cabezas extraídos de imágenes de obras de construcción y predice si la persona lleva casco o no.

**Pipeline completo:**

1. Parseo de anotaciones en formato XML (Pascal VOC) para extraer bounding boxes
2. Recorte de cada región de interés desde la imagen original
3. Agrupación de las clases originales en 2 categorías binarias: *con casco* y *sin casco*
4. Redimensionamiento de recortes a 64×64 píxeles y normalización al rango [0, 1]
5. Data augmentation (flips horizontales, variaciones de brillo y contraste) sobre el conjunto de entrenamiento
6. Entrenamiento de una CNN con bloques convolucionales, BatchNormalization, MaxPooling y Dropout
7. Evaluación con métricas de clasificación: accuracy, precision, recall, F1-score y AUC-ROC

El enfoque de trabajar sobre recortes pre-anotados (en lugar de detección extremo a extremo) simplifica el problema y permite obtener resultados de alta calidad con una arquitectura relativamente compacta. En un sistema de producción, esta CNN se combinaría con un detector de objetos (p. ej. YOLO) que localice automáticamente a las personas en la imagen completa.

---

## 3. Descripción de los datos

### Fuente y contenido

Se utiliza el dataset **"Hard Hat Detection"** disponible públicamente en Kaggle (Andrew Mvd, [https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)). El dataset consta de **5.000 imágenes** con anotaciones en formato XML (Pascal VOC) que contienen instancias de las siguientes clases:

| Clase          | Instancias |
|----------------|:----------:|
| helmet         | 18.966     |
| head           | 5.785      |
| person         | 751        |
| **Total**      | **25.502** |

### Mapeo a clasificación binaria

Las clases originales se agrupan en dos categorías:

- **Con casco (1):** `helmet`
- **Sin casco (0):** `head`, `person`

### Preprocesamiento y partición

Del total de anotaciones, se aplicó un filtro mínimo de 16×16 píxeles para descartar bounding boxes demasiado pequeñas (se descartaron 3.987 recortes). El resultado fue un conjunto de **19.187 recortes** con la siguiente distribución:

| Partición  | Muestras | Porcentaje | Con casco |
|------------|:--------:|:----------:|:---------:|
| Train      | 13.437   | 70,0%      | 78,5%     |
| Validación | 2.871    | 15,0%      | 78,5%     |
| Test       | 2.879    | 15,0%      | 78,5%     |

La partición se realizó con estratificación para preservar la proporción de clases en cada subconjunto. El desbalance entre clases (78,5% con casco vs 21,5% sin casco) refleja la distribución natural del dataset.

**Data augmentation** (solo en entrenamiento): flips horizontales aleatorios y variaciones aleatorias de brillo y contraste, implementados como capa `tf.data` para eficiencia.

---

## 4. Métricas de evaluación

Para evaluar la solución se utilizan las siguientes métricas:

- **Accuracy:** Proporción de predicciones correctas sobre el total. Útil como referencia general, pero puede ser engañosa con clases desbalanceadas.

- **Precision:** De todos los recortes que el modelo predijo como "con casco", qué fracción realmente tenía casco. Relevante para minimizar falsas alarmas. Definición: *TP / (TP + FP)*.

- **Recall (Sensibilidad):** De todos los recortes que realmente no tenían casco, qué fracción detectó correctamente el modelo. Crucial en seguridad laboral: un falso negativo (persona sin casco que el sistema no alerta) es el error más peligroso. Definición: *TP / (TP + FN)*.

- **F1-Score:** Media armónica de Precision y Recall. Balancea ambas métricas, especialmente útil con clases desbalanceadas. Definición: *2 × (Precision × Recall) / (Precision + Recall)*.

- **AUC-ROC:** Área bajo la curva ROC (Receiver Operating Characteristic). Mide la capacidad del modelo de discriminar entre clases independientemente del umbral de decisión. Un valor de 1.0 es perfecto; 0.5 es equivalente a clasificación aleatoria.

La métrica más crítica para este dominio es el **recall de la clase "sin casco"**, ya que los falsos negativos (personas sin casco no detectadas) representan el mayor riesgo en seguridad laboral.

---

## 5. Arquitectura del modelo

Se diseñó una CNN personalizada de clasificación binaria compuesta por **4 bloques convolucionales** con filtros crecientes:

| Bloque | Filtros | Operaciones                                    | Salida       |
|--------|:-------:|------------------------------------------------|:------------:|
| 1      | 32      | Conv2D(3×3) → BatchNorm → ReLU → MaxPool       | 32×32×32     |
| 2      | 64      | Conv2D(3×3) → BatchNorm → ReLU → MaxPool       | 16×16×64     |
| 3      | 128     | Conv2D(3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.3) | 8×8×128 |
| 4      | 256     | Conv2D(3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.4) | 4×4×256 |
| Head   | —       | GlobalAvgPool → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid) | 1 |

**Total de parámetros: 423.361 (1,61 MB)**

Se eligió **Global Average Pooling** en lugar de Flatten para reducir la cantidad de parámetros en la cabeza clasificadora y disminuir el riesgo de sobreajuste. El **Dropout progresivo** (0.3 → 0.4 → 0.5) actúa como regularizador en las capas más profundas. La **BatchNormalization** en cada bloque estabiliza el entrenamiento y acelera la convergencia.

**Configuración de entrenamiento:**
- Función de pérdida: Binary Cross-Entropy
- Optimizador: Adam (lr inicial = 1×10⁻⁴)
- Épocas: 50 (con early stopping)
- Batch size: 32
- Callbacks: `EarlyStopping` (patience=10), `ReduceLROnPlateau` (patience=5, factor=0.5), `ModelCheckpoint` (guarda el mejor modelo según val_AUC)

---

## 6. Resultados y discusión

### Resultados en el conjunto de test

| Métrica   | Valor  |
|-----------|:------:|
| Loss      | 0,0639 |
| Accuracy  | 97,95% |
| Precision | 99,02% |
| Recall    | 98,36% |
| AUC-ROC   | 0,9943 |

**Reporte de clasificación por clase:**

|            | Precision | Recall | F1-Score | Soporte |
|------------|:---------:|:------:|:--------:|:-------:|
| Sin casco  | 0,94      | 0,96   | 0,95     | 618     |
| Con casco  | 0,99      | 0,98   | 0,99     | 2.261   |
| Macro avg  | 0,97      | 0,97   | 0,97     | 2.879   |

### Reflexión sobre las métricas

- La **accuracy del 97,95%** confirma que el modelo clasifica correctamente la gran mayoría de los recortes. Sin embargo, dado el desbalance de clases, esta métrica por sí sola no es suficiente.
- El **recall de "sin casco" es 0,96**: el modelo detecta correctamente el 96% de las personas sin protección. El 4% restante representa **falsos negativos**, el error más peligroso en este contexto.
- La **precision de "sin casco" es 0,94**: cuando el modelo alerta que alguien no tiene casco, acierta el 94% de las veces.
- El **AUC-ROC de 0,9943** (muy cercano a 1,0) indica excelente capacidad de discriminación entre clases independientemente del umbral de decisión.

### Análisis de sobreajuste (overfitting)

Las curvas de entrenamiento mostraron convergencia estable: la accuracy de validación (~97,7%) se mantuvo cercana a la de entrenamiento (~98,5%) a lo largo de las 50 épocas, sin divergencia significativa. La val_loss se estabilizó alrededor de 0,06 mientras que la train_loss descendió hasta ~0,04.

El learning rate fue reducido automáticamente tres veces por el callback `ReduceLROnPlateau` (en las épocas 23, 40 y 46), permitiendo al modelo refinar los pesos sin oscilaciones. El early stopping no se activó, lo que indica que el modelo seguía mejorando marginalmente hasta la época 50 (los mejores pesos del modelo corresponden a la época 49).

**Estrategias anti-sobreajuste implementadas:**
- Dropout progresivo (0,3 / 0,4 / 0,5) en las capas más profundas
- BatchNormalization en cada bloque convolucional
- Data augmentation en entrenamiento
- Early stopping (patience=10 épocas)
- ReduceLROnPlateau para ajuste fino del learning rate
- Global Average Pooling en lugar de Flatten (menor cantidad de parámetros)

### Principales dificultades y limitaciones

1. **Variabilidad en tamaño de bounding boxes:** recortes muy pequeños pierden información relevante al redimensionarse a 64×64 píxeles. Se descartaron 3.987 bounding boxes por ser menores a 16×16.
2. **Límite de recortes por imagen (MAX=10):** se impuso para poder ejecutar en Google Colab con RAM limitada (~12 GB), reduciendo la cantidad total de datos de entrenamiento disponibles.
3. **Clasificación sin detección:** el modelo clasifica recortes individuales pre-anotados, no localiza personas en imágenes completas. Para un sistema en producción, sería necesario combinar esta CNN con un detector de objetos (YOLO, Faster R-CNN).
4. **Desbalance de clases:** 78,5% de las muestras corresponden a "con casco", lo que puede sesgar el modelo hacia esa clase. Se monitorizó el recall de "sin casco" específicamente para mitigar este efecto.

---

## 7. Conclusiones

Se implementó una CNN de clasificación binaria compacta (423.361 parámetros) para detectar el uso de cascos de seguridad en imágenes de entornos de construcción, alcanzando un **accuracy del 97,95%, precision del 99,02%, recall del 98,36% y AUC-ROC de 0,9943** en el conjunto de test. El F1-score fue de 0,95 para la clase "sin casco" y 0,99 para "con casco".

El modelo demostró ser altamente efectivo para clasificar recortes individuales de personas con y sin casco, con un entrenamiento estable y sin señales de sobreajuste significativo. Las estrategias de regularización implementadas (Dropout, BatchNormalization, Data Augmentation, ReduceLROnPlateau) contribuyeron a la generalización del modelo.

La principal limitación del sistema es que opera sobre recortes pre-anotados y no localiza personas automáticamente en imágenes completas. Como **trabajo futuro**, la integración con un detector de objetos end-to-end (como YOLOv8) permitiría un pipeline completo capaz de detectar y clasificar personas en tiempo real a partir de imágenes de cámaras de vigilancia en obras de construcción, brindando alertas automáticas ante incumplimientos del uso de EPP.

---

*Código disponible: `Cascos_editado.ipynb` (ejecutable en Google Colab)*  
*Dataset: [https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)*
