# Detección de Veracidad en Declaraciones Políticas

Valentina Ghan

## 1. Introducción

La creciente circulación de información en medios digitales y redes sociales ha incrementado la exposición a contenido engañoso o desinformación. En el ámbito político, la verificación manual de declaraciones resulta costosa y difícil de escalar.

Este proyecto busca desarrollar un modelo de clasificación automática capaz de predecir el nivel de veracidad de declaraciones políticas utilizando técnicas de procesamiento de lenguaje natural (NLP).

## 2. Visión general de la solución

La solución se basa en la aplicación de modelos de aprendizaje automático para clasificar declaraciones políticas según su veracidad. Para representar el contenido textual se utiliza la técnica TF-IDF, que permite transformar el texto en vectores numéricos que capturan la relevancia de las palabras dentro del corpus.

Sobre estas representaciones se entrenan distintos modelos de clasificación supervisada, incluyendo Logistic Regression, Support Vector Machines (SVM) y Naive Bayes.

## 3. Dataset

El proyecto utiliza el **LIAR Dataset**, un conjunto de datos ampliamente utilizado en investigación sobre detección de desinformación. El dataset contiene miles de declaraciones políticas verificadas por la organización PolitiFact y clasificadas en distintas categorías de veracidad.

Cada registro incluye el texto de la declaración y variables contextuales como el autor de la afirmación, su afiliación política y el contexto en que fue realizada.

## 4. Metodología

El pipeline de modelado incluye las siguientes etapas:

- Limpieza y preprocesamiento del texto
- Tokenización y normalización
- Transformación mediante TF-IDF
- Entrenamiento de modelos de clasificación

Para evaluar el desempeño se utilizan métricas estándar de clasificación como precisión, recall y F1-score. También se aplica validación cruzada para asegurar la robustez de los resultados.

## 5. Resultados

Entre los modelos evaluados, **Logistic Regression combinada con TF-IDF** mostró el mejor equilibrio entre precisión y recall. El modelo logró identificar correctamente gran parte de las declaraciones engañosas manteniendo un nivel adecuado de precisión en las predicciones.

Estos resultados evidencian que las técnicas de procesamiento de lenguaje natural pueden utilizarse para apoyar procesos de detección automatizada de desinformación.

## 6. Conclusiones

El proyecto demuestra que los métodos de machine learning aplicados a texto permiten desarrollar herramientas capaces de analizar la veracidad de afirmaciones políticas.

Aunque estos modelos no sustituyen el fact-checking humano, pueden funcionar como sistemas de apoyo que ayuden a priorizar contenido potencialmente engañoso para su posterior verificación.
