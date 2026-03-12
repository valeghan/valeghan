# Sistema de Recomendación de Productos

Valentina Ghan

## 1. Introducción

Las plataformas digitales ofrecen a los usuarios una gran cantidad de productos y contenido, lo que puede dificultar la identificación de opciones relevantes. En este contexto, los sistemas de recomendación cumplen un rol clave al filtrar información y sugerir ítems personalizados en función de las preferencias de los usuarios.

Este proyecto tiene como objetivo desarrollar y comparar distintos enfoques de sistemas de recomendación basados en valoraciones de usuarios. El trabajo se enfoca en modelos de filtrado colaborativo capaces de aprender patrones de comportamiento a partir de interacciones históricas entre usuarios y productos.

## 2. Visión general de la solución

La solución propuesta consiste en la implementación y comparación de diferentes estrategias de recomendación. En primer lugar se utiliza un modelo baseline basado en popularidad. Posteriormente se aplican técnicas de filtrado colaborativo basadas en similitud entre usuarios y entre ítems mediante algoritmos K-Nearest Neighbors (KNN).

Finalmente se implementa un modelo basado en factorización matricial utilizando Singular Value Decomposition (SVD). Este enfoque permite capturar factores latentes que representan preferencias implícitas de los usuarios y características de los productos.

## 3. Dataset

El dataset utilizado contiene valoraciones de usuarios sobre distintos productos. Cada observación representa una interacción usuario-producto con un rating asociado.

Antes del modelado se realiza un proceso de limpieza y filtrado para eliminar usuarios e ítems con muy pocas interacciones, reduciendo la sparsity del dataset y mejorando la calidad del entrenamiento.

## 4. Metodología

El análisis comienza con un estudio exploratorio del dataset para comprender la distribución de ratings, la actividad de los usuarios y el número de interacciones por producto.

Posteriormente se entrenan distintos modelos de recomendación utilizando la librería Surprise. Los modelos se evalúan mediante validación cruzada y métricas como:

- RMSE (Root Mean Squared Error)
- Precision@k
- Recall@k

También se realiza optimización de hiperparámetros mediante Grid Search para mejorar el desempeño de los modelos.

## 5. Resultados

Los resultados muestran que el modelo basado en factorización matricial (SVD) obtiene el mejor desempeño general. Este enfoque logra reducir el error de predicción y generar recomendaciones más precisas en comparación con los métodos basados únicamente en similitud.

La incorporación de factores latentes permite capturar patrones de preferencia más complejos, mejorando la capacidad del sistema para recomendar productos relevantes a cada usuario.

## 6. Conclusiones

Este proyecto demuestra el potencial de los sistemas de recomendación basados en filtrado colaborativo para personalizar la experiencia de los usuarios en plataformas digitales.

Los resultados sugieren que los modelos de factorización matricial representan una alternativa eficiente para capturar patrones de comportamiento en datasets de interacción usuario-producto, constituyendo una base sólida para el desarrollo de sistemas de recomendación a mayor escala.
