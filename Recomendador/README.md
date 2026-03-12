# Sistema de Recomendación de Productos

## Descripción

Este proyecto desarrolla y compara distintos enfoques de sistemas de
recomendación para predecir qué productos pueden resultar relevantes
para cada usuario a partir de valoraciones históricas.

Se implementan métodos basados en popularidad, filtrado colaborativo
(user-user e item-item mediante KNN) y factorización matricial mediante
SVD. Los modelos se evalúan utilizando métricas como RMSE, Precision@k y
Recall@k.

Los resultados muestran que el modelo basado en factorización matricial
optimizado logra el mejor desempeño general, capturando de forma más
efectiva los patrones de preferencia de los usuarios.

## Tecnologías utilizadas

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Surprise
-   Matplotlib
-   Seaborn

## Estructura del repositorio

-   data/ → dataset utilizado\
-   notebooks/ → análisis exploratorio y modelado\
-   models/ → modelos entrenados\
-   results/ → métricas y evaluaciones

## Objetivo académico

Proyecto realizado en el marco de la especialización en **Análisis de
Datos e Inteligencia Artificial**, aplicando técnicas de machine
learning a sistemas de recomendación.
