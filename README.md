# Proyecto_OlyGiant
# Proyecto de Predicción de Pozos Petrolíferos para OilyGiant

## Introducción

En el competitivo sector de la extracción de petróleo, la identificación precisa de ubicaciones prometedoras para la perforación de nuevos pozos es crucial para maximizar la rentabilidad y minimizar los riesgos financieros. La compañía OilyGiant busca optimizar sus inversiones al determinar las regiones con mayor potencial de producción. Este proyecto de ciencia de datos aborda este desafío mediante el análisis de datos geológicos y la aplicación de técnicas de modelado predictivo y evaluación de riesgos.

## Objetivos

El objetivo principal de este proyecto es identificar la región más rentable para la apertura de 200 nuevos pozos petrolíferos para OilyGiant, considerando tanto el volumen de reservas estimado como el riesgo de incurrir en pérdidas. Los objetivos específicos incluyen:

1. **Carga y Preprocesamiento de Datos:** Leer y preparar los datos de exploración geológica de tres regiones, incluyendo la verificación de la calidad y la comprensión de las características disponibles.
2. **Construcción de un Modelo Predictivo:** Desarrollar un modelo de regresión lineal para predecir el volumen de reservas de petróleo en nuevos pozos basándose en las características geológicas proporcionadas.
3. **Evaluación del Potencial de Reservas:** Utilizar el modelo para estimar el volumen de reservas en un conjunto de puntos de exploración en cada región y seleccionar los 200 pozos con las mayores reservas estimadas.
4. **Estimación de Beneficio:** Calcular el beneficio potencial para cada región, considerando los ingresos por barril y el costo de perforación de los pozos seleccionados.
5. **Análisis de Riesgo:** Evaluar el riesgo de pérdidas en cada región mediante la técnica de bootstrapping, proporcionando una estimación de la probabilidad de obtener ganancias negativas.
6. **Selección de la Región Óptima:** Seleccionar la región que ofrezca el mayor beneficio promedio con un riesgo de pérdida aceptable (inferior al 2.5%).

## Descripción de Datos

Los datos de exploración geológica de las tres regiones se almacenan en archivos CSV. Cada archivo contiene las siguientes columnas:

* `id`: Identificador único del pozo petrolífero.
* `f0`, `f1`, `f2`: Tres características de los puntos de exploración geológica. El significado específico de estas características es desconocido, pero se consideran significativas para la predicción.
* `product`: Volumen de reservas de petróleo en el pozo petrolífero, expresado en miles de barriles.

## Pasos Realizados

1. **Carga de Datos:** Se cargaron los datos de exploración geológica de tres regiones, almacenados en los archivos `geo_data_0.csv`, `geo_data_1.csv` y `geo_data_2.csv` utilizando la librería pandas.

2. **Exploración Inicial de Datos (EDA):** Se realizó un análisis exploratorio de datos para cada conjunto de datos con la función `exploration(df)`. Esto incluyó:
    * Información general del DataFrame (`df.info()`).
    * Verificación de valores faltantes (`df.isnull().sum()`).
    * Detección de duplicados (`df.duplicated().sum()`).
    * Estadísticas descriptivas de las características (`df.describe()`).


3. **Entrenamiento del Modelo de Regresión Lineal:** Se definió la función `regresion(df, obj, id)` para entrenar un modelo de regresión lineal para predecir la variable objetivo (`product`) utilizando las tres características (`f0`, `f1`, `f2`). Para cada región:
    * Se dividieron los datos en conjuntos de entrenamiento y prueba (75% entrenamiento, 25% prueba) utilizando `train_test_split` de scikit-learn.
    * Se entrenó un modelo de `LinearRegression`.
    * Se realizaron predicciones sobre el conjunto de prueba.
    * Se calculó el Root Mean Squared Error (RMSE) para evaluar la precisión del modelo.

4. **Evaluación del Modelo:** Se imprimieron los RMSE obtenidos para cada región:

5. **Cálculo de Beneficio Esperado:** Se definió la función `ganancia(prediccion, y_test, count)` para calcular el beneficio potencial basado en las predicciones y los datos de prueba. Se seleccionaron los `WELLS_TO_DRILL` (200) pozos con las predicciones más altas. Se utilizaron los siguientes parámetros:
 * `BUDGET = 100000000` (Presupuesto para el desarrollo de 200 pozos).
 * `INCOME_PER_UNIT = 4500` (Ingreso por unidad de producto, en miles de barriles).
6. **Resultados Iniciales de Beneficio:** Se calcularon e imprimieron las ganancias esperadas para cada región:

7. **Análisis de Riesgo con Bootstrapping:** Se implementó la función `boot_ganancia(prediccion, y_test, n_bootstraps)` para realizar un análisis de riesgo utilizando la técnica de bootstrapping. Para cada región:
 * Se generaron `n_bootstraps` (2000 en este caso) muestras de los datos de prueba con reemplazo.
 * Se calculó el beneficio para cada muestra.
 * Se calcularon el beneficio promedio, los cuantiles del 2.5% y 97.5% para obtener un intervalo de confianza del 95%, y la probabilidad de pérdida (porcentaje de veces que la ganancia fue negativa).
8. **Resultados del Análisis de Riesgo:** Se imprimieron los resultados del análisis de bootstrapping para cada región:

## Archivos Utilizados

* `geo_data_0.csv`
* `geo_data_1.csv`
* `geo_data_2.csv`

## Requisitos

* Python 3.x
* Librerías de Python:
 * pandas
 * numpy
 * matplotlib
 * seaborn
 * scikit-learn

## Instrucciones para Ejecutar el Código

1. Asegúrate de tener instaladas las librerías necesarias. Puedes instalarlas usando pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn