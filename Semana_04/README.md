    # Bank Marketing - Experimentación con MLflow

## 1. Descripción

Este proyecto tiene como objetivo entrenar y evaluar un modelo de regresión logística para predecir si un cliente aceptará una campaña de marketing bancario.

Se utiliza MLflow para registrar los hiperparámetros, las métricas de evaluación, la matriz de confusión y el modelo entrenado.

## 2. Dataset

Se utiliza el conjunto de datos **Bank Marketing**, disponible en Kaggle:

https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

El archivo utilizado es `bank-additional-full.csv`. El programa descarga los datos mediante `kagglehub`, elimina registros duplicados y excluye la variable `duration` para evitar fuga de información.

## 3. Requisitos

- Ubuntu o WSL2.
- Python 3.14.4.
- pip y venv.
- Bibliotecas especificadas en `requirements.txt`.

## 4. Instalación

Desde la carpeta del proyecto, crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## 5. Entrenamiento

Ejecutar el modelo con los parámetros predeterminados:

```bash
python train.py
```

También se pueden modificar los hiperparámetros desde la terminal:

```bash
python train.py --C 10.0 --max_iter 1000 --class_weight balanced --seed 42
```

Los argumentos disponibles son:

- `--C`: controla la regularización de la regresión logística.
- `--max_iter`: número máximo de iteraciones.
- `--class_weight`: permite utilizar `none` o `balanced`.
- `--seed`: semilla aleatoria utilizada para dividir los datos en entrenamiento y prueba. Su valor predeterminado es 42, y se mantuvo constante en los cinco experimentos para facilitar la comparación y reproducibilidad de los resultados.

## 6. Seguimiento con MLflow

Para visualizar los experimentos, iniciar MLflow desde la carpeta del proyecto:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir en el navegador:

http://localhost:5000

Seleccionar el experimento `Bank_Marketing_Hyperparameter_Tuning`.

En MLflow se pueden consultar los parámetros, las métricas, los artefactos y los modelos generados durante las ejecuciones.

## 7. Resultados

Se realizaron cinco configuraciones de hiperparámetros. La configuración con `C=10.0` y `class_weight=balanced` obtuvo el mayor F1-score observado, aproximadamente 0.4618.

El balanceo de clases tuvo mayor influencia sobre el recall y el F1-score que las modificaciones realizadas al parámetro `C`.
    
