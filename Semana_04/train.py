
import os
import pandas as pd
import kagglehub
import argparse

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 1. Cargar los datos
def cargar_datos():
    """Descarga y carga el dataset Bank Marketing."""

    path = kagglehub.dataset_download(
        "henriqueyamahata/bank-marketing"
    )

    archivo = os.path.join(path, "bank-additional-full.csv")

    DF = pd.read_csv(archivo, sep=";")

    return DF


# 2. Limpiar los datos
def limpiar_datos(DF):
    """Elimina duplicados y conserva las demás observaciones."""

    DF = DF.drop_duplicates().reset_index(drop=True)

    return DF


# 3. Preparar las variables
def preparar_datos(DF):
    """Separa predictores y objetivo, excluyendo duration."""

    y = DF["y"]

    X = DF.drop(columns=["y", "duration"])

    return X, y


# 4. Dividir los datos
def dividir_datos(X, y,seed):
    """Divide los datos en entrenamiento y prueba."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# 5. Crear el pipeline
def crear_pipeline(X_train, args):
    """Construye el preprocesamiento y la regresión logística."""

    columnas_categoricas = X_train.select_dtypes(include=["object"]).columns
    columnas_numericas = X_train.select_dtypes(include=["int64", "float64"]).columns

    preprocesador = ColumnTransformer(
        transformers=[("categoricas", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
                      ("numericas", StandardScaler(), columnas_numericas)])

    peso_clases = None if args.class_weight == "none" else "balanced"

    modelo = Pipeline(
        steps=[("preprocesamiento", preprocesador),
               ("clasificador", LogisticRegression(C=args.C, max_iter=args.max_iter,
                                                   class_weight=peso_clases, random_state=args.seed))])
    return modelo
# 6. Entrenar el modelo
def entrenar_modelo(modelo, X_train, y_train):
    """Entrena el pipeline utilizando los datos de entrenamiento."""

    modelo.fit(X_train, y_train)

    return modelo


# 7. Evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    """Calcula las métricas de clasificación."""

    y_pred = modelo.predict(X_test)

    # Identificar la posición de la clase positiva
    clases = list(modelo.named_steps["clasificador"].classes_)
    indice_yes = clases.index("yes")

    y_prob = modelo.predict_proba(X_test)[:, indice_yes]

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, pos_label="yes"
        ),
        "recall": recall_score(
            y_test, y_pred, pos_label="yes"
        ),
        "f1_score": f1_score(
            y_test, y_pred, pos_label="yes"
        ),
        "roc_auc": roc_auc_score(
            (y_test == "yes").astype(int),
            y_prob
        )
    }

    print("\nMétricas del modelo:")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(
        y_test, y_pred, labels=["no", "yes"]
    ))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    return metricas

#Método que nos permite obtener los paramétros del modelo
def obtener_argumentos():
    """Define los hiperparámetros configurables desde la terminal."""

    parser = argparse.ArgumentParser(
        description="Entrenamiento de regresión logística para Bank Marketing")
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la regularización. Predeterminado: 1.0")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Máximo de iteraciones. Predeterminado: 1000")
    parser.add_argument(
        "--class_weight",
        type=str,
        choices=["none", "balanced"],
        default="none",
        help="Ponderación de clases: none o balanced")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria. Predeterminado: 42")
    args = parser.parse_args()

    if args.C <= 0:
        parser.error("--C debe ser mayor que cero")

    if args.max_iter <= 0:
        parser.error("--max_iter debe ser mayor que cero")

    return args

def guardar_matriz_confusion(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred, labels=["no", "yes"])
    fig, ax = plt.subplots()
    imagen = ax.imshow(matriz, cmap="Blues")
    #Identifica los ejes
    ax.set_xticks([0, 1], labels=["no", "yes"])
    ax.set_yticks([0, 1], labels=["no", "yes"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión")
    #Agregar valores númericos
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matriz[i, j], ha="center", va="center")
    #Guardar imagen
    fig.colorbar(imagen, ax=ax)
    fig.savefig("matriz_confusion.png", bbox_inches="tight")
    plt.close(fig)



# 8. Función principal
def main():
    args = obtener_argumentos()
    #Registro del experimento
    mlflow.set_experiment("Bank_Marketing_Hyperparameter_Tuning")

    print("Iniciando entrenamiento del modelo baseline...")
    print(f"\nConfiguración: C={args.C}, max_iter={args.max_iter}, class_weight={args.class_weight}, seed={args.seed}")

    DF = cargar_datos()
    print(f"Datos originales: {DF.shape}")

    DF = limpiar_datos(DF)
    print(f"Datos después de limpieza: {DF.shape}")

    X, y = preparar_datos(DF)
    X_train, X_test, y_train, y_test = dividir_datos(X, y, args.seed)

    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")

    #Inicio de corrida de MLflow
    with mlflow.start_run(run_name=f"LR_C{args.C}_{args.class_weight}_seed{args.seed}"):
        # Registrar los hiperparámetros
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("seed", args.seed)

        modelo = crear_pipeline(X_train, args)
        modelo = entrenar_modelo(modelo, X_train, y_train)
        metricas = evaluar_modelo(modelo, X_test, y_test)

        # Registrar métricas
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)
	#Guardar el artefacto. En este caso es la matriz de confusión
        guardar_matriz_confusion(modelo, X_test, y_test)
        mlflow.log_artifact("matriz_confusion.png")
	#Guardar el experimento}
        mlflow.sklearn.log_model(sk_model=modelo, name="modelo_bank_marketing",serialization_format="cloudpickle")
    print("\nEntrenamiento y evaluación finalizados.")

if __name__ == "__main__":
    main()


