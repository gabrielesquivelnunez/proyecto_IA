"""
Proyecto 1 IE0435 — Entrenamiento y evaluación de modelos de clasificación
==========================================================================
Lee dataset.csv propio + todos los CSV de la carpeta 'datasets/' de compañeros,
entrena 4 modelos clásicos, muestra resumen de métricas y exporta el mejor
modelo en formato .joblib.

ESTRUCTURA ESPERADA:
    proyecto_IA_IE0435/
    ├── dataset.csv              ← tu dataset
    ├── datasets/                ← datasets de compañeros
    │   ├── dataset_companero1.csv
    │   └── dataset_companero2.csv
    └── entrenar_modelos.py

USO:
    python entrenar_modelos.py
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
CSV_PROPIO       = "dataset.csv"   # Tu dataset en la raíz del proyecto
CARPETA_EXTRAS   = "datasets"      # Carpeta con datasets de compañeros
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
NOMBRE_MODELO    = "C22799_gabriel_esquivel_nunez.joblib"
N_PIXELES        = 16384           # 128 × 128
# ──────────────────────────────────────────────────────────────────────────────


def leer_csv_flexible(archivo):
    """
    Lee un CSV sin importar si tiene encabezado o no, y estandariza
    a N_PIXELES columnas de píxeles + columna 'label' al final.
    """
    with open(archivo, 'r') as f:
        primera_celda = f.readline().strip().split(',')[0]

    try:
        float(primera_celda)
        df = pd.read_csv(archivo, header=None)
    except ValueError:
        df = pd.read_csv(archivo)

    df = df.dropna()
    n_cols = df.shape[1]

    if n_cols == N_PIXELES + 1:
        df.columns = [f"p{i}" for i in range(N_PIXELES)] + ["label"]
        return df
    elif n_cols == N_PIXELES:
        print(f"  ⚠  Sin columna de etiqueta ({n_cols} cols), se omite")
        return None
    else:
        print(f"  ⚠  Dimensión inesperada {df.shape}, se omite")
        return None


def cargar_todos_los_datasets():
    """Carga el CSV propio + todos los de la carpeta datasets/."""
    dfs = []

    # 1. Cargar dataset propio
    if os.path.exists(CSV_PROPIO):
        try:
            df = leer_csv_flexible(CSV_PROPIO)
            if df is not None:
                dfs.append(df)
                print(f"  ✓ {CSV_PROPIO:45s} → {df.shape[0]} muestras")
        except Exception as e:
            print(f"  ✗ {CSV_PROPIO} → Error: {e}")
    else:
        print(f"  ⚠  No se encontró {CSV_PROPIO}")

    # 2. Cargar datasets de compañeros
    if os.path.isdir(CARPETA_EXTRAS):
        for archivo in sorted(glob.glob(os.path.join(CARPETA_EXTRAS, "*.csv"))):
            try:
                df = leer_csv_flexible(archivo)
                if df is not None:
                    dfs.append(df)
                    print(f"  ✓ {os.path.basename(archivo):45s} → {df.shape[0]} muestras")
            except Exception as e:
                print(f"  ✗ {os.path.basename(archivo)} → Error: {e}")
    else:
        print(f"  ℹ  Carpeta '{CARPETA_EXTRAS}/' no encontrada, se usa solo el dataset propio")

    if not dfs:
        raise ValueError("No se pudo cargar ningún dataset.")

    datos = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total combinado: {datos.shape[0]} muestras × {datos.shape[1]} columnas")
    return datos


def preparar_datos(datos):
    """Separa features y etiqueta, divide en train/test."""
    X = datos.iloc[:, :-1].values.astype(float)
    y = datos.iloc[:, -1].values.astype(int)

    print(f"\n  Positivos (arroz):     {int(y.sum())}")
    print(f"  Negativos (sin arroz): {int(len(y) - y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
    return X_train, X_test, y_train, y_test


def definir_modelos():
    """Define los 4 modelos con sus grillas de hiperparámetros."""
    return {
        "KNN": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            "params": {
                "clf__n_neighbors": [3, 5, 7, 9],
                "clf__weights":     ["uniform", "distance"],
                "clf__metric":      ["euclidean", "manhattan"]
            }
        },
        "Decision Tree": {
            "pipeline": Pipeline([
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__max_depth":        [3, 5, 10, None],
                "clf__min_samples_split":[2, 5, 10],
                "clf__criterion":        ["gini", "entropy"]
            }
        },
        "Naive Bayes": {
            "pipeline": Pipeline([
                ("clf", GaussianNB())
            ]),
            "params": {
                "clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]
            }
        },
        "SVM": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__C":      [0.1, 1, 10],
                "clf__kernel": ["linear", "rbf"],
                "clf__gamma":  ["scale", "auto"]
            }
        }
    }


def entrenar_y_evaluar(modelos, X_train, X_test, y_train, y_test):
    """Entrena cada modelo con GridSearchCV y evalúa en test."""
    resultados    = []
    mejores_modelos = {}

    for nombre, config in modelos.items():
        print(f"\n  Entrenando {nombre}...")
        grid = GridSearchCV(
            config["pipeline"],
            config["params"],
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        mejor = grid.best_estimator_
        mejores_modelos[nombre] = mejor

        y_pred = mejor.predict(X_test)
        resultados.append({
            "modelo":    nombre,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        })
        print(f"     Mejores params : {grid.best_params_}")
        print(f"     F1 (CV)        : {grid.best_score_:.4f}")

    return resultados, mejores_modelos


def mostrar_resumen(resultados):
    """Muestra la tabla de resultados ordenada por F1."""
    df = pd.DataFrame(resultados).sort_values("f1_score", ascending=False).reset_index(drop=True)
    sep = "=" * 68
    print(f"\n{sep}")
    print("RESUMEN DE RESULTADOS")
    print(sep)
    print(df.to_string(index=True, float_format="{:.6f}".format))
    print(sep)
    return df


def exportar_modelos(df_resultados, mejores_modelos):
    """
    Exporta todos los modelos con su nombre en formato .joblib.
    El mejor también se exporta con el nombre carne_nombre_apellido.joblib.
    """
    CARPETA_MODELOS = "modelos"
    os.makedirs(CARPETA_MODELOS, exist_ok=True)

    nombres_archivo = {
        "KNN":           "KNN.joblib",
        "Decision Tree": "Arbol_Decision.joblib",
        "Naive Bayes":   "Naive_Bayes.joblib",
        "SVM":           "SVM_Lineal.joblib",
    }

    print(f"\n  Guardando todos los modelos en '{CARPETA_MODELOS}/'...")
    for nombre, modelo in mejores_modelos.items():
        nombre_archivo = nombres_archivo.get(nombre, f"{nombre.replace(' ', '_')}.joblib")
        ruta = os.path.join(CARPETA_MODELOS, nombre_archivo)
        joblib.dump(modelo, ruta)
        print(f"  ✓ {nombre_archivo}")

    mejor_nombre = df_resultados.iloc[0]["modelo"]
    mejor_modelo = mejores_modelos[mejor_nombre]
    joblib.dump(mejor_modelo, NOMBRE_MODELO)

    print(f"\n  🏆 Mejor modelo : {mejor_nombre}")
    print(f"  📦 Entrega      : {NOMBRE_MODELO}")
    print(f"  F1-score        : {df_resultados.iloc[0]['f1_score']:.6f}")
    print(f"  Accuracy        : {df_resultados.iloc[0]['accuracy']:.6f}")


def main():
    print("=" * 68)
    print("  PROYECTO 1 IE0435 — ENTRENAMIENTO DE MODELOS")
    print("=" * 68)

    print(f"\n📂 Cargando datasets...")
    datos = cargar_todos_los_datasets()

    print("\n📊 Distribución del dataset:")
    X_train, X_test, y_train, y_test = preparar_datos(datos)

    print("\n🤖 Entrenando modelos (GridSearchCV 5-fold)...")
    modelos = definir_modelos()
    resultados, mejores_modelos = entrenar_y_evaluar(
        modelos, X_train, X_test, y_train, y_test
    )

    df_resultados = mostrar_resumen(resultados)

    print("\n💾 Exportando mejor modelo...")
    exportar_modelos(df_resultados, mejores_modelos)

    print("\n✅ ¡Listo!")


if __name__ == "__main__":
    main()
