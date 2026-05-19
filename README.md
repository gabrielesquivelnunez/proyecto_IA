# Proyecto 1 IE0435 — Clasificacion de Contaminaciones en Linea de Produccion

**Estudiante:** Gabriel Esquivel Nunez — C22799
**Curso:** IE0435 Inteligencia Artificial — I-2026
**Universidad de Costa Rica**

---

## Descripcion

Pipeline completo para detectar contaminaciones (granos de arroz) en una linea
de produccion simulada. El flujo va desde fotos crudas tomadas con el celular
hasta un modelo de clasificacion entrenado y exportado.

---

## Estructura del repositorio

```
proyecto_IA_IE0435/
├── datasets/                        # Datasets de companeros del grupo
│   ├── dataset_carlos_naranjo.csv
│   ├── dataset_cristopher.csv
│   ├── dataset_daniel_valverde.csv
│   ├── dataset_danna.csv
│   ├── dataset_felipe.csv
│   ├── dataset_ignacio_montenegro.csv
│   └── dataset_sheyla.csv
│
├── fotos/                           # Fotos originales tomadas con el celular
│   ├── aros/                        # Fotos sin arroz (label = 0)
│   └── arroz/                       # Fotos con arroz (label = 1)
│
├── modelos/                         # Todos los modelos entrenados
│   ├── Arbol_Decision.joblib
│   ├── KNN.joblib
│   ├── Naive_Bayes.joblib
│   └── SVM_Lineal.joblib
│
├── procesadas/                      # Imagenes preprocesadas (128x128, grises)
│
├── reports/                         # Informe final del proyecto
│
├── dataset.csv                      # Dataset propio (30 muestras)
├── entrenar_modelos.py              # Script de entrenamiento
├── procesar_2.py                    # Script de preprocesamiento y vectorizacion
├── C22799_gabriel_esquivel_nunez.joblib  # Mejor modelo (entrega)
├── DATASET.md
├── LICENSE
├── MODEL_CARD.md
├── README.md
└── requirements.txt
```

---

## Flujo de trabajo

### Paso 1 — Preprocesar fotos y generar dataset (`procesar_2.py`)

Las fotos en `fotos/arroz/` y `fotos/aros/` fueron tomadas con camara de
celular sobre una hoja blanca. El script las procesa automaticamente:

1. Convierte cada imagen a escala de grises.
2. Aplica binarizacion adaptativa con metodo Otsu.
3. Redimensiona a 128x128 pixeles.
4. Aplana la matriz a un vector de 16,384 valores.
5. Agrega la etiqueta (1 = arroz, 0 = no arroz).
6. Guarda todo en `dataset.csv`.

```bash
python3 procesar_2.py
```

### Paso 2 — Entrenar modelos (`entrenar_modelos.py`)

Carga el dataset propio y los de companeros en `datasets/`, entrena 4 modelos
con busqueda de hiperparametros y exporta el mejor.

```bash
python3 entrenar_modelos.py
```

Modelos evaluados: KNN, Arbol de Decision, Naive Bayes, SVM.
La seleccion se basa en el F1-score (split 80/20, random_state=42).

**Resultados:**

| Modelo        | Accuracy | Precision | Recall | F1-score   |
|---------------|----------|-----------|--------|------------|
| SVM           | 0.6458   | 0.5897    | 0.9583 | **0.7302** |
| Decision Tree | 0.5625   | 0.6000    | 0.3750 | 0.4615     |
| KNN           | 0.5417   | 0.5714    | 0.3333 | 0.4211     |
| Naive Bayes   | 0.4792   | 0.4667    | 0.2917 | 0.3590     |

**Modelo seleccionado:** SVM (`C=1`, `kernel=rbf`, `gamma=scale`)

---

## Inferencia con el modelo exportado

```python
import joblib
import cv2
import numpy as np

# Cargar el modelo
modelo = joblib.load("C22799_gabriel_esquivel_nunez.joblib")

# Preparar una imagen
img = cv2.imread("imagen.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
_, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
vector = (binaria / 255).astype(int).flatten().reshape(1, -1)

# Predecir
prediccion = modelo.predict(vector)
print("Arroz detectado" if prediccion[0] == 1 else "Sin arroz")
```

---

## Instalacion de dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Autor

**Gabriel Esquivel Nunez** — C22799
Ingenieria Electrica, Universidad de Costa Rica
