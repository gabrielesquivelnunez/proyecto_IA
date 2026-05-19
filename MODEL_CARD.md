# MODEL_CARD.md

## Model name + version

**Clasificador de Contaminaciones en Linea de Produccion v1.0**
Proyecto 1 — IE0435 Inteligencia Artificial, I-2026
Estudiante: Gabriel Esquivel Nunez — C22799

---

## Intended use

**Uso previsto:**
Clasificacion binaria de imagenes de una linea de produccion simulada para
detectar la presencia de granos de arroz (contaminacion).

**Fuera del alcance:**
- Imagenes de entornos industriales reales.
- Deteccion de multiples tipos de contaminantes simultaneamente.
- Imagenes con fondos distintos a superficie blanca.
- Inferencia en tiempo real sobre video.

---

## Data summary

- **Recoleccion:** Imagenes tomadas manualmente con smartphone sobre hoja blanca.
- **Tamano:** 240 imagenes de 8 estudiantes (30 por estudiante: 15 positivas + 15 negativas).
- **Variaciones:** Diferentes condiciones de luz, angulos y cantidades de arroz.
- **Formato final:** Vectores de 16,384 valores binarios (128x128 px) + etiqueta.

| Estudiante            | Muestras | Positivos | Negativos |
|-----------------------|----------|-----------|-----------|
| Gabriel Esquivel      | 30       | 15        | 15        |
| Carlos Naranjo        | 30       | 15        | 15        |
| Cristopher            | 30       | 15        | 15        |
| Daniel Valverde       | 30       | 15        | 15        |
| Danna                 | 30       | 15        | 15        |
| Felipe                | 30       | 15        | 15        |
| Ignacio Montenegro    | 30       | 15        | 15        |
| Sheyla Miller         | 30       | 15        | 15        |
| **Total**             | **240**  | **120**   | **120**   |

---

## Labeling process

- Etiquetado manual por cada estudiante recolector.
- `label = 1`: presencia de al menos un grano de arroz en la imagen.
- `label = 0`: ausencia de arroz (puede haber otros objetos como aros o clips).
- Sin revision de segunda persona; posible sesgo de etiquetado individual.

---

## Metrics

Se evaluaron 4 modelos con validacion cruzada de 5 particiones (GridSearchCV)
y split 80/20 con semilla fija (random_state=42):

| Modelo        | Accuracy | Precision | Recall   | F1-score   |
|---------------|----------|-----------|----------|------------|
| SVM           | 0.6458   | 0.5897    | 0.9583   | **0.7302** |
| Decision Tree | 0.5625   | 0.6000    | 0.3750   | 0.4615     |
| KNN           | 0.5417   | 0.5714    | 0.3333   | 0.4211     |
| Naive Bayes   | 0.4792   | 0.4667    | 0.2917   | 0.3590     |

**Modelo seleccionado:** SVM (mayor F1-score: 0.7302)
**Hiperparametros optimos:** `C=1`, `kernel=rbf`, `gamma=scale`
**Metrica principal:** F1-score — balancea precision y recall, adecuado para
deteccion de contaminaciones donde tanto los falsos positivos como los falsos
negativos tienen impacto en la calidad del proceso.

---

## Ethical / safety notes

- **Sesgo por iluminacion:** El modelo puede fallar con condiciones de luz muy
  diferentes a las del entrenamiento.
- **Sesgo por camara:** Entrenado con imagenes de smartphones; puede no
  generalizar a camaras industriales.
- **Sesgo de fondo:** Entrenado solo sobre fondo blanco; sensible a cambios
  de superficie.
- **Dataset pequeno:** 240 muestras es insuficiente para aplicaciones criticas
  de seguridad industrial.

---

## Limitations

- Granos de arroz muy pequenos o fuera de foco pueden no ser detectados.
- Alta dimensionalidad del vector (16,384) puede causar sobreajuste.
- No se aplico data augmentation.
- El modelo no distingue cantidad de arroz, solo presencia o ausencia.
- Rendimiento moderado (accuracy 64.6%) refleja la variabilidad entre datasets
  de diferentes estudiantes con distintas condiciones de captura.

---

## Reproducibility

```bash
# 1. Crear entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Preprocesar imagenes
python3 procesar_2.py

# 3. Entrenar y exportar modelo
python3 entrenar_modelos.py
```

**Modelo exportado:** `C22799_gabriel_esquivel_nunez.joblib`
**Hardware usado:** Computadora personal, CPU, sin GPU requerida.
**SO:** Ubuntu Linux
**Python:** 3.12
**Semilla aleatoria:** `random_state=42`
