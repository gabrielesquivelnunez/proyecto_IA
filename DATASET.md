# DATASET.md — Descripcion del Conjunto de Datos

## Como se recolecto

Las imagenes fueron capturadas manualmente simulando una linea de produccion:

- **Superficie:** Hoja blanca como fondo neutro (simula la banda de produccion).
- **Objetos:**
  - **Positivos (label=1):** Granos de arroz (contaminacion a detectar).
  - **Negativos (label=0):** Aros metalicos, clips, superficie vacia u otros objetos no-arroz.
- **Total por estudiante:** 30 imagenes (15 positivas + 15 negativas).
- **Camara:** Smartphone con camara trasera, foco automatico.

## Variaciones presentes

| Variable       | Detalle                                           |
|----------------|---------------------------------------------------|
| Iluminacion    | Luz natural y artificial, variaciones de sombras  |
| Angulo         | Cenital (~90 grados) con ligeras variaciones      |
| Distancia      | Aproximadamente 20-40 cm de la superficie         |
| Cantidad arroz | 1 a varios granos por imagen positiva             |

## Preprocesamiento aplicado

1. Conversion a escala de grises.
2. Binarizacion con metodo Otsu (umbral calculado automaticamente segun histograma).
3. Conversion de pixeles: blanco (fondo) -> 1, negro (objeto) -> 0.
4. Redimensionado a 128x128 pixeles.
5. Aplanamiento a vector fila de 16,384 valores + columna de etiqueta.

## Dataset combinado del grupo

El modelo fue entrenado con datos de 8 estudiantes:

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

## Proceso de etiquetado

- Etiquetado manual por cada estudiante recolector.
- Criterio: `label=1` si y solo si hay al menos un grano de arroz visible en la imagen.
- `label=0` si no hay arroz, independientemente de si hay otros objetos.
- No se uso herramienta de anotacion externa; la etiqueta se asigno por carpeta.

## Limitaciones

- Variaciones de iluminacion entre estudiantes pueden afectar la binarizacion.
- Dataset pequeno (240 muestras total); puede no generalizar bien a entornos
  industriales reales.
- Dependencia del fondo blanco: el modelo es sensible a cambios de superficie.
- Objetos muy pequenos, parcialmente ocluidos o desenfocados pueden no ser
  detectados correctamente.
- Cada estudiante uso su propio dispositivo movil, introduciendo variabilidad
  en resolucion, color y nitidez de las imagenes.
- Sin revision cruzada del etiquetado; posible inconsistencia entre estudiantes.
