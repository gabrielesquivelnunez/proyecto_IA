import cv2
import numpy as np
import os

# 1. Configuración de carpetas basada en tu nuevo árbol (tree)
# Ahora buscamos dentro de la carpeta 'fotos'
BASE_PATH = 'fotos'
CATEGORIAS = {
    'arroz': 1,  # Positivo
    'aros': 0    # Negativo
}
OUTPUT_FOLDER = 'procesadas'
IMG_SIZE = 128

def preparar_datos_gabriel():
    lista_vectores = []
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("--- Iniciando preprocesamiento corregido ---")

    for subcarpeta, etiqueta in CATEGORIAS.items():
        # La ruta ahora es fotos/arroz o fotos/aros
        ruta_completa = os.path.join(BASE_PATH, subcarpeta)
        
        if not os.path.exists(ruta_completa):
            print(f"⚠️ Carpeta '{ruta_completa}' no encontrada. Saltando...")
            continue
        
        archivos = [f for f in os.listdir(ruta_completa) if f.lower().endswith('.png')]
        print(f"Procesando {len(archivos)} imágenes en '{ruta_completa}'...")

        for filename in archivos:
            path = os.path.join(ruta_completa, filename)
            
            # 1. Leer y Grises
            img = cv2.imread(path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Redimensionar
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            
            # 3. Suavizado (Gaussian Blur)
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            
            # 4. UMBRAL ADAPTATIVO (Limpieza de sombras)
            binarizada = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 4
            )
            
            # 5. Limpieza morfológica
            kernel = np.ones((2,2), np.uint8)
            binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)
            
            # 6. Normalizar a 0 y 1 (1=Fondo, 0=Objeto)
            matriz_final = (binarizada / 255).astype(int)
            
            # 7. Aplanar y Guardar etiqueta
            vector_fila = matriz_final.flatten()
            registro = np.append(vector_fila, etiqueta)
            lista_vectores.append(registro)
            
            # Guardar imagen procesada para control visual
            # Se guardará como: procesadas/bin_arroz_IMG_XXXX.png
            nombre_salida = f"bin_{subcarpeta}_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, nombre_salida), binarizada)

    # 8. Guardar el CSV final
    if lista_vectores:
        dataset = np.array(lista_vectores)
        nombre_csv = "dataset.csv"
        np.savetxt(nombre_csv, dataset, delimiter=",", fmt='%d')
        
        media = np.mean(dataset[:, :-1])
        print(f"\n--- Auditoría Final ---")
        print(f"Media de píxeles: {media:.4f}")
        print(f"Archivo '{nombre_csv}' generado exitosamente con {len(lista_vectores)} muestras.")
    else:
        print("❌ Error: No se pudo procesar ninguna imagen. Revisa las carpetas.")

if __name__ == "__main__":
    preparar_datos_gabriel()
