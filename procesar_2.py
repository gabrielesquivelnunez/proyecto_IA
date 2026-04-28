import cv2
import numpy as np
import os

# 1. Configuración de TUS carpetas según el comando tree
CATEGORIAS = {
    'conArroz': 1,  # Positivo
    'sinArroz': 0   # Negativo
}
OUTPUT_FOLDER = 'procesadas_limpias'
IMG_SIZE = 128

def preparar_datos_gabriel():
    lista_vectores = []
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("--- Iniciando preprocesamiento con carpetas conArroz/sinArroz ---")

    for carpeta, etiqueta in CATEGORIAS.items():
        if not os.path.exists(carpeta):
            print(f"Carpeta '{carpeta}' no encontrada. Revisa la ruta.")
            continue
        
        archivos = [f for f in os.listdir(carpeta) if f.lower().endswith('.png')]
        print(f"Procesando {len(archivos)} imágenes en '{carpeta}'...")

        for filename in archivos:
            path = os.path.join(carpeta, filename)
            
            # 1. Leer y Grises
            img = cv2.imread(path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Redimensionar primero
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            
            # 3. Suavizado para eliminar ruido de sensores
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            
            # 4. UMBRAL ADAPTATIVO (Soluciona sombras en esquinas)
            # 11 es el tamaño del bloque, 4 es la constante restada
            binarizada = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 4
            )
            
            # 5. Limpieza morfológica (quita puntitos negros sueltos)
            kernel = np.ones((2,2), np.uint8)
            binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel)
            
            # 6. Convertir a 0 y 1 (1=Fondo, 0=Objeto)
            matriz_final = (binarizada / 255).astype(int)
            
            # 7. Aplanar y Guardar
            vector_fila = matriz_final.flatten()
            registro = np.append(vector_fila, etiqueta)
            lista_vectores.append(registro)
            
            # Guardar visual para que veas qué tan limpia quedó
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"test_{carpeta}_{filename}"), binarizada)

    # 8. Generar CSV final
    if lista_vectores:
        dataset = np.array(lista_vectores)
        nombre_csv = "dataset.csv"
        np.savetxt(nombre_csv, dataset, delimiter=",", fmt='%d')
        
        media = np.mean(dataset[:, :-1])
        print(f"\n--- Auditoría Final ---")
        print(f"Media de píxeles: {media:.4f} (Debería ser > 0.90)")
        print(f"Archivo '{nombre_csv}' actualizado con {len(lista_vectores)} muestras.")
    else:
        print("Error: No se procesó ninguna imagen.")

if __name__ == "__main__":
    preparar_datos_gabriel()
