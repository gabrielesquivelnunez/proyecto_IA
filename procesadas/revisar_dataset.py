import numpy as np

def auditar_dataset(archivo):
    print(f"--- Auditoría de: {archivo} ---")
    try:
        # Cargar el archivo
        data = np.loadtxt(archivo, delimiter=",")
        
        # 1. Verificar Dimensiones
        filas, columnas = data.shape
        print(f"Filas detectadas: {filas} {'[OK]' if filas == 30 else '[ERROR]'}")
        print(f"Columnas detectadas: {columnas} {'[OK]' if columnas == 16385 else '[ERROR]'}")
        
        # 2. Análisis de Etiquetas (Última columna)
        etiquetas = data[:, -1]
        clase_1 = np.sum(etiquetas == 1)
        clase_0 = np.sum(etiquetas == 0)
        print(f"Muestras de Arroz (1): {clase_1}")
        print(f"Muestras de Aros/Clips (0): {clase_0}")
        
        # 3. Análisis de Píxeles (Primeras 16384 columnas)
        pixeles = data[:, :-1]
        media_pixeles = np.mean(pixeles)
        
        # Un valor cercano a 1 significa fondo blanco mayoritario
        # Un valor de 0 o 1 exacto indicaría un fallo (imagen toda negra o toda blanca)
        print(f"Media de valores de píxeles: {media_pixeles:.4f}")
        
        if 0 < media_pixeles < 1:
            print("Estado de píxeles: VARIABILIDAD DETECTADA [OK]")
        else:
            print("Estado de píxeles: ERROR (Valores constantes detectados)")
            
        # 4. Verificación de valores únicos
        valores_unicos = np.unique(pixeles)
        print(f"Valores encontrados en píxeles: {valores_unicos}")

    except Exception as e:
        print(f"Error crítico: {e}")

if __name__ == "__main__":
    auditar_dataset("dataset.csv")
