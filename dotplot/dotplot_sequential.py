import numpy as np
from tqdm import tqdm

def dotplot_secuencial(secuencia1, secuencia2, bloque_tamano=100):
    """
    Calcula el dotplot de dos secuencias en bloques para evitar problemas de memoria,
    mostrando una barra de progreso.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.

    Returns:
        np.ndarray: Matriz del dotplot.
    """
    len1, len2 = len(secuencia1), len(secuencia2)
    dotplot = np.zeros((len1, len2), dtype=np.int8)  # Usa np.int8 para ahorrar memoria

    # Calcular el número total de bloques para la barra de progreso
    total_bloques = (len1 // bloque_tamano + (1 if len1 % bloque_tamano != 0 else 0)) * \
                    (len2 // bloque_tamano + (1 if len2 % bloque_tamano != 0 else 0))

    # Procesar en bloques con barra de progreso
    with tqdm(total=total_bloques, desc="Calculando Dotplot", unit="bloques") as pbar:
        for i in range(0, len1, bloque_tamano):
            for j in range(0, len2, bloque_tamano):
                # Definir los límites del bloque
                bloque1 = secuencia1[i:i + bloque_tamano]
                bloque2 = secuencia2[j:j + bloque_tamano]

                # Crear una submatriz para el bloque actual
                submatriz = np.zeros((len(bloque1), len(bloque2)), dtype=np.int8)

                # Calcular el dotplot para el bloque actual
                for bi, base1 in enumerate(bloque1):
                    for bj, base2 in enumerate(bloque2):
                        submatriz[bi, bj] = 1 if base1 == base2 else 0

                # Insertar la submatriz en la matriz principal
                dotplot[i:i + len(bloque1), j:j + len(bloque2)] = submatriz

                # Actualizar la barra de progreso
                pbar.update(1)

    return dotplot