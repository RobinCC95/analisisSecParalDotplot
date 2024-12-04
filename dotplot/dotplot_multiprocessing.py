import numpy as np
import multiprocessing as mp
from tqdm import tqdm


def calcular_bloque(args):
    """
    Calcula un bloque de la matriz dotplot entre submatrices de las secuencias.
    """
    i, j, bloque1, bloque2 = args
    submatriz = np.zeros((len(bloque1), len(bloque2)), dtype=np.int32)
    for bi, base1 in enumerate(bloque1):
        for bj, base2 in enumerate(bloque2):
            submatriz[bi, bj] = 1 if base1 == base2 else 0
    return i, j, submatriz


def dotplot_multiprocessing_memmap(secuencia1, secuencia2, output_file='dotplot_memmap_multiprocessing.dat',
                                   num_procesos=mp.cpu_count(), bloque_tamano=1000):
    """
    Calcula el dotplot de dos secuencias utilizando multiprocessing y np.memmap,
    procesando en bloques para evitar saturar la memoria.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        output_file (str): Ruta del archivo memmap donde se guardará el dotplot.
        num_procesos (int): Número de procesos a utilizar.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.

    Returns:
        np.ndarray: Matriz del dotplot almacenada en el archivo memmap.
    """
    len1, len2 = len(secuencia1), len(secuencia2)

    # Crear un archivo memmap para almacenar el dotplot
    dotplot_memmap = np.memmap(output_file, dtype=np.int32, mode='w+', shape=(len1, len2))

    # Generar las tareas en bloques
    tareas = []
    for i in range(0, len1, bloque_tamano):
        for j in range(0, len2, bloque_tamano):
            bloque1 = secuencia1[i:i + bloque_tamano]
            bloque2 = secuencia2[j:j + bloque_tamano]
            tareas.append((i, j, bloque1, bloque2))

    # Crear la barra de progreso
    with tqdm(total=len(tareas), desc="Calculando Dotplot", unit="bloques") as pbar:
        # Usar multiprocessing para procesar los bloques
        with mp.Pool(num_procesos) as pool:
            for i, j, submatriz in pool.imap(calcular_bloque, tareas):
                # Insertar la submatriz calculada en el archivo memmap
                dotplot_memmap[i:i + submatriz.shape[0], j:j + submatriz.shape[1]] = submatriz
                pbar.update(1)  # Actualizar la barra de progreso

    # Después de procesar, se debe hacer un flush para asegurar que los datos se escriben en el disco
    dotplot_memmap.flush()

    return dotplot_memmap
