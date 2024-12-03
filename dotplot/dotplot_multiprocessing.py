import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def comparar_indices(args):
    """
    Compara un índice de la secuencia1 con todos los índices de la secuencia2.
    """
    i, secuencia1, secuencia2 = args
    return [1 if secuencia1[i] == secuencia2[j] else 0 for j in range(len(secuencia2))]

def dotplot_multiprocessing(secuencia1, secuencia2, num_procesos=mp.cpu_count(), bloque_tamano=100):
    """
    Calcula el dotplot de dos secuencias utilizando multiprocessing, procesando por lotes y mostrando una barra de progreso.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        num_procesos (int): Número de procesos a utilizar.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.

    Returns:
        np.ndarray: Matriz del dotplot.
    """
    # Dividir la secuencia1 en bloques
    bloques = [(i, secuencia1, secuencia2) for i in range(len(secuencia1))]

    # Crear la barra de progreso
    with tqdm(total=len(bloques), desc="Calculando Dotplot", unit="líneas") as pbar:
        # Usar multiprocessing para procesar los bloques
        with mp.Pool(num_procesos) as pool:
            resultados = []
            for resultado in pool.imap(comparar_indices, bloques, chunksize=bloque_tamano):
                resultados.append(resultado)
                pbar.update(1)  # Actualizar la barra de progreso

    # Convertir los resultados en una matriz numpy
    dotplot = np.array(resultados, dtype=np.int8)
    return dotplot