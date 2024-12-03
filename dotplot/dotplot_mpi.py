from mpi4py import MPI
import numpy as np
from tqdm import tqdm

def dotplot_mpi(secuencia1, secuencia2, bloque_tamano=100):
    """
    Calcula el dotplot de dos secuencias utilizando MPI, procesando por lotes y mostrando una barra de progreso.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.

    Returns:
        np.ndarray: Matriz del dotplot (solo en el proceso root).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Dividir la secuencia1 entre los procesos
    chunk_size = len(secuencia1) // size
    inicio = rank * chunk_size
    fin = len(secuencia1) if rank == size - 1 else (rank + 1) * chunk_size

    # Crear la matriz local para el dotplot
    dotplot_local = np.zeros((fin - inicio, len(secuencia2)), dtype=np.int8)

    # Barra de progreso para cada proceso
    total_bloques = (fin - inicio) // bloque_tamano + (1 if (fin - inicio) % bloque_tamano != 0 else 0)
    with tqdm(total=total_bloques, desc=f"Proceso {rank}", unit="bloques", position=rank) as pbar:
        for i in range(inicio, fin, bloque_tamano):
            bloque1 = secuencia1[i:i + bloque_tamano]
            for j in range(0, len(secuencia2), bloque_tamano):
                bloque2 = secuencia2[j:j + bloque_tamano]

                # Crear una submatriz para el bloque actual
                submatriz = np.zeros((len(bloque1), len(bloque2)), dtype=np.int8)

                # Calcular el dotplot para el bloque actual
                for bi, base1 in enumerate(bloque1):
                    for bj, base2 in enumerate(bloque2):
                        submatriz[bi, bj] = 1 if base1 == base2 else 0

                # Insertar la submatriz en la matriz local
                dotplot_local[i - inicio:i - inicio + len(bloque1), j:j + len(bloque2)] = submatriz

            # Actualizar la barra de progreso
            pbar.update(1)

    # Reunir los resultados en el proceso root
    dotplot = None
    if rank == 0:
        dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.int8)

    comm.Gather(dotplot_local, dotplot, root=0)

    # Solo el proceso root devuelve el resultado
    if rank == 0:
        return dotplot
    return None