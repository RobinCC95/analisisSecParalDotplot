from mpi4py import MPI
import numpy as np

def dotplot_mpi(secuencia1, secuencia2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk_size = len(secuencia1) // size
    inicio = rank * chunk_size
    fin = len(secuencia1) if rank == size - 1 else (rank + 1) * chunk_size

    dotplot_local = np.zeros((fin - inicio, len(secuencia2)))

    for i in range(inicio, fin):
        for j in range(len(secuencia2)):
            dotplot_local[i - inicio, j] = 1 if secuencia1[i] == secuencia2[j] else 0

    dotplot = None
    if rank == 0:
        dotplot = np.zeros((len(secuencia1), len(secuencia2)))

    comm.Gather(dotplot_local, dotplot, root=0)

    if rank == 0:
        return dotplot
    return None