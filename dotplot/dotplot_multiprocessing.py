import numpy as np
import multiprocessing as mp

def comparar_indices(args):
    i, secuencia1, secuencia2 = args
    return [1 if secuencia1[i] == secuencia2[j] else 0 for j in range(len(secuencia2))]

def dotplot_multiprocessing(secuencia1, secuencia2, num_procesos=mp.cpu_count()):
    with mp.Pool(num_procesos) as pool:
        resultados = pool.map(
            comparar_indices, [(i, secuencia1, secuencia2) for i in range(len(secuencia1))]
        )

    dotplot = np.array(resultados)
    return dotplot