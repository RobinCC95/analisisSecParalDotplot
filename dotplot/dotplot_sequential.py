import numpy as np

def dotplot_secuencial(secuencia1, secuencia2):
    dotplot = np.zeros((len(secuencia1), len(secuencia2)))

    for i in range(len(secuencia1)):
        for j in range(len(secuencia2)):
            dotplot[i, j] = 1 if secuencia1[i] == secuencia2[j] else 0

    return dotplot