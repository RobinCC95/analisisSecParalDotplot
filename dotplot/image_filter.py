import numpy as np

def filtrar_diagonales(dotplot, umbral=5):
    diagonales = []
    for d in range(-dotplot.shape[0] + 1, dotplot.shape[1]):
        diagonal = np.diag(dotplot, k=d)
        if np.sum(diagonal) >= umbral:
            diagonales.append((d, diagonal))
    return diagonales