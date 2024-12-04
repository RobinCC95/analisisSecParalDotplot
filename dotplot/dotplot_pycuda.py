import os
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from tqdm import tqdm

# Código CUDA
mod = SourceModule("""
__global__ void generar_dotplot(char *sec1, char *sec2, int *dotplot, int len1, int len2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < len1 && j < len2) {
        dotplot[i * len2 + j] = (sec1[i] == sec2[j]) ? 1 : 0;
    }
}
""")

# Función para convertir secuencias de caracteres a valores numéricos
def convertir_secuencia_a_numeros(secuencia):
    mapa = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return np.array([mapa[base] for base in secuencia], dtype=np.byte)

# Función principal para generar el dotplot con memmap
def dotplot_pycuda_memmap(secuencia1, secuencia2, output_file='dotplot_memmap.dat', bloque_tamano=500, subbloque_tamano=100):
    """
    Calcula el dotplot de dos secuencias utilizando PyCUDA y guarda directamente el resultado en un archivo memmap.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        output_file (str): Archivo donde se almacenará la matriz resultante.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.
        subbloque_tamano (int): Tamaño del subbloque para dividir los bloques.

    Returns:
        np.memmap: Objeto memmap que representa el dotplot.
    """
    # Convertir las secuencias a valores numéricos
    sec1_numerica = convertir_secuencia_a_numeros(secuencia1)
    sec2_numerica = convertir_secuencia_a_numeros(secuencia2)

    len1, len2 = len(secuencia1), len(secuencia2)

    # Crear un archivo memmap para almacenar los resultados
    dotplot = np.memmap(output_file, dtype=np.int32, mode='w+', shape=(len1, len2))

    # Configurar el tamaño de los bloques y la cuadrícula
    block_size = (32, 32, 1)

    # Barra de progreso
    total_bloques = (len(secuencia1) // bloque_tamano + (1 if len(secuencia1) % bloque_tamano != 0 else 0))
    with tqdm(total=total_bloques, desc="Calculando Dotplot", unit="bloques") as pbar:
        for i in range(0, len(secuencia1), bloque_tamano):
            # Dividir la secuencia1 en bloques
            bloque1 = sec1_numerica[i:i + bloque_tamano]
            len_bloque1 = len(bloque1)

            # Dividir el bloque en subbloques para manejar la memoria
            for j in range(0, len(secuencia2), subbloque_tamano):
                subbloque2 = sec2_numerica[j:j + subbloque_tamano]
                len_subbloque2 = len(subbloque2)

                # Transferir los datos del subbloque a la GPU
                sec1_gpu = gpuarray.to_gpu(bloque1)
                sec2_gpu = gpuarray.to_gpu(subbloque2)

                # Crear una submatriz para el subbloque actual en la GPU
                dotplot_gpu = gpuarray.zeros((len_bloque1, len_subbloque2), dtype=np.int32)

                # Configurar la cuadrícula para el subbloque actual
                grid_size = (
                    (len_bloque1 + block_size[0] - 1) // block_size[0],
                    (len_subbloque2 + block_size[1] - 1) // block_size[1],
                )

                # Obtener la función CUDA
                func = mod.get_function("generar_dotplot")

                # Ejecutar el kernel CUDA
                func(
                    sec1_gpu, sec2_gpu, dotplot_gpu,
                    np.int32(len_bloque1), np.int32(len_subbloque2),
                    block=block_size, grid=grid_size,
                )

                # Copiar el resultado del subbloque desde la GPU a la CPU
                dotplot_bloque = dotplot_gpu.get()

                # Escribir el subbloque directamente en el archivo memmap
                dotplot[i:i + len_bloque1, j:j + len_subbloque2] = dotplot_bloque

                # Liberar la memoria de la GPU después de cada subbloque
                del sec1_gpu
                del sec2_gpu
                del dotplot_gpu

            # Actualizar la barra de progreso
            pbar.update(1)

    # Asegurarse de que los datos estén escritos en el archivo
    dotplot.flush()

    return dotplot
