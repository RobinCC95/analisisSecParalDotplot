import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

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
    # Mapeo de caracteres a valores numéricos
    mapa = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapa[base] for base in secuencia], dtype=np.byte)

# Función principal para generar el dotplot
def dotplot_pycuda(secuencia1, secuencia2):
    # Convertir las secuencias a valores numéricos
    sec1_numerica = convertir_secuencia_a_numeros(secuencia1)
    sec2_numerica = convertir_secuencia_a_numeros(secuencia2)

    # Transferir las secuencias a la GPU
    sec1_gpu = gpuarray.to_gpu(sec1_numerica)
    sec2_gpu = gpuarray.to_gpu(sec2_numerica)

    # Crear el arreglo para el dotplot en la GPU
    dotplot_gpu = gpuarray.zeros((len(secuencia1), len(secuencia2)), dtype=np.int32)

    # Configurar el tamaño de los bloques y la cuadrícula
    block_size = (16, 16, 1)
    grid_size = (
        (len(secuencia1) + block_size[0] - 1) // block_size[0],
        (len(secuencia2) + block_size[1] - 1) // block_size[1],
    )

    # Obtener la función CUDA
    func = mod.get_function("generar_dotplot")

    # Ejecutar el kernel CUDA
    func(
        sec1_gpu, sec2_gpu, dotplot_gpu,
        np.int32(len(secuencia1)), np.int32(len(secuencia2)),
        block=block_size, grid=grid_size,
    )

    # Devolver el resultado del dotplot
    return dotplot_gpu.get()