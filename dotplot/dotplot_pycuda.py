import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void generar_dotplot(char *sec1, char *sec2, int *dotplot, int len1, int len2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < len1 && j < len2) {
        dotplot[i * len2 + j] = (sec1[i] == sec2[j]) ? 1 : 0;
    }
}
""")

def dotplot_pycuda(secuencia1, secuencia2):
    sec1_gpu = gpuarray.to_gpu(np.array(list(secuencia1), dtype=np.byte))
    sec2_gpu = gpuarray.to_gpu(np.array(list(secuencia2), dtype=np.byte))
    dotplot_gpu = gpuarray.zeros((len(secuencia1), len(secuencia2)), dtype=np.int32)

    block_size = (16, 16, 1)
    grid_size = (
        (len(secuencia1) + block_size[0] - 1) // block_size[0],
        (len(secuencia2) + block_size[1] - 1) // block_size[1],
    )

    func = mod.get_function("generar_dotplot")
    func(
        sec1_gpu, sec2_gpu, dotplot_gpu,
        np.int32(len(secuencia1)), np.int32(len(secuencia2)),
        block=block_size, grid=grid_size,
    )

    return dotplot_gpu.get()