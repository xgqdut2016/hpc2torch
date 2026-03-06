#include "kernel.cuh"
#include <cuda.h>

template <typename Tdata, unsigned int BLOCK_SIZE>
__global__ void perTensorQuantI8Sym(
    int8_t *x_packed, float *x_scale, const Tdata *x, int num_elements)
{
    perTensorQuantI8SymKernel<Tdata, BLOCK_SIZE>(x_packed, x_scale, x, num_elements);
}

template <unsigned int BLOCK_SIZE, typename Tdata>
void PerTensorQuantI8Kernel(void *x_packed, void *x_scale, void *x_zero, const void *x, int num_elements)
{

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (x_zero == nullptr)
    {
        perTensorQuantI8Sym<Tdata, BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>((int8_t *)x_packed, (float *)x_scale, (Tdata *)x, num_elements);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}

extern "C" void PerTensorQuantI8_nv(void *x_packed, void *x_scale, void *x_zero, const void *x, int num_elements, int byteSize)
{
    if (byteSize == 2)
    {
        PerTensorQuantI8Kernel<1024, half>(x_packed, x_scale, x_zero, x, num_elements);
    }
    if (byteSize == 4)
    {
        PerTensorQuantI8Kernel<1024, float>(x_packed, x_scale, x_zero, x, num_elements);
    }
}
