#include "kernel.cuh"
#include <cuda.h>

template <typename Tdata, unsigned int BLOCK_SIZE>
__global__ void blockPerChannelQuantI8(
    int8_t *x_packed, float *x_scale, float *x_zero, const Tdata *x, int M, int K)
{
    blockPerChannelQuantI8Kernel<Tdata, BLOCK_SIZE>(x_packed, x_scale, x_zero, x, M, K);
}
template <typename Tdata, unsigned int BLOCK_SIZE>
__global__ void blockPerChannelQuantI8Sym(
    int8_t *x_packed, float *x_scale, const Tdata *x, int M, int K)
{
    blockPerChannelQuantI8SymKernel<Tdata, BLOCK_SIZE>(x_packed, x_scale, x, M, K);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__global__ void warpPerChannelQuantI8(
    int8_t *x_packed, float *x_scale, float *x_zero, const Tdata *x, int M, int K)
{
    warpPerChannelQuantI8Kernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x_packed, x_scale, x_zero, x, M, K);
}
template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__global__ void warpPerChannelQuantI8Sym(
    int8_t *x_packed, float *x_scale, const Tdata *x, int M, int K)
{
    warpPerChannelQuantI8SymKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x_packed, x_scale, x, M, K);
}

template <unsigned int BLOCK_SIZE, typename Tdata>
void PerChannelQuantI8Kernel(void *x_packed, void *x_scale, void *x_zero, const void *x, int M, int K)
{

    if (K >= 1024)
    {
        if (x_zero == nullptr)
        {
            blockPerChannelQuantI8Sym<Tdata, BLOCK_SIZE>
                <<<M, BLOCK_SIZE>>>((int8_t *)x_packed, (float *)x_scale, (Tdata *)x, M, K);
        }
        else
        {
            blockPerChannelQuantI8<Tdata, BLOCK_SIZE>
                <<<M, BLOCK_SIZE>>>((int8_t *)x_packed, (float *)x_scale, (float *)x_zero, (Tdata *)x, M, K);
        }
    }
    else
    {
        constexpr unsigned int BLOCK_SIZE_x = 32;
        constexpr unsigned int BLOCK_SIZE_y = 32;
        int num_block_x = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
        dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);
        if (x_zero == nullptr)
        {
            warpPerChannelQuantI8Sym<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                <<<grid_dim, block_dim>>>((int8_t *)x_packed, (float *)x_scale, (Tdata *)x, M, K);
        }
        else
        {
            warpPerChannelQuantI8<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                <<<grid_dim, block_dim>>>((int8_t *)x_packed, (float *)x_scale, (float *)x_zero, (Tdata *)x, M, K);
        }
    }
}

extern "C" void PerChannelQuantI8_nv(void *x_packed, void *x_scale, void *x_zero, const void *x, int M, int K, int byteSize)
{
    if (byteSize == 2)
    {
        PerChannelQuantI8Kernel<1024, half>(x_packed, x_scale, x_zero, x, M, K);
    }
    if (byteSize == 4)
    {
        PerChannelQuantI8Kernel<1024, float>(x_packed, x_scale, x_zero, x, M, K);
    }
}
