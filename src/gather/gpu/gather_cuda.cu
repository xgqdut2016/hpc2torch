#include <cuda.h>
#include <cub/cub.cuh>

template <typename T, typename Tind>
__global__ void blockGatherKernel(T const *input, Tind const *indices, T *output, int stride, int indSize)
{
    // blockIdx.x < othersize, indSize >= 1024
    // input = [A, dimsize, D], axis = 1, indices = [B, C], output = [A, B, C, D] = i(BCD) + j(CD) + k(D) + s
    int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * indSize; // tid = s + i(BCD)
    for (int index = threadIdx.x; index < indSize; index += blockDim.x)
    {
        output[tid + index * stride] = input[tid + indices[index] * stride];
    }
    // int remain = indSize % blockDim.x;
    // int step = (indSize - remain) / blockDim.x;
    // if (threadIdx.x < remain)
    // {
    //     for (int index = threadIdx.x * (step + 1); index < (threadIdx.x + 1) * (step + 1); index++)
    //     {
    //         output[tid + index * stride] = input[tid + indices[index] * stride];
    //     }
    // }
    // else
    // {
    //     for (int index = remain + threadIdx.x * step; index < remain + (threadIdx.x + 1) * step; index++)
    //     {
    //         output[tid + index * stride] = input[tid + indices[index] * stride];
    //     }
    // }
}
template <typename T, typename Tind>
__global__ void warpGatherKernel(T const *input, Tind const *indices, T *output, int stride, int indSize)
{
    // indSize < 1024
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * indSize; // tid = s + i(BCD)
    for (int index = threadIdx.x; index < indSize; index += blockDim.x)
    {
        output[tid + index * stride] = input[tid + indices[index] * stride];
    }
}
template <typename T, typename Tind>
void gatherLaunch(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    if (indSize > 1024)
    {

        int BLOCK_DIM = 1024;
        blockGatherKernel<T, Tind>
            <<<othersize, BLOCK_DIM>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
    else if (indSize > 31)
    {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (othersize + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
    else if (indSize > 15)
    {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (othersize + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
    else if (indSize > 7)
    {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (othersize + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
    else
    {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (othersize + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
    }
}
extern "C" void gather_nv_f32(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    gatherLaunch<float, uint64_t>(input, indices, output, stride, indSize, othersize);
}
extern "C" void gather_nv_f16(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    gatherLaunch<half, uint64_t>(input, indices, output, stride, indSize, othersize);
}
