#include <cuda.h>
#include <cub/cub.cuh>

template <typename T, typename Tind>
__global__ void blockGatherKernel(T const *input, Tind const *indices, T *output, int dimsize, int behindsize, int indsize, int numThread)
{
    // input = [A, dimsize, D, E], indices = [B, C], axis = 1, output = [A, B, C, D, E]
    // frontsize表示axis前面的所有，即frontsize = A, behindisize表示axis后面所有，即behindsize = DE, indsize = BC
    // 专门对付indsize * behindsize 比较大的情况，此时num_blocks_x = frontsize
    //  blockIdx.x解决frontsize， (threadIdx.x + blockIdx.y * blockDim.x) * numThread解决indsize * behindsize
    // blockDim.x = 1, numThread表示一个线程处理多少个元素
    int otherIdx = threadIdx.x + blockIdx.y * blockDim.x;
    int frontIdx = blockIdx.x;

    for (int i = 0; i < numThread; i++)
    {
        int idx = otherIdx * numThread + i; // idx = indicesIdx * behindsize + behindIdx
        if (idx >= indsize * behindsize)
        {
            break;
        }
        int indicesIdx = idx / behindsize;
        int behindIdx = idx % behindsize;
        int inputIdx = frontIdx * dimsize * behindsize + indices[indicesIdx] * behindsize + behindIdx;
        int outputIdx = frontIdx * indsize * behindsize + idx;
        output[outputIdx] = input[inputIdx];
    }
}
template <typename T, typename Tind>
__global__ void warpGatherKernel(T const *input, Tind const *indices, T *output, int frontsize, int dimsize, int behindsize, int indsize, int numThread)
{
    // input = [A, dimsize, D, E], indices = [B, C], axis = 1, output = [A, B, C, D, E]
    // frontsize表示axis前面的所有，即frontsize = A, behindisize表示axis后面所有，即behindsize = DE, indsize = BC
    // 专门对付dimsize * behindsize 比较大的情况，此时num_blocks_x = frontsize
    //  blockIdx.x * blockDim.x + threadIdx.x解决frontsize， (threadIdx.y + blockIdx.y * blockDim.y) * numThread解决indsize * behindsize
    // numThread表示一个线程处理多少个元素
    int otherIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int frontIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (frontIdx >= frontsize)
    {
        return;
    }
    for (int i = 0; i < numThread; i++)
    {
        int idx = otherIdx * numThread + i; // idx = indicesIdx * behindsize + behindIdx
        if (idx >= indsize * behindsize)
        {
            break;
        }
        int indicesIdx = idx / behindsize;
        int behindIdx = idx % behindsize;
        int inputIdx = frontIdx * dimsize * behindsize + indices[indicesIdx] * behindsize + behindIdx;
        int outputIdx = frontIdx * indsize * behindsize + idx;
        output[outputIdx] = input[inputIdx];
    }
}
template <typename T, typename Tind>
void gatherLaunch(void const *input, void const *indices, void *output, int frontsize, int dimsize, int behindsize, int indsize)
{
    int othersize = indsize * behindsize;
    int numThread = 2;                 // 一个线程在othersize中处理多少个元素，这个参数对性能影响很大，需要仔细调整
    int count = othersize / numThread; // 处理othersize需要的总线程数目
    if (count > 1024)
    {
        int BLOCK_DIM = 1024;
        int num_block_x = frontsize;
        int num_block_y = (othersize + BLOCK_DIM * numThread - 1) / (BLOCK_DIM * numThread);
        dim3 block_dim(BLOCK_DIM, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        blockGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, dimsize, behindsize, indsize, numThread);
    }
    else if (count > 31)
    {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) / (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, frontsize, dimsize, behindsize, indsize, numThread);
    }
    else if (count > 15)
    {
        int BLOCK_DIM_x = 64;
        int BLOCK_DIM_y = 16;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) / (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, frontsize, dimsize, behindsize, indsize, numThread);
    }
    else if (count > 7)
    {
        int BLOCK_DIM_x = 128;
        int BLOCK_DIM_y = 8;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) / (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, frontsize, dimsize, behindsize, indsize, numThread);
    }
    else
    {
        int BLOCK_DIM_x = 256;
        int BLOCK_DIM_y = 4;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) / (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, frontsize, dimsize, behindsize, indsize, numThread);
    }
}
extern "C" void gather_nv_f32(void const *input, void const *indices, void *output, int frontsize, int dimsize, int behindsize, int indsize)
{
    gatherLaunch<float, uint64_t>(input, indices, output, frontsize, dimsize, behindsize, indsize);
}
extern "C" void gather_nv_f16(void const *input, void const *indices, void *output, int frontsize, int dimsize, int behindsize, int indsize)
{
    gatherLaunch<half, uint64_t>(input, indices, output, frontsize, dimsize, behindsize, indsize);
}
