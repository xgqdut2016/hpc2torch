#include <cuda.h>
#include <cub/cub.cuh>
template <typename T, int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM)
    __global__ void blockLayernormKernel(T const *input, T const *scale, T const *bias, T *output, float eps, int behindsize)
{
    // 假设input= [A, B, C, D], axis = 2, frontsize = AB = blockDim.x, behindsize = CD
    // 全局索引index = i(BCD) + j (CD) + k(D) + s
    // blockIdx.x = i(B) + j;默认behindsize >= BLOCK_DIM
    // scale,bias长度为behindsize,形状为[C,D]
    int tid = blockIdx.x * behindsize;
    float muPartial = 0.0f;
    for (int id = threadIdx.x; id < behindsize; id += BLOCK_DIM)
    {
        muPartial += static_cast<float>(input[tid + id]); // half很多操作不支持，运算过程使用float数据
    }
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu;
    float muBlock = BlockReduce(temp_storage).Reduce(muPartial, cub::Sum());
    if (threadIdx.x == 0)
    {
        mu = muBlock * __fdividef(1.0F, behindsize);
    } // threadIdx.x = 0对应的是全局sum
    __syncthreads();
    float sigma2Partial = 0.0f;
    for (int id = threadIdx.x; id < behindsize; id += BLOCK_DIM)
    {
        sigma2Partial += (static_cast<float>(input[tid + id]) - mu) * (static_cast<float>(input[tid + id]) - mu);
    }
    __shared__ float sigma2;
    float sigma2Block = BlockReduce(temp_storage).Reduce(sigma2Partial, cub::Sum());
    if (threadIdx.x == 0)
    {
        float sigmaTmp = sqrt(sigma2Block * __fdividef(1.0F, behindsize) + eps);
        sigma2 = __fdividef(1.0F, sigmaTmp);
    }
    __syncthreads();
    for (int id = threadIdx.x; id < behindsize; id += BLOCK_DIM)
    {
        output[tid + id] = static_cast<T>(static_cast<float>(scale[id]) * (static_cast<float>(input[tid + id]) - mu) * sigma2 + static_cast<float>(bias[id]));
    }
}
template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val)
{
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
template <typename T, int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void warpLayernormKernel(T const *input, T const *scale, T const *bias, T *output, float eps, int behindsize)
{
    // 默认behindsize < 1024
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx * behindsize;
    float muPartial = 0.0f;
    for (int id = threadIdx.x; id < behindsize; id += BLOCK_DIM_x)
    {
        muPartial += static_cast<float>(input[tid + id]);
    }
    muPartial = WarpAllReduce<SumOp, float, BLOCK_DIM_x>(muPartial);
    __shared__ float mu[BLOCK_DIM_y];

    if (threadIdx.x == 0)
    {
        mu[threadIdx.y] = muPartial * __fdividef(1.0F, behindsize);
    } // threadIdx.x = 0对应的是全局sum
    __syncthreads();
    float sigma2Partial = 0.0f;
    for (int id = threadIdx.x; id < behindsize; id += BLOCK_DIM_x)
    {
        sigma2Partial += (static_cast<float>(input[tid + id]) - mu[threadIdx.y]) * (static_cast<float>(input[tid + id]) - mu[threadIdx.y]);
    }
    sigma2Partial = WarpAllReduce<SumOp, float, BLOCK_DIM_x>(sigma2Partial);
    __shared__ float sigma2[BLOCK_DIM_y];

    if (threadIdx.x == 0)
    {
        float sigmaTmp = sqrt(sigma2Partial * __fdividef(1.0F, behindsize) + eps);
        sigma2[threadIdx.y] = __fdividef(1.0F, sigmaTmp);
    }
    __syncthreads();
    for (int id = threadIdx.x; id < behindsize; id += BLOCK_DIM_x)
    {
        output[tid + id] = static_cast<T>(static_cast<float>(scale[id]) * (static_cast<float>(input[tid + id]) - mu[threadIdx.y]) * sigma2[threadIdx.y] + static_cast<float>(bias[id]));
    }
}
template <typename T>
void layernormLaunch(void const *input, void const *scale, void const *bias, void *output, float eps, int size, int behindsize)
{

    int num_blocks = size / behindsize;
    if (behindsize >= 1024)
    {
        int BLOCK_DIM = 1024;
        blockLayernormKernel<T, 1024>
            <<<num_blocks, BLOCK_DIM>>>((T *)input, (T *)scale, (T *)bias, (T *)output, eps, behindsize);
    }
    else if (behindsize > 31)
    {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLayernormKernel<T, 32, 32>
            <<<grid_dim, block_dim>>>((T *)input, (T *)scale, (T *)bias, (T *)output, eps, behindsize);
    }
    else if (behindsize > 15)
    {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLayernormKernel<T, 16, 64>
            <<<grid_dim, block_dim>>>((T *)input, (T *)scale, (T *)bias, (T *)output, eps, behindsize);
    }
    else if (behindsize > 7)
    {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLayernormKernel<T, 8, 128>
            <<<grid_dim, block_dim>>>((T *)input, (T *)scale, (T *)bias, (T *)output, eps, behindsize);
    }
    else
    {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLayernormKernel<T, 4, 256>
            <<<grid_dim, block_dim>>>((T *)input, (T *)scale, (T *)bias, (T *)output, eps, behindsize);
    }
    cudaDeviceSynchronize();
}
extern "C" void layernorm_nv(void const *input, void const *scale, void const *bias, void *output, float eps, int size, int behindsize, int byteSize)
{
    if (byteSize == 2)
    {
        layernormLaunch<half>(input, scale, bias, output, eps, size, behindsize);
    }
    if (byteSize == 4)
    {
        layernormLaunch<float>(input, scale, bias, output, eps, size, behindsize);
    }
}
