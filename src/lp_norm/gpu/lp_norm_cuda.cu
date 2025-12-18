#include <cub/block/block_reduce.cuh>

template <typename T, unsigned int BLOCK_SIZE>
__device__ void blockLPNormKernel(
    T const *input, T *output, float p, int dimsize,
    int stride, float eps)
{

    int tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) * dimsize; // now, tid = i(JKS) + k(S) + s;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float local_max = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE)
    {
        local_max = max(local_max, fabsf((float)input[tid + ind * stride]));
    }
    __shared__ float global_max;
#if CUDART_VERSION >= 12090
    float max_block = BlockReduce(temp_storage).Reduce(local_max, ::cuda::maximum());
#else
    float max_block = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
#endif
    if (threadIdx.x == 0)
    { // must set threadIdx.x = 0 write the output to memory
        global_max = max_block;
    }
    __syncthreads();
    float global_max_inv = __fdividef(1.0F, max(global_max, eps));

    float p_partial = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE)
    {
        p_partial += powf((float)input[tid + ind * stride] * global_max_inv, p);
    }

    __shared__ float p_total;
    float p_block = BlockReduce(temp_storage).Sum(p_partial);
    if (threadIdx.x == 0)
    { // must set threadIdx.x = 0 write the output to memory
        p_total = powf(p_block, 1.0f / p);
    }
    __syncthreads();
    float inv = __fdividef(1.0F, p_total + eps) * global_max_inv;

    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE)
    {
        output[tid + ind * stride] = static_cast<T>(
            static_cast<float>(
                input[tid + ind * stride]) *
            inv);
    }
}

template <typename T, unsigned int BLOCK_SIZE>
__device__ void blockLPNormStridesKernel(
    T const *input, T *output, const int *output_strides,
    const int *input_strides,
    const int *shape, int ndim, float p, int dimsize,
    float eps)
{

    // 只能处理axis=-1
    int ind_i = 0; // input id
    int ind_o = 0; // output id
    int tid = blockIdx.x;
    for (int j = ndim - 2; j >= 0; j--)
    {
        ind_i += (tid % (int)shape[j]) * (int)input_strides[j];
        ind_o += (tid % (int)shape[j]) * (int)output_strides[j];
        tid = tid / (int)shape[j];
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float local_max = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE)
    {
        local_max = max(local_max, fabsf((float)input[ind_i + ind]));
    }
    __shared__ float global_max;
#if CUDART_VERSION >= 12090
    float max_block = BlockReduce(temp_storage).Reduce(local_max, ::cuda::maximum());
#else
    float max_block = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
#endif
    if (threadIdx.x == 0)
    { // must set threadIdx.x = 0 write the output to memory
        global_max = max_block;
    }
    __syncthreads();
    float global_max_inv = __fdividef(1.0F, max(global_max, eps));

    float p_partial = 0.0f;
    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE)
    {
        p_partial += powf((float)input[ind_i + ind] * global_max_inv, p);
    }

    __shared__ float p_total;
    float p_block = BlockReduce(temp_storage).Sum(p_partial);
    if (threadIdx.x == 0)
    { // must set threadIdx.x = 0 write the output to memory
        p_total = powf(p_block, 1.0f / p);
    }
    __syncthreads();
    float inv = __fdividef(1.0F, p_total + eps) * global_max_inv;

    for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE)
    {
        output[ind_o + ind] = static_cast<T>(
            static_cast<float>(
                input[ind_i + ind]) *
            inv);
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

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return max(a, b);
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

template <typename T, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpLPNormKernel(T const *input, T *output,
                                 float p, int othersize, int dimsize,
                                 int stride, float eps)
{
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;

    if (otherIdx < othersize)
    {

        __shared__ float p_total[BLOCK_SIZE_y];
        __shared__ float p_max[BLOCK_SIZE_y];
        float local_max = 0.0f;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x)
        {
            local_max = max(local_max, fabsf((float)input[tid + ind * stride]));
        }
        local_max = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(local_max);
        if (threadIdx.x == 0)
        {
            p_max[threadIdx.y] = local_max;
        }
        __syncthreads();
        float global_max = max(p_max[threadIdx.y], eps);
        float global_max_inv = __fdividef(1.0F, max(p_max[threadIdx.y], eps));
        float p_data = 0.0f;

        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x)
        {
            float v = fabsf((float)input[tid + ind * stride]) * global_max_inv;
            p_data += powf(v, p);
        }

        p_data = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(p_data);

        if (threadIdx.x == 0)
        {
            p_total[threadIdx.y] = powf(p_data, 1.0f / p);
        }
        __syncthreads();

        //--------------------------------------------
        float inv = __fdividef(1.0F, p_total[threadIdx.y] + eps) * global_max_inv;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x)
        {
            output[tid + ind * stride] = static_cast<T>(
                (float)input[tid + ind * stride] * inv);
        }
    }
}

template <typename T, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpLPNormStridesKernel(T const *input, T *output, const int *output_strides,
                                        const int *input_strides,
                                        const int *shape, int ndim,
                                        float p, int othersize, int dimsize,
                                        float eps)
{
    int ind_i = 0; // input id
    int ind_o = 0; // output id
    int tid = blockIdx.x * blockDim.y + threadIdx.y;

    if (tid < othersize)
    {
        for (int j = ndim - 2; j >= 0; j--)
        {
            ind_i += (tid % (int)shape[j]) * (int)input_strides[j];
            ind_o += (tid % (int)shape[j]) * (int)output_strides[j];
            tid = tid / (int)shape[j];
        }
        __shared__ float p_total[BLOCK_SIZE_y];
        __shared__ float p_max[BLOCK_SIZE_y];
        float local_max = 0.0f;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x)
        {
            local_max = max(local_max, fabsf((float)input[ind_i + ind]));
        }
        local_max = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(local_max);
        if (threadIdx.x == 0)
        {
            p_max[threadIdx.y] = local_max;
        }
        __syncthreads();
        float global_max = max(p_max[threadIdx.y], eps);
        float global_max_inv = __fdividef(1.0F, max(p_max[threadIdx.y], eps));
        float p_data = 0.0f;

        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x)
        {
            float v = fabsf((float)input[ind_i + ind]) * global_max_inv;
            p_data += powf(v, p);
        }

        p_data = WarpAllReduce<SumOp, float, BLOCK_SIZE_x>(p_data);

        if (threadIdx.x == 0)
        {
            p_total[threadIdx.y] = powf(p_data, 1.0f / p);
        }
        __syncthreads();

        //--------------------------------------------
        float inv = __fdividef(1.0F, p_total[threadIdx.y] + eps) * global_max_inv;
        for (int ind = threadIdx.x; ind < dimsize; ind += BLOCK_SIZE_x)
        {
            output[ind_o + ind] = static_cast<T>(
                (float)input[ind_i + ind] * inv);
        }
    }
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__global__ void blockLPNorm(
    Tdata *y, const Tdata *x,
    float p,
    int dimsize,
    int stride, float eps)
{
    blockLPNormKernel<Tdata, BLOCK_SIZE>(x, y, p, dimsize, stride, eps);
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__global__ void blockLPNormStrides(
    Tdata *y, const Tdata *x,
    const int *output_strides,
    const int *input_strides,
    const int *shape, int ndim, float p, int dimsize,
    float eps)
{
    blockLPNormStridesKernel<Tdata, BLOCK_SIZE>(x, y, output_strides, input_strides, shape, ndim, p, dimsize, eps);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__global__ void warpLPNorm(
    Tdata *y, const Tdata *x,
    float p,
    int othersize,
    int dimsize,
    int stride, float eps)
{
    warpLPNormKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x, y, p, othersize, dimsize, stride, eps);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__global__ void warpLPNormStrides(
    Tdata *y, const Tdata *x,
    const int *output_strides,
    const int *input_strides,
    const int *shape, int ndim,
    float p, int othersize, int dimsize,
    float eps)
{
    warpLPNormStridesKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x, y, output_strides, input_strides, shape, ndim, p, othersize, dimsize, eps);
}
template <typename Tdata>
void launchKernel(void *y, const void *x, const int *output_strides,
                  const int *input_strides,
                  const int *shape, int ndim, int axis,
                  int p,
                  float eps, bool continuous)
{
    if (axis < 0)
    {
        axis += ndim;
    }

    int othersize = 1;
    for (int i = 0; i < (int)ndim; i++)
    {
        if (i != axis)
        {
            othersize *= shape[i];
        }
    }
    int dimsize = shape[axis];
    int stride = 1;
    for (int i = ndim - 1; i > axis; i--)
    {
        stride *= (int)shape[i];
    }
    float p_f = static_cast<float>(p);

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("流创建失败: %s\n", cudaGetErrorString(err));
        // 错误处理
    }
    constexpr unsigned int BLOCK_SIZE = 1024;
    int num_blocks = (int)othersize;
    if (continuous)
    {

        if (dimsize > 1024)
        {
            blockLPNorm<Tdata, BLOCK_SIZE>
                <<<num_blocks, BLOCK_SIZE, 0, stream>>>((Tdata *)y, (Tdata *)x,
                                                        p_f, dimsize, stride, eps);
        }
        else
        {
            constexpr unsigned int BLOCK_SIZE_x = 32;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpLPNorm<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                <<<grid_dim, block_dim, 0, stream>>>((Tdata *)y, (Tdata *)x,
                                                     p_f, othersize, dimsize, stride, eps);
        }
    }
    else
    {
        char *workspace_ptr;
        cudaMalloc((void **)&workspace_ptr, 3 * ndim * sizeof(int));
        int *input_strides_cuda = reinterpret_cast<int *>(workspace_ptr);
        int *output_strides_cuda = input_strides_cuda + ndim;
        cudaMemcpyAsync(input_strides_cuda, input_strides, sizeof(int) * ndim, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(output_strides_cuda, output_strides, sizeof(int) * ndim, cudaMemcpyHostToDevice, stream);
        int ptrdiff_array_size = 2 * ndim * sizeof(int);
        int *shape_cuda = reinterpret_cast<int *>(workspace_ptr + ptrdiff_array_size);
        cudaMemcpyAsync(shape_cuda, shape, sizeof(int) * ndim, cudaMemcpyHostToDevice, stream);
        if (axis == ndim - 1)
        {
            if (dimsize > 1024)
            {
                blockLPNormStrides<Tdata, BLOCK_SIZE>
                    <<<num_blocks, BLOCK_SIZE, 0, stream>>>((Tdata *)y, (Tdata *)x, output_strides_cuda, input_strides_cuda, shape_cuda, ndim,
                                                            p_f, dimsize, eps);
            }
            else
            {
                constexpr unsigned int BLOCK_SIZE_x = 32;
                constexpr unsigned int BLOCK_SIZE_y = 32;
                int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
                dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
                dim3 grid_dim(num_block_x, 1, 1);
                warpLPNormStrides<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                    <<<grid_dim, block_dim, 0, stream>>>((Tdata *)y, (Tdata *)x, output_strides_cuda, input_strides_cuda, shape_cuda, ndim,
                                                         p_f, othersize, dimsize, eps);
            }
        }
        cudaFree(workspace_ptr);
    }
    err = cudaStreamDestroy(stream);
    if (err != cudaSuccess)
    {
        printf("流销毁失败: %s\n", cudaGetErrorString(err));
        // 错误处理
    }
}
extern "C" void lp_norm_nv(void *y, const void *x, const int *output_strides,
                           const int *input_strides,
                           const int *shape, int ndim, int axis,
                           int p,
                           float eps, bool continuous, int byteSize)
{
    if (byteSize == 2)
    {
        launchKernel<half>(y, x, output_strides, input_strides, shape, ndim, axis, p, eps, continuous);
    }
    if (byteSize == 4)
    {
        launchKernel<float>(y, x, output_strides, input_strides, shape, ndim, axis, p, eps, continuous);
    }
}