#include <cub/block/block_reduce.cuh>
#include <cuda.h>
__device__ inline int round_half_away_from_zero(float x)
{
    float ax = fabsf(x);
    float r = floorf(ax + 0.5f);
    return (x >= 0.0f) ? (int)r : -(int)r;
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void blockPerChannelQuantI8Kernel(
    int8_t *x_packed, float *x_scale, float *x_zero, const Tdata *x,
    int M, int K)
{
    int row = blockIdx.x;
    int tid = row * K;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float thread_max = -__FLT_MAX__;
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE)
    {
        thread_max = fmaxf(thread_max, (float)x[tid + ind]);
    }
    float local_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    __shared__ float global_max_f;
    if (threadIdx.x == 0)
    {
        global_max_f = local_max;
    }
    __syncthreads();

    // ---- 2. reduce min ----
    float thread_min = __FLT_MAX__;
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE)
    {
        thread_min = fminf(thread_min, (float)x[tid + ind]);
    }
    float local_min = BlockReduce(temp_storage).Reduce(thread_min, cub::Min());

    __shared__ float global_min_f;
    if (threadIdx.x == 0)
    {
        global_min_f = local_min;
    }
    __syncthreads();

    float global_max = global_max_f;
    float global_min = global_min_f;

    float scale = (global_max - global_min) / 255.0f;
    if (scale < 1e-8f)
    {
        scale = 1e-8f;
    }

    float inv_scale = 1.0f / scale;
    float zero = -global_min * inv_scale - 128.0f;

    x_scale[row] = scale;
    x_zero[row] = zero;

    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE)
    {

        float v = (float)x[tid + ind];
        float qf = v * inv_scale + zero;

        int q = round_half_away_from_zero(qf);

        if (q > 127)
        {
            q = 127;
        }
        if (q < -128)
        {
            q = -128;
        }

        x_packed[tid + ind] = (int8_t)q;
    }
}
/**
 * Performs per-channel symmetric quantization to int8 for large matrices (K >= 1024).
 * Uses zero-centered scaling only, no zero point, and packs quantized data.
 */
template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void blockPerChannelQuantI8SymKernel(
    int8_t *x_packed, float *x_scale, const Tdata *x,
    int M, int K)
{
    int row = blockIdx.x;
    int tid = row * K;

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ---- 2. reduce min ----
    float thread_max = -__FLT_MAX__;
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE)
    {
        thread_max = fmaxf(thread_max, fabs((float)x[tid + ind]));
    }
    float local_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    __shared__ float global_max_f;
    if (threadIdx.x == 0)
    {
        global_max_f = local_max;
    }
    __syncthreads();

    float global_max = global_max_f;

    float scale = global_max / 127.0f;
    if (scale < 1e-8f)
    {
        scale = 1e-8f;
    }

    float inv_scale = 1.0f / scale;

    x_scale[row] = scale;

    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE)
    {

        float v = (float)x[tid + ind];
        float qf = v * inv_scale;

        int q = round_half_away_from_zero(qf);

        if (q > 127)
        {
            q = 127;
        }
        if (q < -127)
        {
            q = -127;
        }

        x_packed[tid + ind] = (int8_t)q;
    }
}

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return max(a, b);
    }
};
template <typename T>
struct MinOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return min(a, b);
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
/**
 * Performs per-channel asymmetric quantization to int8 for large matrices (K < 1024).
 * Computes scale/zero point per channel (column) and packs quantized data.
 */
template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpPerChannelQuantI8Kernel(
    int8_t *x_packed, float *x_scale, float *x_zero, const Tdata *x,
    int M, int K)
{
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx * K;

    if (otherIdx < M)
    {

        __shared__ float max_total[BLOCK_SIZE_y];
        __shared__ float min_total[BLOCK_SIZE_y];

        float max_data = -__FLT_MAX__;
        float min_data = __FLT_MAX__;

        // ---- reduce max/min ----
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x)
        {
            float v = (float)x[tid + ind];
            max_data = fmaxf(max_data, v);
            min_data = fminf(min_data, v);
        }

        max_data = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(max_data);
        min_data = WarpAllReduce<MinOp, float, BLOCK_SIZE_x>(min_data);

        if (threadIdx.x == 0)
        {
            max_total[threadIdx.y] = max_data;
            min_total[threadIdx.y] = min_data;
        }
        __syncthreads();

        float max_f = max_total[threadIdx.y];
        float min_f = min_total[threadIdx.y];

        float scale = (max_f - min_f) / 255.0f;
        if (scale < 1e-8f)
        {
            scale = 1e-8f;
        }

        float inv_scale = 1.0f / scale;
        float zero = -min_f * inv_scale - 128.0f;

        x_scale[otherIdx] = scale;
        x_zero[otherIdx] = zero;

        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x)
        {
            float v = (float)x[tid + ind];
            float qf = v * inv_scale + zero;

            int q = round_half_away_from_zero(qf);

            if (q > 127)
            {
                q = 127;
            }
            if (q < -128)
            {
                q = -128;
            }

            x_packed[tid + ind] = (int8_t)q;
        }
    }
}
/**
 * Performs per-channel symmetric quantization to int8 for large matrices (K < 1024).
 * Uses zero-centered scaling only, no zero point, and packs quantized data.
 */
template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpPerChannelQuantI8SymKernel(
    int8_t *x_packed, float *x_scale, const Tdata *x,
    int M, int K)
{
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx * K;

    if (otherIdx < M)
    {

        __shared__ float max_total[BLOCK_SIZE_y];

        float max_data = -__FLT_MAX__;

        // ---- reduce max/min ----
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x)
        {
            float v = fabs((float)x[tid + ind]);
            max_data = fmaxf(max_data, v);
        }

        max_data = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(max_data);

        if (threadIdx.x == 0)
        {
            max_total[threadIdx.y] = max_data;
        }
        __syncthreads();

        float max_f = max_total[threadIdx.y];

        float scale = max_f / 127.0f;
        if (scale < 1e-8f)
        {
            scale = 1e-8f;
        }

        float inv_scale = 1.0f / scale;

        x_scale[otherIdx] = scale;

        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x)
        {
            float v = (float)x[tid + ind];
            float qf = v * inv_scale;

            int q = round_half_away_from_zero(qf);

            if (q > 127)
            {
                q = 127;
            }
            if (q < -127)
            {
                q = -127;
            }

            x_packed[tid + ind] = (int8_t)q;
        }
    }
}

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
