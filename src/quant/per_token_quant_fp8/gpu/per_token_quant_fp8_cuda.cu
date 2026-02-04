#ifndef __PER_TOKEN_QUANT_FP8_KERNEL_CUH__
#define __PER_TOKEN_QUANT_FP8_KERNEL_CUH__

#ifdef ENABLE_NVIDIA_API
#include <flashinfer/vec_dtypes.cuh>
#endif
#include "../../utils.h"
#include <c10/util/Float8_e4m3fn.h>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cub/block/block_reduce.cuh>

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

static constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
// 1. Warp‑local, no shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, unsigned int BLOCK_SIZE, int kTokensPerCTA = 8, int kVecSize = 16>
__device__ void per_token_quant_fp8_kernel(
    const T *__restrict__ input,
    DST_DTYPE *__restrict__ output_q,
    float *__restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens)
{
#ifdef ENABLE_NVIDIA_API
    const int warp_id = threadIdx.x / kWarpSize;       // 0‑7  (8 warps)
    const int lane_id = threadIdx.x & (kWarpSize - 1); // 0‑31
    const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
    if (token_id >= num_tokens)
    {
        return;
    }

    // Global tensors for this token
    const T *token_input = input + token_id * hidden_dim;
    DST_DTYPE *token_output = output_q + token_id * hidden_dim;
    float *token_scale = output_s + token_id;

    //
    // Pass-1: Perform a warp reduce to find the max_value of a token's hidden_dim
    //
    float max_value = 0.f;
    using vec_t = flashinfer::vec_t<T, kVecSize>;
    const int32_t num_vec_elems = hidden_dim / kVecSize;

    for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize)
    {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j)
        {
            max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
        }
    }

    float warp_max = warpReduceMax(max_value);

    // NOTE: one CTA has multiple warps (each warp handles one token), so `scale`
    // must be per-warp/per-thread (register) instead of a single shared variable.
    const float scale = warp_max / FP8_E4M3_MAX;
    // Broadcast scale
    if (lane_id == 0)
    {
        token_scale[0] = scale;
    }
    const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    //
    // Pass-2: quantize and write back
    //
    for (int i = lane_id; i < num_vec_elems; i += kWarpSize)
    {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);
        DST_DTYPE output_arr[kVecSize];
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j)
        {
            float val = static_cast<float>(input_vec[j]) * scale_inv;
            val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
            output_arr[j] = static_cast<DST_DTYPE>(val);
#else
            output_arr[j] = c10::Float8_e4m3fnuz(
                __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
                c10::Float8_e4m3fnuz::from_bits());
#endif
        }
        if constexpr (kVecSize == 16)
        {
            *(uint4 *)(token_output + i * kVecSize) = *(uint4 *)output_arr;
        }
        else
        {
            // Use element-wise copy for vector size 8 to ensure correctness
            for (int k = 0; k < kVecSize; ++k)
            {
                token_output[i * kVecSize + k] = output_arr[k];
            }
        }
    }
#elif defined ENABLE_QL_API
    const int warp_id = threadIdx.x / kWarpSize;       // 0‑7  (8 warps)
    const int lane_id = threadIdx.x & (kWarpSize - 1); // 0‑31
    const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
    if (token_id >= num_tokens)
    {
        return;
    }
    float *token_scale = output_s + token_id;
    int tid = token_id * hidden_dim;
    __shared__ float max_total[kTokensPerCTA];
    float max_data = -__FLT_MAX__;

    // ---- reduce max ----
    for (int ind = lane_id; ind < hidden_dim; ind += kWarpSize)
    {
        float v = fabsf((float)input[tid + ind]);
        max_data = fmaxf(max_data, v);
    }
    float warp_max = warpReduceMax(max_data);

    // NOTE: one CTA has multiple warps (each warp handles one token), so `scale`
    // must be per-warp/per-thread (register) instead of a single shared variable.
    const float scale = warp_max / FP8_E4M3_MAX;
    // Broadcast scale
    if (lane_id == 0)
    {
        token_scale[0] = scale;
    }

    float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;
    for (int ind = lane_id; ind < hidden_dim; ind += kWarpSize)
    {
        float val = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[tid + ind]) * scale_inv, FP8_E4M3_MAX));
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
        output_q[tid + ind] = static_cast<DST_DTYPE>(val);
#else
        output_q[tid + ind] = c10::Float8_e4m3fnuz(
            __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
            c10::Float8_e4m3fnuz::from_bits());
#endif
    }

#endif
}

// ---------------------------------------------------------------------------
// 2.  Baseline kernel (1 token / CTA, CUB block reduce)
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, unsigned int BLOCK_SIZE, int kVecSize = 16>
__device__ void per_token_quant_fp8_small_batch_kernel(
    const T *__restrict__ input,
    DST_DTYPE *__restrict__ output_q,
    float *__restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens)
{
#ifdef ENABLE_NVIDIA_API
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;

    const T *token_input = input + token_idx * hidden_dim;
    DST_DTYPE *token_output = output_q + token_idx * hidden_dim;

    float max_value = 0.0f;

    // Use template parameter for vector size
    using vec_t = flashinfer::vec_t<T, kVecSize>;
    const int32_t num_vec_elems = hidden_dim / kVecSize;

    // Find max using vectorized loads
    for (int32_t i = tid; i < num_vec_elems; i += block_dim)
    {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j)
        {
            float val = static_cast<float>(input_vec[j]);
            max_value = fmaxf(max_value, fabsf(val));
        }
    }

    max_value = blockReduceMax(max_value);

    __shared__ float scale;
    if (tid == 0)
    {
        scale = max_value / FP8_E4M3_MAX;
        output_s[token_idx] = scale;
    }
    __syncthreads();

    float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    // Quantize using vectorized loads
    for (int32_t i = tid; i < num_vec_elems; i += block_dim)
    {
        vec_t input_vec;
        input_vec.cast_load(token_input + i * kVecSize);

        DST_DTYPE output_arr[kVecSize];
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j)
        {
            float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
            output_arr[j] = static_cast<DST_DTYPE>(val);
#else
            output_arr[j] = c10::Float8_e4m3fnuz(
                __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
                c10::Float8_e4m3fnuz::from_bits());
#endif
        }

        if constexpr (kVecSize == 16)
        {
            *(uint4 *)(token_output + i * kVecSize) = *(uint4 *)output_arr;
        }
        else
        {
            // Use element-wise copy for vector size 8 to ensure correctness
            for (int k = 0; k < kVecSize; ++k)
            {
                token_output[i * kVecSize + k] = output_arr[k];
            }
        }
    }
#elif defined ENABLE_QL_API
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
    {
        return;
    }
    int tid = token_idx * hidden_dim;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ---- 2. reduce min ----
    float thread_max = -__FLT_MAX__;
    for (int ind = threadIdx.x; ind < hidden_dim; ind += BLOCK_SIZE)
    {
        thread_max = fmaxf(thread_max, fabsf((float)input[tid + ind]));
    }
    float local_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    __shared__ float scale;
    if (threadIdx.x == 0)
    {
        scale = local_max / FP8_E4M3_MAX;
        output_s[token_idx] = scale;
    }
    __syncthreads();

    float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    for (int ind = threadIdx.x; ind < hidden_dim; ind += BLOCK_SIZE)
    {
        float val = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[tid + ind]) * scale_inv, FP8_E4M3_MAX));
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
        output_q[tid + ind] = static_cast<DST_DTYPE>(val);
#else
        output_q[tid + ind] = c10::Float8_e4m3fnuz(
            __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
            c10::Float8_e4m3fnuz::from_bits());

#endif
    }

#endif
}
template <typename Tdata, typename DST_DTYPE, unsigned int BLOCK_SIZE, int kTokensPerCTA = 8, int kVecSize = 16>
__global__ void warpPerTokenQuantF8Sym(
    DST_DTYPE *x_packed, float *x_scale, const Tdata *x, const int64_t hidden_dim, const int64_t num_tokens)
{
    per_token_quant_fp8_kernel<Tdata, DST_DTYPE, BLOCK_SIZE, kTokensPerCTA, kVecSize>(
        x, x_packed, x_scale, hidden_dim, num_tokens);
}

template <typename Tdata, typename DST_DTYPE, unsigned int BLOCK_SIZE, int kVecSize = 16>
__global__ void blockPerTokenQuantF8Sym(
    DST_DTYPE *x_packed, float *x_scale, const Tdata *x, const int64_t hidden_dim, const int64_t num_tokens)
{
    per_token_quant_fp8_small_batch_kernel<Tdata, DST_DTYPE, BLOCK_SIZE, kVecSize>(
        x, x_packed, x_scale, hidden_dim, num_tokens);
}

#ifdef ENABLE_NVIDIA_API
inline int getSMCount()
{
    int device = -1;
    cudaGetDevice(&device);

    int sm_count = 0;
    cudaDeviceGetAttribute(
        &sm_count,
        cudaDevAttrMultiProcessorCount,
        device);

    return sm_count;
}

#endif

template <unsigned int BLOCK_SIZE, typename Tdata>
void PerTokenQuantF8Kernel(__nv_fp8_e4m3 *x_packed, float *x_scale, float *x_zero, const Tdata *x,
                           const int64_t hidden_dim,
                           const int64_t num_tokens)
{
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }
    const int TOKENS_PER_CTA = 8;
#ifdef ENABLE_NVIDIA_API
    int sm_count = getSMCount();
    const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
#else
    const bool use_warp_kernel = (hidden_dim < 1024);
#endif

    const bool use_vec16 = (hidden_dim % 16 == 0);
    const bool use_vec8 = (hidden_dim % 8 == 0);

    if (x_zero == nullptr)
    {
        if (use_warp_kernel)
        {

            // -------- warp‑local ---------------------------------------------------
            constexpr int THREADS = TOKENS_PER_CTA * kWarpSize; // 256
            dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
            dim3 block(THREADS);

            if (use_vec16)
            {
                warpPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, TOKENS_PER_CTA, 16><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
            else if (use_vec8)
            {
                warpPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, TOKENS_PER_CTA, 8><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
            else
            {
                warpPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, TOKENS_PER_CTA, 4><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
        }
        else
        {
            // -------- baseline -----------------------------------------------------
#ifdef ENABLE_NVIDIA_API
            constexpr unsigned int THREADS = 256;
#else
            constexpr unsigned int THREADS = BLOCK_SIZE;
#endif
            dim3 grid(num_tokens);
            dim3 block(THREADS);

            if (use_vec16)
            {
                blockPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, 16><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
            else if (use_vec8)
            {
                blockPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, 8><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
            else
            {
                blockPerTokenQuantF8Sym<Tdata, __nv_fp8_e4m3, THREADS, 4><<<grid, block, 0, stream>>>(
                    x_packed,
                    x_scale,
                    x,
                    hidden_dim,
                    num_tokens);
            }
        }
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}

extern "C" void PerTokenQuantF8_nv(void *x_packed, void *x_scale, void *x_zero, const void *x, int hidden_dim,
                                   int num_tokens, int byteSize)
{
    if (byteSize == 2)
    {
        PerTokenQuantF8Kernel<1024, half>((__nv_fp8_e4m3 *)x_packed, (float *)x_scale, (float *)x_zero, (half *)x, static_cast<uint64_t>(hidden_dim),
                                          static_cast<uint64_t>(num_tokens));
    }
    if (byteSize == 4)
    {
        PerTokenQuantF8Kernel<1024, float>((__nv_fp8_e4m3 *)x_packed, (float *)x_scale, (float *)x_zero, (float *)x, static_cast<uint64_t>(hidden_dim),
                                           static_cast<uint64_t>(num_tokens));
    }
}

#endif // __PER_TOKEN_QUANT_FP8_KERNEL_CUH__
