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