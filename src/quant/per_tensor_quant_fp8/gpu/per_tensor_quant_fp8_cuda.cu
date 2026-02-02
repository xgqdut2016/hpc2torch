#ifndef __PER_TENSOR_QUANT_FP8_KERNEL_CUH__
#define __PER_TENSOR_QUANT_FP8_KERNEL_CUH__

#ifdef ENABLE_NVIDIA_API
#include <flashinfer/vec_dtypes.cuh>
#endif
#include "../../utils.h"
#include <c10/util/Float8_e4m3fn.h>
#include <cmath>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cub/block/block_reduce.cuh>

template <typename T, unsigned int BLOCK_SIZE>
__device__ void
per_tensor_absmax_kernel(const T *__restrict__ input, float *__restrict__ output_s, const int64_t num_elements)
{
#ifdef ENABLE_NVIDIA_API
    float max_value = 0.0f;
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;

    constexpr uint32_t vec_size = 16 / sizeof(T);
    using vec_t = flashinfer::vec_t<T, vec_size>;

    const int32_t num_vec_elems = num_elements / vec_size;

    for (int32_t i = gid; i < num_vec_elems; i += grid_size)
    {
        vec_t input_vec;
        input_vec.cast_load(input + i * vec_size);

#pragma unroll
        for (uint32_t j = 0; j < vec_size; ++j)
        {
            float val = static_cast<float>(input_vec[j]);
            max_value = fmaxf(max_value, fabsf(val));
        }
    }

    const int32_t remaining_start = num_vec_elems * vec_size;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size)
    {
        float val = static_cast<float>(input[idx]);
        max_value = fmaxf(max_value, fabsf(val));
    }

    max_value = blockReduceMax(max_value);

    if (tid == 0)
    {
        atomicMaxFloat(output_s, max_value / FP8_E4M3_MAX);
    }
#elif defined ENABLE_QL_API
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ---- 2. reduce min ----
    float thread_max = -__FLT_MAX__;
    for (int ind = threadIdx.x; ind < num_elements; ind += BLOCK_SIZE)
    {
        thread_max = fmaxf(thread_max, fabs((float)input[ind]));
    }
    float local_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
    if (tid == 0)
    {
        output_s[0] = local_max / FP8_E4M3_MAX;
    }
#endif
}

template <typename T, typename DST_DTYPE>
__device__ void per_tensor_quant_fp8_kernel(
    const T *__restrict__ input,
    DST_DTYPE *__restrict__ output,
    const float *__restrict__ scale,
    const int64_t num_elements)
{
#ifdef ENABLE_NVIDIA_API
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;
    const float scale_val = 1.0f / (*scale);

    // We want to store 128 bits of data at a time. 16 = 128 / 8 bits
    // Load is already vectorized, so 16 elements work for T.
    const uint32_t VEC_SIZE = 16;
    using vec_t = flashinfer::vec_t<T, VEC_SIZE>;

    const int32_t num_vec_elems = num_elements / VEC_SIZE;

    for (int32_t i = gid; i < num_vec_elems; i += grid_size)
    {
        vec_t input_vec;
        input_vec.cast_load(input + i * VEC_SIZE);

        DST_DTYPE output_arr[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j)
        {
            float val = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
            output_arr[j] = static_cast<DST_DTYPE>(val);
#else
            output_arr[j] = c10::Float8_e4m3fnuz(
                __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
                c10::Float8_e4m3fnuz::from_bits());
#endif
        }
        *(uint4 *)(output + i * VEC_SIZE) = *(uint4 *)output_arr;
    }

    const int32_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size)
    {
        float val = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[idx]) * scale_val, FP8_E4M3_MAX));
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
        output[idx] = static_cast<DST_DTYPE>(val);
#else
        output[idx] = c10::Float8_e4m3fnuz(
            __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
            c10::Float8_e4m3fnuz::from_bits());
#endif
    }
#elif defined ENABLE_QL_API
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const float scale_val = 1.0f / scale[0];

    if (tid < num_elements)
    {
        float val = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[tid]) * scale_val, FP8_E4M3_MAX));
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
        output[tid] = static_cast<DST_DTYPE>(val);
#else
        output[tid] = c10::Float8_e4m3fnuz(
            __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
            c10::Float8_e4m3fnuz::from_bits());
#endif
    }

#endif
}

template <typename Tdata, unsigned int BLOCK_SIZE>
void __global__ blockPerTensorAbsmaxSym(
    float *x_scale, const Tdata *x, const int64_t num_elements)
{
    per_tensor_absmax_kernel<Tdata, BLOCK_SIZE>(
        x, x_scale, num_elements);
}

template <typename Tdata, typename DST_DTYPE, unsigned int BLOCK_SIZE>
void __global__ blockPerTensorQuantF8Sym(
    DST_DTYPE *x_packed, float *x_scale, const Tdata *x, const int64_t num_elements)
{
    per_tensor_quant_fp8_kernel<Tdata, DST_DTYPE>(
        x,
        x_packed,
        x_scale,
        num_elements);
}

template <unsigned int BLOCK_SIZE, typename Tdata>
void PerTensorQuantF8Kernel(void *x_packed, void *x_scale, void *x_zero, const void *x, uint64_t num_elements, bool is_static)
{

#ifdef ENABLE_NVIDIA_API
    constexpr unsigned int block_size = 256;
#else
    constexpr unsigned int block_size = BLOCK_SIZE;
#endif
    int num_blocks = (static_cast<int>(num_elements) + block_size - 1) / block_size;

    dim3 grid(num_blocks);
    dim3 block(block_size);
    if (x_zero == nullptr)
    {
        if (is_static == false)
        {
            blockPerTensorAbsmaxSym<Tdata, block_size>
                <<<grid, block>>>((float *)x_scale, (Tdata *)x, num_elements);
        }
        blockPerTensorQuantF8Sym<Tdata, __nv_fp8_e4m3, block_size>
            <<<grid, block>>>((__nv_fp8_e4m3 *)x_packed, (float *)x_scale, (Tdata *)x, num_elements);
    }
}

extern "C" void PerTensorQuantF8_nv(void *x_packed, void *x_scale, void *x_zero, const void *x, int num_elements, bool is_static, int byteSize)
{
    if (byteSize == 2)
    {
        PerTensorQuantF8Kernel<1024, half>(x_packed, x_scale, x_zero, x, static_cast<uint64_t>(num_elements), is_static);
    }
    if (byteSize == 4)
    {
        PerTensorQuantF8Kernel<1024, float>(x_packed, x_scale, x_zero, x, static_cast<uint64_t>(num_elements), is_static);
    }
}

#endif // __PER_TENSOR_QUANT_FP8_KERNEL_CUH__
