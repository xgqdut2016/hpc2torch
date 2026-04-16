#include "gpu/common_gpu.h"
#include "marlin.cuh"
#include "../core/utils.h"

// for only non-zp format (like gptq)
__global__ void marlin_int4_fp8_preprocess_kernel_without_zp(
    // qweight: (size_k * size_n // 8,)
    const int32_t *__restrict__ qweight,
    // output: same shape with qweight
    int32_t *__restrict__ output)
{
    int32_t val = qweight[blockIdx.x * 32 + threadIdx.x];
    int32_t new_val = 0;

#pragma unroll
    for (int32_t i = 0; i < 8; i++)
    {
        int32_t single_val = val & 0xF;
        single_val = single_val >= 8 ? single_val - 8 : 15 - single_val;
        new_val |= single_val << (i * 4);
        val >>= 4;
    }

    output[blockIdx.x * 32 + threadIdx.x] = new_val;
}

// for awq format only (with zp and with awq weight layout)
__global__ void marlin_int4_fp8_preprocess_kernel_awq(
    // AWQ qweight: (size_k, size_n // 8)
    const int32_t *__restrict__ qweight,
    // output: same shape with qweight
    int32_t *__restrict__ output,
    // AWQ zeros: (size_k // group_size, size_n // 8)
    const int32_t *__restrict__ qzeros, int32_t size_n, int32_t size_k,
    int32_t group_size)
{
    int32_t val =
        qweight[(blockIdx.x * 32 + threadIdx.x) * size_n / 8 + blockIdx.y];
    int32_t zero =
        qzeros[(blockIdx.x * 32 + threadIdx.x) / group_size * size_n / 8 +
               blockIdx.y];
    int32_t new_val = 0;

#pragma unroll
    for (int32_t i = 0; i < 8; i++)
    {
        int32_t single_val = val & 0xF;
        int32_t single_zero = zero & 0xF;

        single_val =
            single_val >= single_zero ? single_val - single_zero : 15 - single_val;
        new_val |= single_val << (i * 4);
        val >>= 4;
        zero >>= 4;
    }

    output[(blockIdx.x * 32 + threadIdx.x) * size_n / 8 + blockIdx.y] = new_val;
}

void marlin_int4_fp8_preprocess(
    void *output, const void *qweight, const void *qzeros,
    int K, int N, int num_groups)
{
    // qweight = [K, N]
    // qzeros = [K, N]
    int num_elements = K * N;
    if (qzeros == nullptr)
    {
        RUNTIME_CHECK((num_elements * 8 % 256 == 0), "num_elements * 8 % 256 != 0");

        int blocks = num_elements * 8 / 256;
        marlin_int4_fp8_preprocess_kernel_without_zp<<<blocks, 32>>>(
            (const int32_t *)qweight, (int32_t *)output);
    }
    else
    {
        int size_n = N * 8;
        RUNTIME_CHECK((K % 32 == 0), "K % 32 != 0");

        int32_t group_size = K / num_groups;
        RUNTIME_CHECK((group_size % 8 == 0), "group_size % 8 != 0");

        dim3 blocks(K / 32, size_n / 8);
        marlin_int4_fp8_preprocess_kernel_awq<<<blocks, 32>>>(
            (const int32_t *)qweight, (int32_t *)output,
            (const int32_t *)qzeros, size_n, K, group_size);
    }
}

extern "C" void marlin_int4_fp8_preprocess_nv(
    void *output, const void *qweight, const void *qzeros,
    int K, int N, int num_groups)
{
    marlin_int4_fp8_preprocess(
        output, qweight, qzeros,
        K, N, num_groups);
}