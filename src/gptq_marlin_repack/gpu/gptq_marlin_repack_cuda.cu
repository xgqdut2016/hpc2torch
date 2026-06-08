#if defined ENABLE_NVIDIA_API
#include "kernel.cuh"
#include <cassert>

template <int const num_threads, int const num_bits, bool const has_perm,
          bool is_a_8bit>
__global__ void gptqMarlinRepackKernel(
    const uint32_t *__restrict__ b_q_weight_ptr,
    const uint32_t *__restrict__ perm_ptr, uint32_t *__restrict__ out_ptr,
    int size_k, int size_n)
{

    marlin::gptq_marlin_repack_kernel<num_threads, num_bits, has_perm, is_a_8bit>(
        b_q_weight_ptr, perm_ptr, out_ptr,
        size_k, size_n);
}

#define CALL_IF(NUM_BITS, HAS_PERM, IS_A_8BIT)                            \
    else if (num_bits == NUM_BITS && has_perm == HAS_PERM &&              \
             is_a_8bit == IS_A_8BIT)                                      \
    {                                                                     \
        cudaFuncSetAttribute(                                             \
            gptqMarlinRepackKernel<marlin::repack_threads, NUM_BITS,      \
                                   HAS_PERM, IS_A_8BIT>,                  \
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem); \
        gptqMarlinRepackKernel<marlin::repack_threads, NUM_BITS,          \
                               HAS_PERM, IS_A_8BIT>                       \
            <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>( \
                b_q_weight_ptr, perm_ptr, out_ptr, size_k, size_n);       \
    }

void gptqMarlinRepack(uint32_t *out_ptr, const uint32_t *b_q_weight_ptr, const uint32_t *perm_ptr,
                      int64_t size_k, int64_t size_n, int64_t num_bits,
                      bool is_a_8bit, bool has_perm)
{
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("流创建失败: %s\n", cudaGetErrorString(err));
        // 错误处理
    }
    // Verify compatibility with marlin tile of 16x64
    if (size_k % marlin::tile_k_size != 0)
    {
        std::cout << "size_k = " << size_k << " is not divisible by tile_k_size = " << marlin::tile_k_size << std::endl;
    }
    if (size_n % marlin::tile_n_size != 0)
    {
        std::cout << "size_n = " << size_n << " is not divisible by tile_n_size = " << marlin::tile_n_size << std::endl;
    }
    if (num_bits != 4 && num_bits != 8)
    {
        std::cout << "num_bits must be 4 or 8. Got = " << num_bits << std::endl;
    }
    // size_k / pack_factor == b_q_weight.size(0); size_n = b_q_weights.size(1)
    // out.shape = [size_k / marlin::tile_size, size_n * marlin::tile_size / pack_factor]

    // Get dev info
    int device_id = 0;

    int blocks;
    cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, device_id);

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    assert(max_shared_mem > 0 && "max_shared_mem must be greater than 0");

    if (false)
    {
    }
    CALL_IF(4, false, false)
    CALL_IF(4, true, false)
    CALL_IF(8, false, false)
    CALL_IF(8, true, false)

    CALL_IF(4, false, true)
    CALL_IF(8, false, true)
    else
    {
        fprintf(stderr, "Unsupported repack config: num_bits = %ld, has_perm = %s, is_a_8bit = %s\n",
                num_bits,
                has_perm ? "true" : "false",
                is_a_8bit ? "true" : "false");
        assert(false);
    }
    err = cudaStreamDestroy(stream);
    if (err != cudaSuccess)
    {
        printf("流销毁失败: %s\n", cudaGetErrorString(err));
        // 错误处理
    }
}

extern "C" void gptq_marlin_repack_nv(void *output, const void *input, const void *perm,
                                      int size_k, int size_n, int num_bits,
                                      bool is_a_8bit, bool has_perm)
{

    gptqMarlinRepack((uint32_t *)output, (const uint32_t *)input, (const uint32_t *)perm,
                     static_cast<int64_t>(size_k),
                     static_cast<int64_t>(size_n),
                     static_cast<int64_t>(num_bits),
                     is_a_8bit, has_perm);
}
#endif
