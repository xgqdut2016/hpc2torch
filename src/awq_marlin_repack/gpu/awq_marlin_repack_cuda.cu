#if defined ENABLE_NVIDIA_API
#include "kernel.cuh"
#include <cassert>

template <int const num_threads, int const num_bits, bool is_a_8bit>
__global__ void awqMarlinRepackKernel(
    uint32_t const *__restrict__ b_q_weight_ptr, uint32_t *__restrict__ out_ptr,
    int size_k, int size_n)
{
    marlin::awq_marlin_repack_kernel<num_threads, num_bits, is_a_8bit>(
        b_q_weight_ptr, out_ptr,
        size_k, size_n);
}

#define CALL_IF(NUM_BITS, IS_A_8BIT)                                      \
    else if (num_bits == NUM_BITS && is_a_8bit == IS_A_8BIT)              \
    {                                                                     \
        cudaFuncSetAttribute(                                             \
            awqMarlinRepackKernel<marlin::repack_threads, NUM_BITS,       \
                                  IS_A_8BIT>,                             \
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem); \
        awqMarlinRepackKernel<marlin::repack_threads, NUM_BITS,           \
                              IS_A_8BIT>                                  \
            <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>( \
                b_q_weight_ptr, out_ptr, size_k, size_n);                 \
    }

void awqMarlinRepack(uint32_t *out_ptr, const uint32_t *b_q_weight_ptr, int64_t size_k,
                     int64_t size_n, int64_t num_bits,
                     bool is_a_8bit)
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

    int const pack_factor = 32 / num_bits;

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
    CALL_IF(4, false)
    CALL_IF(8, false)
    CALL_IF(4, true)
    CALL_IF(8, true)
    else
    {
        assert(false && "Unsupported repack config: num_bits, is_a_8bit");
    }
    err = cudaStreamDestroy(stream);
    if (err != cudaSuccess)
    {
        printf("流销毁失败: %s\n", cudaGetErrorString(err));
        // 错误处理
    }
}

extern "C" void awq_marlin_repack_nv(void *output, const void *input,
                                     int size_k, int size_n, int num_bits,
                                     bool is_a_8bit)
{

    awqMarlinRepack((uint32_t *)output, (const uint32_t *)input,
                    static_cast<int64_t>(size_k),
                    static_cast<int64_t>(size_n),
                    static_cast<int64_t>(num_bits),
                    is_a_8bit);
}
#endif
