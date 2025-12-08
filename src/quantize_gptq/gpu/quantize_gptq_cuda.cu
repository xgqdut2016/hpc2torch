#if defined ENABLE_NVIDIA_API
#include "gptq_marlin.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
extern "C" void caculate_cuda(void *C, void *A, void *packed_weights, void *b_scale, void *zero,
                              int M, int K, int N, int group_size)
{
    int device_id = 0;
    int bits = 4;
    int bytes = 2;
    int num_groups = (group_size == -1 ? 1 : K / group_size);
    int max_par = gptq_marlin::max_par;
    size_t min_workspace_size = N / gptq_marlin::min_thread_n * max_par * sizeof(int) + M * K * bytes;
    void *workspace;
    cudaMalloc(&workspace, min_workspace_size);
    cudaStream_t stream;

    // 初始化 stream
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("Failed to create stream (error code %s)!\n",
               cudaGetErrorString(err));
    }
    gptq_marlin::gptq_marlin_mm_fp16(C, A, packed_weights, b_scale,
                                     M, N, K,
                                     workspace, bits,
                                     num_groups, group_size,
                                     device_id, stream);
    cudaStreamDestroy(stream);
    cudaFree(workspace);
}
#endif