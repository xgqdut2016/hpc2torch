#ifndef __PER_TENSOR_DEQUANT_FP8_KERNEL_CUH__
#define __PER_TENSOR_DEQUANT_FP8_KERNEL_CUH__
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdio.h>

template <typename Tin, typename Tout>
__device__ void
perTensorDequantFp8SymKernel(Tout *x, const Tin *x_packed, const float *x_scale, int num_elements)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;
    for (int i = gid; i < num_elements; i += grid_size)
    {
        float val = static_cast<float>(x_packed[i]) * x_scale[0];
        x[i] = static_cast<Tout>(val);
    }
}

template <typename Tin, typename Tout>
__global__ void perTensorDequantFp8Sym(
    Tout *x, const Tin *x_packed, const float *x_scale, int num_elements)
{
    perTensorDequantFp8SymKernel<Tin, Tout>(
        x,
        x_packed,
        x_scale,
        num_elements);
}

template <unsigned int BLOCK_SIZE, typename Tdata>
void perTensorDequantFp8(void *x, const void *x_packed, const void *x_scale, const void *x_zero, int num_elements)
{
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }
#ifdef ENABLE_NVIDIA_API
    constexpr unsigned int block_size = 256;
#else
    constexpr unsigned int block_size = BLOCK_SIZE;
#endif
    int num_blocks = min((static_cast<int>(num_elements) + block_size - 1) / block_size, 1024);

    dim3 grid(num_blocks);
    dim3 block(block_size);
    if (x_zero == nullptr)
    {

        perTensorDequantFp8Sym<__nv_fp8_e4m3, Tdata>
            <<<grid, block, 0, stream>>>((Tdata *)x, (__nv_fp8_e4m3 *)x_packed, (float *)x_scale, num_elements);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}

extern "C" void PerTensorDequantF8_nv(void *x, const void *x_packed, const void *x_scale, const void *x_zero, int num_elements, int byteSize)
{
    if (byteSize == 2)
    {
        perTensorDequantFp8<1024, half>(x, x_packed, x_scale, x_zero, num_elements);
    }
    if (byteSize == 4)
    {
        perTensorDequantFp8<1024, float>(x, x_packed, x_scale, x_zero, num_elements);
    }
}

#endif // __PER_TENSOR_DEQUANT_FP8_KERNEL_CUH__
