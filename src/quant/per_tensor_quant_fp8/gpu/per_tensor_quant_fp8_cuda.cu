#ifndef __PER_TENSOR_QUANT_FP8_KERNEL_CUH__
#define __PER_TENSOR_QUANT_FP8_KERNEL_CUH__

#include "kernel.cuh"

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
        if (is_static == false)
        {
            blockPerTensorAbsmaxSym<Tdata, block_size>
                <<<grid, block, 0, stream>>>((float *)x_scale, (Tdata *)x, num_elements);
        }
        blockPerTensorQuantF8Sym<Tdata, __nv_fp8_e4m3, block_size>
            <<<grid, block, 0, stream>>>((__nv_fp8_e4m3 *)x_packed, (float *)x_scale, (Tdata *)x, num_elements);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
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
