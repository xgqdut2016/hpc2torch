#ifndef __PER_TOKEN_DEQUANT_FP8_KERNEL_CUH__
#define __PER_TOKEN_DEQUANT_FP8_KERNEL_CUH__
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdio.h>

static constexpr int kWarpSize = 32;

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE>
__device__ void
blockPerTokenDequantFp8SymKernel(Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
    {
        return;
    }
    int tid = token_idx * hidden_dim;
    for (int i = threadIdx.x; i < hidden_dim; i += BLOCK_SIZE)
    {
        float val = static_cast<float>(x_packed[tid + i]) * x_scale[token_idx];
        x[tid + i] = static_cast<Tout>(val);
    }
}

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE, int kTokensPerCTA = 8>
__device__ void
warpPerTokenDequantFp8SymKernel(Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens)
{
    const int warp_id = threadIdx.x / kWarpSize;       // 0‑7  (8 warps)
    const int lane_id = threadIdx.x & (kWarpSize - 1); // 0‑31
    const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
    if (token_id >= num_tokens)
    {
        return;
    }
    int tid = token_id * hidden_dim;
    for (int i = lane_id; i < hidden_dim; i += kWarpSize)
    {
        float val = static_cast<float>(x_packed[tid + i]) * x_scale[token_id];
        x[tid + i] = static_cast<Tout>(val);
    }
}

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE, int kTokensPerCTA = 8>
__global__ void warpPerTokenDequantFp8Sym(
    Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens)
{
    warpPerTokenDequantFp8SymKernel<Tin, Tout, BLOCK_SIZE, kTokensPerCTA>(
        x, x_packed, x_scale, hidden_dim, num_tokens);
}

template <typename Tin, typename Tout, unsigned int BLOCK_SIZE>
__global__ void blockPerTokenDequantFp8Sym(
    Tout *x, const Tin *x_packed, const float *x_scale, int hidden_dim, int num_tokens)
{
    blockPerTokenDequantFp8SymKernel<Tin, Tout, BLOCK_SIZE>(
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
void perTokenDequantFp8(void *x, const void *x_packed, const void *x_scale, const void *x_zero, int hidden_dim,
                        int num_tokens)
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

    if (x_zero == nullptr)
    {
        if (use_warp_kernel)
        {

            // -------- warp‑local ---------------------------------------------------
            constexpr int THREADS = TOKENS_PER_CTA * kWarpSize; // 256
            dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
            dim3 block(THREADS);

            warpPerTokenDequantFp8Sym<__nv_fp8_e4m3, Tdata, THREADS, TOKENS_PER_CTA><<<grid, block, 0, stream>>>(
                (Tdata *)x, (__nv_fp8_e4m3 *)x_packed, (float *)x_scale, hidden_dim, num_tokens);
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

            blockPerTokenDequantFp8Sym<__nv_fp8_e4m3, Tdata, THREADS><<<grid, block, 0, stream>>>(
                (Tdata *)x, (__nv_fp8_e4m3 *)x_packed, (float *)x_scale, hidden_dim, num_tokens);
        }
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}

extern "C" void PerTokenDequantF8_nv(void *x, const void *x_packed, const void *x_scale, const void *x_zero, int hidden_dim,
                                     int num_tokens, int byteSize)
{
    if (byteSize == 2)
    {
        perTokenDequantFp8<1024, half>(x, x_packed, x_scale, x_zero, hidden_dim, num_tokens);
    }
    if (byteSize == 4)
    {
        perTokenDequantFp8<1024, float>(x, x_packed, x_scale, x_zero, hidden_dim, num_tokens);
    }
}

#endif // __PER_TOKEN_DEQUANT_FP8_KERNEL_CUH__
