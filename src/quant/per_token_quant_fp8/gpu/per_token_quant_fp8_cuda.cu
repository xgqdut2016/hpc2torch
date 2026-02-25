#ifndef __PER_TOKEN_QUANT_FP8_KERNEL_CUH__
#define __PER_TOKEN_QUANT_FP8_KERNEL_CUH__

#include "kernel.cuh"

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
