#ifndef __PER_GROUP_QUANT_FP4_KERNEL_CUH__
#define __PER_GROUP_QUANT_FP4_KERNEL_CUH__
#include "kernel.cuh"

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __launch_bounds__(512, 4) cvt_fp16_to_fp4(
#else
cvt_fp16_to_fp4(
#endif
        int32_t numRows, int32_t numCols, Type const *in, float const *SFScale, uint32_t *out, uint32_t *SFout)
{
    using PackedVec = PackedVec<Type>;
    static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
    static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

    // Get the global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is
    // (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    // Input tensor row/col loops.
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; colIdx += blockDim.x)
        {
            int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
            PackedVec in_vec = reinterpret_cast<PackedVec const *>(in)[inOffset];
            // Get the output tensor offset.
            // Same as inOffset because 8 elements are packed into one uint32_t.
            int64_t outOffset = inOffset;
            auto &out_pos = out[outOffset];

            auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

            out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
        }
    }
}

template <typename T>
void invokeFP4Quantization(
    int m,
    int n,
    T const *input,
    float const *SFScale,
    int64_t *output,
    int32_t *SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream)
{
    // Grid, Block size.
    // Each thread converts 8 values.
    dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = 2048 / block.x;
    dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

    // Launch the cvt kernel.
    if (useUE8M0)
    {
        cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
            m, n, input, SFScale, reinterpret_cast<uint32_t *>(output), reinterpret_cast<uint32_t *>(SFOuput));
    }
    else
    {
        cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
            m, n, input, SFScale, reinterpret_cast<uint32_t *>(output), reinterpret_cast<uint32_t *>(SFOuput));
    }
}

inline int getMultiProcessorCount()
{
    static int multi_processor_count = []()
    {
        int device_id = 0;
        int count = 0;

        // Get the current CUDA device ID
        CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

        // Get the number of multiprocessors for the current device
        CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id));

        return count; // Initialize the static variable
    }();

    return multi_processor_count; // Return the cached value on subsequent calls
}

template <unsigned int BLOCK_SIZE, typename Tdata>
void PerGroupQuantF4Kernel(int64_t *output, int32_t *output_scale, const Tdata *input, const float *input_global_scale, int M, int N)
{
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }

    bool useUE8M0 = false;
    int multiProcessorCount = getMultiProcessorCount();
    invokeFP4Quantization(M, N, input, input_global_scale, output, output_scale, useUE8M0, multiProcessorCount, stream);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}

enum DataType
{
    DT_FLOAT16 = 0,
    DT_BFLOAT16 = 1,
};

extern "C" void PerGroupQuantF4_nv(void *output,
                                   void *output_scale,
                                   const void *input,
                                   const void *input_global_scale,
                                   int M,
                                   int N, int dataType)
{
    switch (static_cast<DataType>(dataType))
    {
    case DT_FLOAT16:
        PerGroupQuantF4Kernel<1024, half>((int64_t *)output, (int32_t *)output_scale, (half *)input, (float *)input_global_scale, M, N);
        break;
    case DT_BFLOAT16:
        PerGroupQuantF4Kernel<1024, __nv_bfloat16>((int64_t *)output, (int32_t *)output_scale, (__nv_bfloat16 *)input, (float *)input_global_scale, M, N);
        break;

    default:
        printf("Unsupported datatype\n");
        break;
    }
}

#endif // __PER_GROUP_QUANT_FP4_KERNEL_CUH__
