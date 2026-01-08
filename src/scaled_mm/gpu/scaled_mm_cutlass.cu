#if defined ENABLE_NVIDIA_API
#include "int8_gemm_kernel.cuh"
#include <type_traits>
#include <string>

inline int getSMVersion()
{
    int device{-1};
    cudaGetDevice(&device);
    int sm_major = 0;
    int sm_minor = 0;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
    return sm_major * 10 + sm_minor;
}

template <typename Tdata>
void int8calculate(
    void *out,
    const void *bias,
    const int8_t *a,
    const float *a_scale,
    const int8_t *b,
    const float *b_scale,
    int M, int K, int N,
    int lda, int ldb, int ldo)
{
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }

    auto sm_version = getSMVersion();
    if (sm_version >= 75 && sm_version < 80)
    {
        if constexpr (std::is_same_v<Tdata, __half>)
        {
            sm75_dispatch_shape<cutlass::half_t, cutlass::arch::Sm75, cutlass::gemm::GemmShape<8, 8, 16>>(
                out, a, b, a_scale, b_scale, bias, M, N, K, lda, ldb, ldo, stream);
        }
    }
    else if (sm_version >= 80 && sm_version < 90)
    {
        // sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
        if (sm_version == 86 || sm_version == 89)
        {
            if constexpr (std::is_same_v<Tdata, __half>)
            {
                sm89_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, M, N, K, lda, ldb, ldo, stream);
            }
            else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>)
            {
                sm89_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, M, N, K, lda, ldb, ldo, stream);
            }
        }
        else
        {
            if constexpr (std::is_same_v<Tdata, __half>)
            {
                sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, M, N, K, lda, ldb, ldo, stream);
            }
            else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>)
            {
                sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                    out, a, b, a_scale, b_scale, bias, M, N, K, lda, ldb, ldo, stream);
            }
        }
    }
    else if (sm_version == 90)
    {
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
        // cutlass 3.x
        if constexpr (std::is_same_v<Tdata, __half>)
        {
            sm90_dispatch_shape<cutlass::half_t>(
                out, a, b, a_scale, b_scale, bias,
                M, N, K, lda, ldb, ldo,
                stream);
        }
        else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>)
        {
            sm90_dispatch_shape<cutlass::bfloat16_t>(
                out, a, b, a_scale, b_scale, bias,
                M, N, K, lda, ldb, ldo,
                stream);
        }
#else
        // // fallback to cutlass 2.x
        if constexpr (std::is_same_v<Tdata, __half>)
        {
            sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                out, a, b, a_scale, b_scale, bias, M, N, K, lda, ldb, ldo, stream);
        }
        else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>)
        {
            sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
                out, a, b, a_scale, b_scale, bias, M, N, K, lda, ldb, ldo, stream);
        }
#endif
    }
    // 同步
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
    DT_FLOAT32 = 2,
};
extern "C" void int8_scaled_gemm_cutlass(
    void *out,
    const void *bias,
    const void *x_packed,
    const void *x_scale,
    const void *w_packed,
    const void *w_scale,
    int M, int K, int N,
    int dataType)
{
    int lda = K;
    int ldb = K;
    int ldo = N;
    switch (static_cast<DataType>(dataType))
    {
    case DT_FLOAT16:
        int8calculate<__half>(
            out,
            bias,
            (int8_t *)x_packed,
            (float *)x_scale,
            (int8_t *)w_packed,
            (float *)w_scale,
            M, K, N,
            lda, ldb, ldo);
        break;
    case DT_BFLOAT16:
        int8calculate<__nv_bfloat16>(
            out,
            bias,
            (int8_t *)x_packed,
            (float *)x_scale,
            (int8_t *)w_packed,
            (float *)w_scale,
            M, K, N,
            lda, ldb, ldo);
        break;
    default:
        printf("Unsupported datatype\n");
        break;
    }
}
#endif
