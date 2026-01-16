#include <type_traits>
#include <string>
#include <cublas_v2.h>
#include "per_channel_dequant_int8.cuh"
#if defined ENABLE_NVIDIA_API
#include "int8_gemm_kernel.cuh"

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
#endif

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
#if defined ENABLE_NVIDIA_API
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
#elif defined ENABLE_QL_API
    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    int32_t *y_packed;
    cudaMalloc((void **)&y_packed, M * N * sizeof(int32_t));
    const int32_t alpha_I = 1;
    const int32_t beta_I = 0;

    cublasGemmEx(
        handle,
        CUBLAS_OP_T, // A = b^T : [N, K]
        CUBLAS_OP_N, // B = a^T viewed column-major : [K, M]
        N,           // m
        M,           // n
        K,           // k
        &alpha_I,
        b, CUDA_R_8I, lda,
        a, CUDA_R_8I, ldb,
        &beta_I,
        y_packed, CUDA_R_32I, ldo,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT);

    constexpr unsigned int BLOCK_SIZE_x = 32;
    constexpr unsigned int BLOCK_SIZE_y = 32;

    int num_block_x = (N + BLOCK_SIZE_x - 1) / BLOCK_SIZE_x;
    int num_block_y = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
    dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    if (bias == nullptr)
    {
        postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>((Tdata *)out, y_packed, a, a_scale, b, b_scale, M, K, N);
    }
    else
    {
        postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>((Tdata *)out, y_packed, (Tdata *)bias, a, a_scale, b, b_scale, M, K, N);
    }
    cublasDestroy(handle);
    cudaFree(y_packed);
#endif
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
    const void *w_packed, // 按照列主元优先排布
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
