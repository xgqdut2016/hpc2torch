#include <cublas_v2.h>
#include <string>

template <typename Tdata>
bool matmulLaunch(void const *dA, void const *dB, void *dC, int batch_size, int M, int K, int N, int dimA, int dimB, float alpha, float beta, std::string computeTypeStr)
{
    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    cudaDataType_t data_type;
    if constexpr (std::is_same<Tdata, uint16_t>::value)
    {
        data_type = CUDA_R_16F;
    }
    else if constexpr (std::is_same<Tdata, float>::value)
    {
        data_type = CUDA_R_32F;
    }
    cublasComputeType_t compute_type;
    if (computeTypeStr == "tf32")
    {
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
    else if (computeTypeStr == "bf16")
    {
        compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
    }
    else if (computeTypeStr == "fp16")
    {
        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    }

    bool transA = false;
    bool transB = false;
    auto opA =
        transA ? CUBLAS_OP_T : CUBLAS_OP_N; // BLAS_N = col major
    auto opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = transA ? M : K, ldb = transB ? K : N,
              ldc = N;

    cublasStatus_t stat;
    long long strideA =
        (dimA == 2 ||
         (dimA == 3 && batch_size == 1))
            ? 0 // Broadcast the batch dimension if batch size is 1
            : M * K;
    long long strideB =
        (dimB == 2 ||
         (dimB == 3 && batch_size == 1))
            ? 0 // Broadcast the batch dimension if batch size is 1
            : N * K;
    int algo = CUBLAS_GEMM_DEFAULT;
    if constexpr (std::is_same<Tdata, uint16_t>::value)
    {
        half alpha_half = static_cast<half>(alpha);
        half beta_half = static_cast<half>(beta);
        stat = cublasGemmStridedBatchedEx(
            handle, opB, opA, N, M, K, &alpha_half,
            dB, data_type, ldb, strideB, dA, data_type, lda,
            strideA, &beta_half, dC, data_type, ldc, M * N, batch_size,
            compute_type, (cublasGemmAlgo_t)algo);
    }
    else if constexpr (std::is_same<Tdata, float>::value)
    {
        stat = cublasGemmStridedBatchedEx(
            handle, opB, opA, N, M, K, &alpha,
            dB, data_type, ldb, strideB, dA, data_type, lda,
            strideA, &beta, dC, data_type, ldc, M * N, batch_size,
            compute_type, (cublasGemmAlgo_t)algo);
    }
    return (stat == CUBLAS_STATUS_SUCCESS);
}
extern "C" void matmul_cudnn(void const *dA, void const *dB, void *dC, int *a_shape, int *b_shape, int *c_shape,
                             int aDim, int bDim, int cDim,
                             float alpha, float beta, int byteSize)
{
    std::string computeTypeStr = "tf32";
    bool condition;
    int batch_size, M, K, N;
    if (aDim == 3)
    {
        batch_size = a_shape[0];
        M = a_shape[1];
        K = a_shape[2];
        N = b_shape[1];
    }
    else if (aDim == 2)
    {
        batch_size = 1;
        M = a_shape[0];
        K = a_shape[1];
        N = b_shape[1];
    }
    if (byteSize == 2)
    {
        condition = matmulLaunch<uint16_t>(dA, dB, dC, batch_size, M, K, N, aDim, bDim, alpha, beta, computeTypeStr);
    }
    else if (byteSize == 4)
    {
        condition = matmulLaunch<float>(dA, dB, dC, batch_size, M, K, N, aDim, bDim, alpha, beta, computeTypeStr);
    }
}
