#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
template <typename Tdata>
__device__ void postKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
    {
        return;
    }
    int idx = row * N + col;
    float output1 = ((float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx] + K * (float)x_zero[row] * (float)w_zero[col]));
    float output2 = 0.0f;
    float output3 = 0.0f;
    float tmp2 = (float)x_scale[row] * (float)w_scale[col] * (float)w_zero[col];
    float tmp3 = (float)x_scale[row] * (float)x_zero[row] * (float)w_scale[col];
    for (int ind = 0; ind < K; ind++)
    {
        output2 += tmp2 * (float)x_packed[row * K + ind];
        output3 += tmp3 * (float)w_packed[ind * N + col];
    }
    float output = alpha * (output1 - output2 - output3) + beta * (float)c[idx] + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}

template <typename Tdata>
__device__ void postKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
    {
        return;
    }
    int idx = row * N + col;
    float output1 = ((float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx] + K * (float)x_zero[row] * (float)w_zero[col]));
    float output2 = 0.0f;
    float output3 = 0.0f;
    float tmp2 = (float)x_scale[row] * (float)w_scale[col] * (float)w_zero[col];
    float tmp3 = (float)x_scale[row] * (float)x_zero[row] * (float)w_scale[col];
    for (int ind = 0; ind < K; ind++)
    {
        output2 += tmp2 * (float)x_packed[row * K + ind];
        output3 += tmp3 * (float)w_packed[ind * N + col];
    }
    float output = alpha * (output1 - output2 - output3) + beta * (float)c[idx];

    y[idx] = static_cast<Tdata>(output);
}

template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
    {
        return;
    }
    int idx = row * N + col;
    float output1 = (float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx]);

    float output = alpha * output1 + beta * (float)c[idx] + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}
template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
    {
        return;
    }
    int idx = row * N + col;
    float output1 = (float)x_scale[row] * (float)w_scale[col] * ((float)y_packed[idx]);

    float output = alpha * output1 + beta * (float)c[idx];

    y[idx] = static_cast<Tdata>(output);
}

#if defined ENABLE_NVIDIA_API
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
void int8Gemm(
    const int8_t *x_packed, // [M, K], RowMajor
    const int8_t *w_packed, // [K, N], RowMajor
    int32_t *y_packed,      // [M, N], RowMajor
    int M, int N, int K,
    cudaStream_t stream)
{
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementAccumulator = int32_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator>;

    // RowMajor: leading dimension = number of columns
    int lda = K;
    int ldb = N;
    int ldc = N;

    Gemm gemm_op;

    typename Gemm::Arguments args(
        {M, N, K},       // GEMM shape
        {x_packed, lda}, // A: [M, K]
        {w_packed, ldb}, // B: [K, N]
        {y_packed, ldc}, // C (ignored when beta = 0)
        {y_packed, ldc}, // D
        {1, 0}           // alpha = 1, beta = 0
    );

    cutlass::Status status = gemm_op(args, stream);
    if (status != cutlass::Status::kSuccess)
    {
        printf("[CUTLASS RowMajor int8 GEMM] failed: %d\n", int(status));
    }
}
#endif

template <typename Tdata>
__global__ void post(
    Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta)
{
    postKernel<Tdata>(y, y_packed, c, bias, x_packed, x_scale, x_zero, w_packed, w_scale, w_zero, M, K, N, alpha, beta);
}
template <typename Tdata>
__global__ void post(
    Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta)
{
    postKernel<Tdata>(y, y_packed, c, x_packed, x_scale, x_zero, w_packed, w_scale, w_zero, M, K, N, alpha, beta);
}

template <typename Tdata>
__global__ void postSym(
    Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta)
{
    postSymKernel<Tdata>(y, y_packed, c, bias, x_packed, x_scale, w_packed, w_scale, M, K, N, alpha, beta);
}
template <typename Tdata>
__global__ void postSym(
    Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta)
{
    postSymKernel<Tdata>(y, y_packed, c, x_packed, x_scale, w_packed, w_scale, M, K, N, alpha, beta);
}

template <unsigned int BLOCK_SIZE, typename Tdata>
void launchKernel(void *y,
                  const void *c,
                  const void *bias,
                  const void *x_packed,
                  const void *x_scale,
                  const void *x_zero,
                  const void *w_packed,
                  const void *w_scale,
                  const void *w_zero,
                  int M, int K, int N, float alpha, float beta)
{
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("流创建失败: %s\n", cudaGetErrorString(err));
        // 错误处理
    }

    int32_t *y_packed;
    cudaMalloc((void **)&y_packed, M * N * sizeof(int32_t));

#if defined ENABLE_NVIDIA_API
    int8Gemm((int8_t *)x_packed, (int8_t *)w_packed, y_packed, M, N, K, stream);
#elif defined ENABLE_QL_API
    const int32_t alpha_I = 1;
    const int32_t beta_I = 0;
    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, // A = w_packed, column-major view
        CUBLAS_OP_N, // B = x_packed, column-major view
        N,           // m = N
        M,           // n = M
        K,           // k = K
        &alpha_I,
        w_packed, CUDA_R_8I, N, // lda = m = N
        x_packed, CUDA_R_8I, K, // ldb = k = K
        &beta_I,
        y_packed, CUDA_R_32I, N, // ldc = m = N
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
#endif

    // int8_t *host_x_packed;
    // int8_t *host_w_packed;
    // int32_t *host_y_packed;
    // host_x_packed = (int8_t *)malloc(M * K * sizeof(int8_t));
    // cudaMemcpy(host_x_packed, x_packed, M * K * sizeof(int8_t), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < M * K; i++)
    // {
    //     printf("%d ", host_x_packed[i]);
    //     if ((i + 1) % K == 0)
    //     {
    //         printf("\n");
    //     }
    // }
    // printf("\n");
    // host_w_packed = (int8_t *)malloc(K * N * sizeof(int8_t));
    // cudaMemcpy(host_w_packed, w_packed, K * N * sizeof(int8_t), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < K * N; i++)
    // {
    //     printf("%d ", host_w_packed[i]);
    //     if ((i + 1) % N == 0)
    //     {
    //         printf("\n");
    //     }
    // }
    // printf("\n");
    // host_y_packed = (int32_t *)malloc(M * N * sizeof(int32_t));
    // cudaMemcpy(host_y_packed, y_packed, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < M * N; i++)
    // {
    //     printf("%d ", host_y_packed[i]);
    //     if ((i + 1) % N == 0)
    //     {
    //         printf("\n");
    //     }
    // }
    // printf("\n");
    constexpr unsigned int BLOCK_SIZE_x = 32;
    constexpr unsigned int BLOCK_SIZE_y = 32;

    int num_block_x = (N + BLOCK_SIZE_x - 1) / BLOCK_SIZE_x;
    int num_block_y = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
    dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    if (bias == nullptr)
    {
        if (x_zero == nullptr && w_zero == nullptr)
        {
            postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>((Tdata *)y, y_packed, (Tdata *)c, (int8_t *)x_packed, (Tdata *)x_scale, (int8_t *)w_packed, (Tdata *)w_scale, M, K, N, alpha, beta);
        }
        else
        {
            post<Tdata><<<grid_dim, block_dim, 0, stream>>>((Tdata *)y, y_packed, (Tdata *)c, (int8_t *)x_packed, (Tdata *)x_scale, (Tdata *)x_zero, (int8_t *)w_packed, (Tdata *)w_scale, (Tdata *)w_zero, M, K, N, alpha, beta);
        }
    }
    else
    {
        if (x_zero == nullptr && w_zero == nullptr)
        {
            postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>((Tdata *)y, y_packed, (Tdata *)c, (Tdata *)bias, (int8_t *)x_packed, (Tdata *)x_scale, (int8_t *)w_packed, (Tdata *)w_scale, M, K, N, alpha, beta);
        }
        else
        {
            post<Tdata><<<grid_dim, block_dim, 0, stream>>>((Tdata *)y, y_packed, (Tdata *)c, (Tdata *)bias, (int8_t *)x_packed, (Tdata *)x_scale, (Tdata *)x_zero, (int8_t *)w_packed, (Tdata *)w_scale, (Tdata *)w_zero, M, K, N, alpha, beta);
        }
    }
    cudaFree(y_packed);
    err = cudaStreamDestroy(stream);
    if (err != cudaSuccess)
    {
        printf("流销毁失败: %s\n", cudaGetErrorString(err));
        // 错误处理
    }
}
extern "C" void linear_nv(void *d,
                          const void *c,
                          const void *bias,
                          const void *x,
                          const void *x_scale,
                          const void *x_zero,
                          const void *weights,
                          const void *weights_scale,
                          const void *weights_zero,
                          int M, int K, int N, float alpha, float beta, int byteSize)
{
    if (byteSize == 2)
    {
        launchKernel<1024, half>(d, c, bias, x, x_scale, x_zero, weights, weights_scale, weights_zero, M, K, N, alpha, beta);
    }
    if (byteSize == 4)
    {
        launchKernel<1024, float>(d, c, bias, x, x_scale, x_zero, weights, weights_scale, weights_zero, M, K, N, alpha, beta);
    }
}
