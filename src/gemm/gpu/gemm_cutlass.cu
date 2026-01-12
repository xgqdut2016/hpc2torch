#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/half.h"
#include <cuda_runtime.h>
#include <cstdio>

// -----------------------------------------------------------------------------
// GEMM kernel wrapper
// -----------------------------------------------------------------------------
template <typename Tin, typename Tout, typename Acc>
void GemmSimtRowMajor(
    const Tin *x_packed,
    const Tin *w_packed,
    Tout *y_packed,
    int M, int K, int N,
    float alpha,
    float beta,
    cudaStream_t stream)
{
    using ElementA = Tin;
    using ElementB = Tin;
    using ElementC = Tout;
    using ElementAccumulator = Acc;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // SIMT GEMM
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm75>;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Epilogue uses ElementCompute to apply alpha/beta
    using EpilogueOp = typename Gemm::EpilogueOutputOp;
    using ElementCompute = typename EpilogueOp::ElementCompute;

    typename Gemm::Arguments args{
        problem_size,
        {x_packed, K}, // A: M x K
        {w_packed, N}, // B: K x N
        {y_packed, N}, // C: M x N
        {y_packed, N}, // D: M x N (output)
        {static_cast<ElementCompute>(alpha), static_cast<ElementCompute>(beta)}};

    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
    {
        printf("[CUTLASS SIMT] initialize failed: %d\n", int(status));
        return;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess)
    {
        printf("[CUTLASS SIMT] run failed: %d\n", int(status));
        return;
    }
}
template <typename Tin, typename Tout, typename Acc>
void GemmSimtRowMajorBatched(
    const Tin *x_packed, // [batch, M, K]
    const Tin *w_packed, // [batch, K, N]
    Tout *y_packed,      // [batch, M, N]
    int batch,
    int M, int K, int N,
    float alpha,
    float beta,
    cudaStream_t stream)
{

    using ElementA = Tin;
    using ElementB = Tin;
    using ElementC = Tout;
    using ElementAccumulator = Acc;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm75>;

    using TensorRefA = typename Gemm::TensorRefA;
    using TensorRefB = typename Gemm::TensorRefB;
    using TensorRefC = typename Gemm::TensorRefC;
    using TensorRefD = typename Gemm::TensorRefD;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    using EpilogueOp = typename Gemm::EpilogueOutputOp;
    using ElementCompute = typename EpilogueOp::ElementCompute;

    typename Gemm::Arguments args(
        problem_size,

        // A
        TensorRefA{x_packed, LayoutA(K)},
        int64_t(M) * K,

        // B
        TensorRefB{w_packed, LayoutB(N)},
        int64_t(K) * N,

        // C
        TensorRefC{y_packed, LayoutC(N)},
        int64_t(M) * N,

        // D
        TensorRefD{y_packed, LayoutC(N)},
        int64_t(M) * N,

        // epilogue
        {static_cast<ElementCompute>(alpha),
         static_cast<ElementCompute>(beta)},

        // batch
        batch);

    Gemm gemm_op;

    auto status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
    {
        printf("[CUTLASS GemmBatched] initialize failed: %d\n", int(status));
        return;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess)
    {
        printf("[CUTLASS GemmBatched] run failed: %d\n", int(status));
        return;
    }
}

// -----------------------------------------------------------------------------
// GEMM with stream create/sync/destroy
// -----------------------------------------------------------------------------
template <typename Tin, typename Tout, typename Acc>
void GemmCutlass(
    const void *x_packed,
    const void *w_packed,
    void *y_packed,
    int batch_size, int M, int K, int N,
    float alpha,
    float beta)
{
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return;
    }

    if (batch_size == 1)
    {
        GemmSimtRowMajor<Tin, Tout, Acc>(
            static_cast<const Tin *>(x_packed),
            static_cast<const Tin *>(w_packed),
            static_cast<Tout *>(y_packed),
            M, K, N,
            alpha, beta,
            stream);
    }
    else
    {
        GemmSimtRowMajorBatched<Tin, Tout, Acc>(
            static_cast<const Tin *>(x_packed),
            static_cast<const Tin *>(w_packed),
            static_cast<Tout *>(y_packed),
            batch_size, M, K, N,
            alpha, beta,
            stream);
    }

    // 同步
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}

// -----------------------------------------------------------------------------
enum DataType
{
    DT_FLOAT16 = 0,
    DT_BFLOAT16 = 1,
    DT_FLOAT32 = 2,
    DT_INT8 = 3,
};
// -----------------------------------------------------------------------------
extern "C" void gemm_cutlass(
    const void *x_packed,
    const void *w_packed,
    void *y_packed,
    int batch_size, int M, int K, int N,
    float alpha,
    float beta,
    int dataType)
{
    switch (static_cast<DataType>(dataType))
    {
    case DT_FLOAT16:
        GemmCutlass<cutlass::half_t, float, float>(
            reinterpret_cast<const cutlass::half_t *>(x_packed),
            reinterpret_cast<const cutlass::half_t *>(w_packed),
            reinterpret_cast<float *>(y_packed),
            batch_size, M, K, N, alpha, beta);
        break;
    case DT_BFLOAT16:
        GemmCutlass<cutlass::bfloat16_t, float, float>(
            reinterpret_cast<const cutlass::bfloat16_t *>(x_packed),
            reinterpret_cast<const cutlass::bfloat16_t *>(w_packed),
            reinterpret_cast<float *>(y_packed),
            batch_size, M, K, N, alpha, beta);
        break;
    case DT_FLOAT32:
        GemmCutlass<float, float, float>(x_packed, w_packed, y_packed, batch_size, M, K, N, alpha, beta);
        break;
    case DT_INT8:
        GemmCutlass<int8_t, int32_t, int32_t>(x_packed, w_packed, y_packed, batch_size, M, K, N, alpha, beta);
        break;
    default:
        printf("Unsupported datatype\n");
        break;
    }
}
