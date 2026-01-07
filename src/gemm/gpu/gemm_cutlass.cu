#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
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

// -----------------------------------------------------------------------------
// GEMM with stream create/sync/destroy
// -----------------------------------------------------------------------------
template <typename Tin, typename Tout, typename Acc>
void GemmCutlass(
    const void *x_packed,
    const void *w_packed,
    void *y_packed,
    int M, int K, int N,
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

    // 调用 GEMM
    GemmSimtRowMajor<Tin, Tout, Acc>(
        static_cast<const Tin *>(x_packed),
        static_cast<const Tin *>(w_packed),
        static_cast<Tout *>(y_packed),
        M, K, N,
        alpha, beta,
        stream);

    // 同步
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(stream);
}

// -----------------------------------------------------------------------------
// C接口
// -----------------------------------------------------------------------------
extern "C" void gemm_cutlass(
    const void *x_packed,
    const void *w_packed,
    void *y_packed,
    int M, int K, int N,
    float alpha,
    float beta,
    int byteSize)
{
    if (byteSize == 4)
    {
        // float32 -> float32
        GemmCutlass<float, float, float>(x_packed, w_packed, y_packed, M, K, N, alpha, beta);
    }
    else if (byteSize == 2)
    {
        // fp16 (uint16_t) -> float32
        // fp16 -> fp32
        GemmCutlass<cutlass::half_t, float, float>(
            reinterpret_cast<const cutlass::half_t *>(x_packed),
            reinterpret_cast<const cutlass::half_t *>(w_packed),
            reinterpret_cast<float *>(y_packed),
            M, K, N, alpha, beta);
    }
    else if (byteSize == 1)
    {
        // int8 -> int32
        GemmCutlass<int8_t, int32_t, int32_t>(x_packed, w_packed, y_packed, M, K, N, alpha, beta);
    }
    else
    {
        printf("Unsupported byteSize: %d\n", byteSize);
    }
}
