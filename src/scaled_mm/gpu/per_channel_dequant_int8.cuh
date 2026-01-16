#include <cuda.h>

template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const Tdata *bias, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
    {
        return;
    }
    int idx = row * N + col;
    float output1 = x_scale[row] * w_scale[col] * ((float)y_packed[idx]);

    float output = output1 + (float)bias[col];

    y[idx] = static_cast<Tdata>(output);
}
// y = x_scale * w_scale * y_packed
template <typename Tdata>
__device__ void postSymKernel(Tdata *y, int32_t *y_packed, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
    {
        return;
    }
    int idx = row * N + col;
    float output = x_scale[row] * w_scale[col] * ((float)y_packed[idx]);

    y[idx] = static_cast<Tdata>(output);
}

template <typename Tdata>
__global__ void postSym(
    Tdata *y, int32_t *y_packed, const Tdata *bias, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N)
{
    postSymKernel<Tdata>(y, y_packed, bias, x_packed, x_scale, w_packed, w_scale, M, K, N);
}
template <typename Tdata>
__global__ void postSym(
    Tdata *y, int32_t *y_packed, const int8_t *x_packed, const float *x_scale, const int8_t *w_packed, const float *w_scale, int M, int K, int N)
{
    postSymKernel<Tdata>(y, y_packed, x_packed, x_scale, w_packed, w_scale, M, K, N);
}