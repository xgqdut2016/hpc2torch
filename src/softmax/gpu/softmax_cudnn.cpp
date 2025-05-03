#include <stdio.h>
#include <cudnn.h>

template <typename T>
void softmaxCudnnDevice(cudnnHandle_t &handle, void const *input, void *output, int *shape, int ndim)
{
    int dim_array[4] = {1, 1, 1, 1}; // cudnn只能处理4维及以下向量
    for (int i = 0; i < ndim; i++)
    {
        dim_array[4 - ndim + i] = shape[i];
    }
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnDataType_t cudnnDataType = CUDNN_DATA_FLOAT;
    if (sizeof(T) == 2)
    {
        cudnnDataType = CUDNN_DATA_HALF;
    }
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);

    cudnnSetTensor4dDescriptor(
        inputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, dim_array[0],
        dim_array[1], dim_array[2], dim_array[3]);
    cudnnSetTensor4dDescriptor(
        outputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, dim_array[0],
        dim_array[1], dim_array[2], dim_array[3]);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
    cudnnSoftmaxForward(
        handle, algo, mode, &alpha,
        inputDesc, input, &beta, outputDesc, output);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
}
template <typename T>
void softmaxCudnn(void const *input, void *output, int *shape, int ndim)
{
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    softmaxCudnnDevice<T>(handle, input, output, shape, ndim);
    cudnnDestroy(handle);
}
extern "C" void softmax_cudnn(void const *input, void *output, int *shape, int ndim, int byteSize)
{
    if (byteSize == 4)
    {
        softmaxCudnn<float>(input, output, shape, ndim);
    }
    else if (byteSize == 2)
    {
        softmaxCudnn<uint16_t>(input, output, shape, ndim);
    }
}
