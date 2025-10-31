#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>

static constexpr cudnnConvolutionFwdAlgo_t ALGOS[8] = {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

static constexpr cudnnConvolutionMode_t MODES[2] = {
    CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION};

template <typename T>
void convolutionCudnnDevice(cudnnHandle_t handle, void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim)
{
    // 默认是4D或者5D
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    // cudnnTensorDescriptor_t b_desc = nullptr;
    // cudnnActivationDescriptor_t act_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;

    cudnnConvolutionFwdAlgo_t algo = ALGOS[0];
    cudnnConvolutionMode_t mode = MODES[1];

    std::vector<int> input_dims(nDim);
    std::vector<int> output_dims(nDim);
    std::vector<int> filter_dims(nDim);
    std::vector<int> input_strides(nDim, 1);
    std::vector<int> output_strides(nDim, 1);

    cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;

    for (int i = nDim - 1; i >= 0; i--)
    {
        input_dims[i] = x_shape[i];
        output_dims[i] = y_shape[i];
        filter_dims[i] = w_shape[i];
        if (i < nDim - 1)
        {
            input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
            output_strides[i] = output_strides[i + 1] * output_dims[i + 1];
        }
    }
    cudnnDataType_t cudnn_data_type;
    if constexpr (std::is_same_v<T, uint16_t>)
    {
        cudnn_data_type = CUDNN_DATA_HALF;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        cudnn_data_type = CUDNN_DATA_FLOAT;
    }
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensorNdDescriptorEx(
        x_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
        nDim, input_dims.data());
    cudnnSetTensorNdDescriptorEx(
        y_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
        nDim, output_dims.data());
    cudnnSetFilterNdDescriptor(
        w_desc, cudnn_data_type, CUDNN_TENSOR_NCHW,
        nDim, filter_dims.data());
    int spatial_ndim = nDim - 2;
    cudnnSetConvolutionNdDescriptor(
        conv_desc,
        spatial_ndim,
        pads,
        strides,
        dilations,
        mode,
        compute_type);
    const float alpha = 1.0f, beta = 0.0f;
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc,
        algo, &workspace_size);
    void *workspace = nullptr;
    if (workspace_size > 0)
    {
        cudaMalloc(&workspace, workspace_size);
    }
    cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc,
        input,
        w_desc,
        scale,
        conv_desc,
        algo,
        workspace, workspace_size,
        &beta,
        y_desc,
        output);
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
}
template <typename T>
void convolutionCudnn(void const *input,
                      void const *scale,
                      void *output,
                      int *pads,
                      int *strides,
                      int *dilations,
                      int *x_shape,
                      int *w_shape,
                      int *y_shape,
                      int nDim)
{
    cudnnHandle_t handle; // cudnn句柄
    cudnnCreate(&handle);

    if (nDim == 3)
    {
        int new_ndim = 4;
        int *new_pads = (int *)malloc(2 * sizeof(int));
        int *new_strides = (int *)malloc(2 * sizeof(int));
        int *new_dilations = (int *)malloc(2 * sizeof(int));
        int *new_x_shape = (int *)malloc(new_ndim * sizeof(int));
        int *new_w_shape = (int *)malloc(new_ndim * sizeof(int));
        int *new_y_shape = (int *)malloc(new_ndim * sizeof(int));

        for (int i = 0; i < 2; i++)
        {
            new_pads[i] = (i < 1 ? pads[i] : 0);
            new_strides[i] = (i < 1 ? strides[i] : 1);
            new_dilations[i] = (i < 1 ? dilations[i] : 1);
        }

        for (int i = 0; i < new_ndim; i++)
        {
            new_x_shape[i] = (i < nDim ? x_shape[i] : 1);
            new_w_shape[i] = (i < nDim ? w_shape[i] : 1);
            new_y_shape[i] = (i < nDim ? y_shape[i] : 1);
        }

        convolutionCudnnDevice<T>(
            handle, // ✅ 传入 cudnn 句柄
            input, scale, output,
            new_pads, new_strides, new_dilations,
            new_x_shape, new_w_shape, new_y_shape,
            new_ndim);

        free(new_pads);
        free(new_strides);
        free(new_dilations);
        free(new_x_shape);
        free(new_w_shape);
        free(new_y_shape);
    }
    else
    {
        convolutionCudnnDevice<T>(
            handle, // ✅ 传入 cudnn 句柄
            input, scale, output,
            pads, strides, dilations,
            x_shape, w_shape, y_shape,
            nDim);
    }

    cudnnDestroy(handle);
}

extern "C" void convolution_cudnn(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, int byteSize)
{
    if (byteSize == 4)
    {
        convolutionCudnn<float>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
    }
    else if (byteSize == 2)
    {
        convolutionCudnn<uint16_t>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
    }
}
