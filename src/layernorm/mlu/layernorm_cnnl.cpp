#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template <typename T>
void layernormCnnlDevice(void const *source, void const *weight, void const *bias, void *destination, int *shape, int nDim, int axis, float eps, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlTensorDescriptor_t yDesc, xDesc, filter_bias_desc, mean_rstd_desc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);
    cnnlCreateTensorDescriptor(&filter_bias_desc);
    cnnlCreateTensorDescriptor(&mean_rstd_desc);

    std::vector<int> inDim(nDim);
    std::vector<int> outDim(nDim);
    std::vector<int> filter_biasDim(nDim - axis);
    std::vector<int> mean_rstdDim(axis);
    int mean_rstd_size = 1;
    for (int i = 0; i < nDim; i++)
    {
        inDim[i] = shape[i];
        outDim[i] = shape[i];
        if (i >= axis)
        {
            filter_biasDim[i - axis] = shape[i];
        }
        else
        {
            mean_rstdDim[i] = shape[i];
            mean_rstd_size *= shape[i];
        }
    }

    size_t dtype_size = 0;
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlGetSizeOfDataType(dataType, &dtype_size);
    cnnlSetTensorDescriptor(
        xDesc, CNNL_LAYOUT_ARRAY, dataType,
        inDim.size(), inDim.data());
    cnnlSetTensorDescriptor(
        yDesc, CNNL_LAYOUT_ARRAY, dataType,
        outDim.size(), outDim.data());
    cnnlSetTensorDescriptor(
        filter_bias_desc, CNNL_LAYOUT_ARRAY, dataType,
        filter_biasDim.size(), filter_biasDim.data());
    cnnlSetTensorDescriptor(
        mean_rstd_desc, CNNL_LAYOUT_ARRAY, dataType,
        mean_rstdDim.size(), mean_rstdDim.data());

    T *mean_dev, *rstd_dev;
    size_t size_mean_rstd = (size_t)mean_rstd_size * dtype_size;

    CNRT_CHECK(cnrtMalloc((void **)&mean_dev, size_mean_rstd));
    CNRT_CHECK(cnrtMalloc((void **)&rstd_dev, size_mean_rstd));
    size_t wsSize;
    cnnlGetLayerNormOpWorkspaceSize(handle, axis, xDesc, &wsSize);

    void *workspace;
    cnrtMalloc(&workspace, wsSize);
    cnnlLayerNormForward(handle,
                         xDesc,
                         source,
                         axis,
                         filter_bias_desc,
                         weight,
                         bias,
                         eps,
                         workspace,
                         wsSize,
                         yDesc,
                         destination,
                         mean_rstd_desc,
                         mean_dev,
                         rstd_dev);

    CNRT_CHECK(cnrtQueueSync(queue));

    cnrtFree(workspace);
    cnrtFree(mean_dev);
    cnrtFree(rstd_dev);

    cnnlDestroyTensorDescriptor(xDesc);
    cnnlDestroyTensorDescriptor(yDesc);
    cnnlDestroyTensorDescriptor(filter_bias_desc);
    cnnlDestroyTensorDescriptor(mean_rstd_desc);
}
template <typename T>
void layernormCnnl(void const *input, void const *scale, void const *bias, void *output, int *shape, int nDim, int axis, float eps)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    layernormCnnlDevice<T>(input, scale, bias, output, shape, nDim, axis, eps, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void layernorm_cnnl(void const *input, void const *scale, void const *bias, void *output,
                               int *shape, int nDim, int axis, float eps, int byteSize)
{
    if (byteSize == 2)
    {
        layernormCnnl<uint16_t>(input, scale, bias, output, shape, nDim, axis, eps);
    }
    else if (byteSize == 4)
    {
        layernormCnnl<float>(input, scale, bias, output, shape, nDim, axis, eps);
    }
}
