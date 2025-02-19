#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template <typename T>
void expandCnnlDevice(void const *input, void *output, int *inputShape, int *outputShape, int nDim, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlTensorDescriptor_t yDesc, xDesc;
    cnnlCreateTensorDescriptor(&yDesc);
    cnnlCreateTensorDescriptor(&xDesc);

    std::vector<int> inDim(nDim);
    std::vector<int> outDim(nDim);
    for (int i = 0; i < nDim; i++)
    {
        inDim[i] = inputShape[i];
        outDim[i] = outputShape[i];
    }
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlSetTensorDescriptor(
        xDesc, CNNL_LAYOUT_ARRAY, dataType,
        inDim.size(), inDim.data());
    cnnlSetTensorDescriptor(
        yDesc, CNNL_LAYOUT_ARRAY, dataType,
        outDim.size(), outDim.data());
    cnnlExpand(handle, xDesc, input, yDesc, output);
    CNRT_CHECK(cnrtQueueSync(queue));

    cnnlDestroyTensorDescriptor(xDesc);
    cnnlDestroyTensorDescriptor(yDesc);
}
template <typename T>
void expandCnnl(void const *input, void *output, int *inputShape, int *outputShape, int nDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    expandCnnlDevice<T>(input, output, inputShape, outputShape, nDim, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void expand_cnnl(void const *input, void *output, int *inputShape, int *outputShape, int nDim, int byteSize)
{
    if (byteSize == 4)
    {
        expandCnnl<float>(input, output, inputShape, outputShape, nDim);
    }
    else if (byteSize == 2)
    {
        expandCnnl<uint16_t>(input, output, inputShape, outputShape, nDim);
    }
}
