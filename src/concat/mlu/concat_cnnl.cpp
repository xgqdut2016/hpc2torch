#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template <typename T>
void concatCnnlDevice(void const **input, void *output, int **inputShape, int *outputShape, int num, int axis, int nDim, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlTensorDescriptor_t desc;
    cnnlCreateTensorDescriptor(&desc);
    cnnlSetTensorDescriptor(
        desc, layout, dataType,
        nDim, outputShape);

    cnnlTensorDescriptor_t descArray[num];
    for (int i = 0; i < num; ++i) {
        cnnlCreateTensorDescriptor(&descArray[i]);
        
        cnnlSetTensorDescriptor(descArray[i], layout,
                                dataType,
                                nDim,
                                inputShape[i]);
    }
    size_t wsSize;
    cnnlGetConcatWorkspaceSize(handle, num, &wsSize);
    void *wsData;
    cnrtMalloc(&wsData, wsSize);

    cnnlStatus_t stat =
        cnnlConcat(handle, num, axis, descArray, input,
                    wsData, wsSize, desc, output);
    if (stat != CNNL_STATUS_SUCCESS)
        return;

    for (int i = 0; i < num; ++i) {
        cnnlDestroyTensorDescriptor(descArray[i]);
    }
    cnnlDestroyTensorDescriptor(desc);
}
template <typename T>
void concatCnnl(void const **input, void *output, int **inputShape, int *outputShape, int num, int axis, int nDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    concatCnnlDevice<T>(input, output, inputShape, outputShape, num, axis, nDim, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void concat_cnnl(void const **input, void *output, int **inputShape, int *outputShape, int num, int axis, int nDim, int byteSize)
{
    if (byteSize == 4)
    {
        concatCnnl<float>(input, output, inputShape, outputShape, num, axis, nDim);
    }
    else if (byteSize == 2)
    {
        concatCnnl<uint16_t>(input, output, inputShape, outputShape, num, axis, nDim);
    }
}
