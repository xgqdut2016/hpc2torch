#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>
template <typename T>
void clipCnnlDevice(void const *aData, void *cData,
                    int *aShape, int aDim,
                    float minValue, float maxValue,
                    cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlTensorDescriptor_t aDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        aDim, aShape);
    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        aDim, aShape);
    cnnlStatus_t stat =
        cnnlClip_v2(handle, CNNL_POINTER_MODE_HOST, aDesc, aData, &minValue, &maxValue, cDesc, cData);
    // cnnlStatus_t stat =
    //     cnnlClip(handle, aDesc, aData, &minValue, &maxValue, cData);
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    CNRT_CHECK(cnrtQueueSync(queue));
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(cDesc);
}
template <typename T>
void clipCnnl(void const *aData, void *cData,
              int *aShape, int aDim,
              float minValue, float maxValue)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。
    clipCnnlDevice<T>(aData, cData,
                      aShape, aDim,
                      minValue, maxValue, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void clip_cnnl(void const *aData, void *cData,
                          int *aShape, int aDim,
                          float minValue, float maxValue, int byteSize)
{
    if (byteSize == 4)
    {
        clipCnnl<float>(aData, cData,
                        aShape, aDim,
                        minValue, maxValue);
    }
    else if (byteSize == 2)
    {
        clipCnnl<uint16_t>(aData, cData,
                           aShape, aDim,
                           minValue, maxValue);
    }
}
