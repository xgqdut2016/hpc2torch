#include "cnnl.h"
#include "cnrt.h"
#include <vector>

template <typename T>
void softmaxCnnlDevice(T const *source, T *destination, int nDim, int axis, int *shape, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlSoftmaxMode_t mode;
    std::vector<int> inDim = {1, 1, 1};
    std::vector<int> outDim = inDim;

    if (nDim >= 3)
    {
        if (axis == 0)
        {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];
            for (int i = 2; i < nDim; ++i)
            {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        }
        else if (axis == nDim - 1)
        {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[0] = shape[0];
            for (int i = 1; i < axis; ++i)
            {
                inDim[1] *= shape[i];
            }
            inDim[2] = shape[axis];
            outDim = inDim;
        }
        else
        {
            mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
            for (int i = 0; i < axis; ++i)
            {
                inDim[0] *= shape[i];
            }
            inDim[1] = shape[axis];
            for (int i = axis + 1; i < nDim; ++i)
            {
                inDim[2] *= shape[i];
            }
            outDim = inDim;
        }
    }
    else if (nDim == 2)
    {
        if (axis == 0)
        {
            mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
            inDim[0] = shape[0];
            inDim[1] = shape[1];

            outDim = inDim;
        }
        else
        {
            mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
            inDim[1] = shape[0];
            inDim[2] = shape[1];

            outDim = inDim;
        }
    }
    else
    {
        mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
        inDim[0] = shape[0];

        outDim = inDim;
    }
    cnnlTensorDescriptor_t aDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&cDesc);
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
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        inDim.size(), inDim.data());
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        outDim.size(), outDim.data());

    T alpha = 1.0;
    T beta = 0.0;
    cnnlStatus_t stat =
        cnnlSoftmaxForward_v2(handle, CNNL_SOFTMAX_ACCURATE,
                              mode, CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                              &alpha, aDesc, source, &beta, cDesc, destination);

    CNRT_CHECK(cnrtQueueSync(queue));

    if (stat != CNNL_STATUS_SUCCESS)
        return;
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(cDesc);
}
template <typename T>
void softmaxCnnl(void const *input, void *output, int nDim, int axis, int *shape)
{
    auto source = reinterpret_cast<const T *>(input);
    auto destination = reinterpret_cast<T *>(output);
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    softmaxCnnlDevice(source, destination, nDim, axis, shape, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}

extern "C" void softmax_cnnl_f32(void const *input, void *output, int nDim, int axis, int *shape, int byteSize)
{

    if (byteSize == 4)
    {
        softmaxCnnl<float>(input, output, nDim, axis, shape);
    }
    else if (byteSize == 2)
    {
        softmaxCnnl<uint16_t>(input, output, nDim, axis, shape);
    }
}
