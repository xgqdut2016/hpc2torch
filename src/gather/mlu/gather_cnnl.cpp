#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template <typename T, typename Tind>
void gatherCnnlDevice(void const *input, void const *indices, void *output,
                      int *x_shape, int *w_shape, int *y_shape,
                      int x_ndim, int w_ndim, int y_ndim, int axis, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    int *shape_data;
    int shape_size = 1;
    for (int i = 0; i < x_ndim; i++)
    {
        shape_size *= x_shape[i];
    }
    CNRT_CHECK(cnrtMalloc((void **)&shape_data, shape_size * sizeof(int)));
    CNRT_CHECK(cnrtMemcpy(shape_data, x_shape, shape_size * sizeof(int), cnrtMemcpyHostToDev));

    cnnlTensorDescriptor_t input_desc, indices_desc, output_desc;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&indices_desc);
    cnnlCreateTensorDescriptor(&output_desc);

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
        input_desc, CNNL_LAYOUT_ARRAY, dataType,
        x_ndim, x_shape);
    cnnlSetTensorDescriptor(
        output_desc, CNNL_LAYOUT_ARRAY, dataType,
        y_ndim, y_shape);
    cnnlSetTensorDescriptor(
        indices_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32,
        w_ndim, w_shape);
    if (sizeof(Tind) == 4)
    {
        cnnlGatherV2(handle, axis, input_desc, input,
                     shape_data, indices_desc,
                     reinterpret_cast<const int *>(indices), output_desc, output);

        CNRT_CHECK(cnrtQueueSync(queue));
    }
    else if (sizeof(Tind) == 8)
    { // indices类型为int64
        int32_t *indices32;
        int num = 1;
        for (int i = 0; i < w_ndim; i++)
        {
            num *= w_shape[i];
        }
        CNRT_CHECK(cnrtMalloc((void **)&indices32, num * sizeof(int32_t)));
        cnnlTensorDescriptor_t bDescInt64;
        cnnlCreateTensorDescriptor(&bDescInt64);
        cnnlSetTensorDescriptor(
            bDescInt64, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT64, w_ndim,
            w_shape);
        cnnlCastDataType(handle, bDescInt64,
                         indices, CNNL_CAST_INT64_TO_INT32,
                         indices_desc, indices32);
        CNRT_CHECK(cnrtQueueSync(queue));
        cnnlDestroyTensorDescriptor(bDescInt64);

        cnnlGatherV2(handle, axis, input_desc, input,
                     shape_data, indices_desc,
                     reinterpret_cast<const int *>(indices32), output_desc, output);

        CNRT_CHECK(cnrtQueueSync(queue));
        cnrtFree(indices32);
    }
    cnrtFree(shape_data);
    cnnlDestroyTensorDescriptor(input_desc);
    cnnlDestroyTensorDescriptor(output_desc);
    cnnlDestroyTensorDescriptor(indices_desc);
}
template <typename T, typename Tind>
void gatherCnnl(void const *input, void const *indices, void *output,
                int *x_shape, int *w_shape, int *y_shape,
                int x_ndim, int w_ndim, int y_ndim, int axis)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    gatherCnnlDevice<T, Tind>(input, indices, output,
                              x_shape, w_shape, y_shape,
                              x_ndim, w_ndim, y_ndim, axis, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void gather_cnnl_f32(void const *input, void const *indices, void *output,
                                int *x_shape, int *w_shape, int *y_shape,
                                int x_ndim, int w_ndim, int y_ndim, int axis, int byteSize)
{
    if (byteSize == 2)
    {
        gatherCnnl<uint16_t, int32_t>(input, indices, output,
                                      x_shape, w_shape, y_shape,
                                      x_ndim, w_ndim, y_ndim, axis);
    }
    else if (byteSize == 4)
    {
        gatherCnnl<float, int32_t>(input, indices, output,
                                   x_shape, w_shape, y_shape,
                                   x_ndim, w_ndim, y_ndim, axis);
    }
}
