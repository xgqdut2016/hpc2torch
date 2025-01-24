#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template<typename T>
void gatherCnnlDevice(void const *input, const int *indices, void *output, 
                        int *x_shape, int *w_shape, int *y_shape, 
                        int x_ndim, int w_ndim, int y_ndim, int axis, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    cnnlTensorDescriptor_t input_desc, indices_desc, output_desc;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&indices_desc);
    cnnlCreateTensorDescriptor(&output_desc);
    
    
    cnnlDataType_t dataType;
    if(sizeof(T) == 2){
        dataType = CNNL_DTYPE_HALF;
    }
    else if(sizeof(T) == 4){
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
    cnnlGatherV2(handle, axis, input_desc, input,
                         x_shape, indices_desc,
                         indices, output_desc, output);
    
    CNRT_CHECK(cnrtQueueSync(queue));
    
    cnnlDestroyTensorDescriptor(input_desc);
    cnnlDestroyTensorDescriptor(output_desc);
    cnnlDestroyTensorDescriptor(indices_desc);
    
}
template<typename T>
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
    auto index = reinterpret_cast<const int *>(indices);
    gatherCnnlDevice<T>(input, index, output, 
                    x_shape, w_shape, y_shape, 
                    x_ndim, w_ndim, y_ndim, axis, handle, queue);
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    
}
extern "C" void gather_cnnl_f32(void const *input, void const *indices, void *output, 
                                int *x_shape, int *w_shape, int *y_shape, 
                                int x_ndim, int w_ndim, int y_ndim, int axis){
    gatherCnnl<float>(input, indices, output, 
                    x_shape, w_shape, y_shape, 
                    x_ndim, w_ndim, y_ndim, axis);
}
extern "C" void gather_cnnl_f16(void const *input, void const *indices, void *output, 
                                int *x_shape, int *w_shape, int *y_shape, 
                                int x_ndim, int w_ndim, int y_ndim, int axis){
    gatherCnnl<uint16_t>(input, indices, output, 
                    x_shape, w_shape, y_shape, 
                    x_ndim, w_ndim, y_ndim, axis);
}






