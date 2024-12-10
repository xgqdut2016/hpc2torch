#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template<typename T>
void RoPECnnlDevice(void *destination, void const *pos_ids, void const *sin_table, void const *cos_table, int *shape, int *strides, int total_seq_len, cnnlHandle_t &handle, cnrtQueue_t &queue){
    int seq_len = shape[0];
    int nhead = shape[1];
    int dim = shape[2];

    cnnlRotaryEmbeddingDescriptor_t ropeDesc;

    cnnlCreateRotaryEmbeddingDescriptor(&ropeDesc);
    cnnlSetRotaryEmbeddingDescriptor_v2(ropeDesc, false, true,
                                        false, false, CNNL_SEQDATA_TNBC);

    cnnlTensorDescriptor_t inDesc, posDesc, sinFullDesc, sinSelectedDesc, sinSelectedFP16Desc;
    cnnlCreateTensorDescriptor(&inDesc);
    cnnlCreateTensorDescriptor(&posDesc);
    cnnlCreateTensorDescriptor(&sinFullDesc);
    cnnlCreateTensorDescriptor(&sinSelectedDesc);
    cnnlCreateTensorDescriptor(&sinSelectedFP16Desc);

    int inShape[4] = {seq_len, 1, nhead, dim};
    int inStrides[4] = {strides[0], strides[0], strides[1], strides[2]};
        
    int sinFullShape[2] = {total_seq_len, dim};
    int sinSelectedShape[2] = {seq_len, dim};

    cnnlDataType_t dataType;
    if(sizeof(T) == 2){
        dataType = CNNL_DTYPE_HALF;
    }
    else if(sizeof(T) == 4){
        dataType = CNNL_DTYPE_FLOAT;
    }
    cnnlSetTensorDescriptorEx(inDesc, CNNL_LAYOUT_ARRAY, dataType, 4, inShape, inStrides);
    cnnlSetTensorDescriptor(posDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT64, 1, &seq_len);
    cnnlSetTensorDescriptor(sinFullDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, sinFullShape);
    cnnlSetTensorDescriptor(sinSelectedDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 2, sinSelectedShape);
    cnnlSetTensorDescriptor(sinSelectedFP16Desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 2, sinSelectedShape);

    size_t size = cnnlGetTensorElementNum(sinSelectedDesc) * sizeof(float);
    void *sinSelectedTable;
    cnrtMalloc(&sinSelectedTable, size);
    void *cosSelectedTable;
    cnrtMalloc(&cosSelectedTable, size);

    cnnlBatchGatherV2_v2(handle, 0, 0, 1,
                        sinFullDesc, sin_table,
                        posDesc, pos_ids,
                        sinSelectedDesc, sinSelectedTable);
    cnnlCastDataType(handle, sinSelectedDesc, sinSelectedTable,
                    CNNL_CAST_FLOAT_TO_HALF, 
                    sinSelectedFP16Desc, sinSelectedTable);
    cnnlBatchGatherV2_v2(handle, 0, 0, 1,
                        sinFullDesc, cos_table,
                        posDesc, pos_ids,
                        sinSelectedDesc, cosSelectedTable);
    cnnlCastDataType(handle, sinSelectedDesc, cosSelectedTable,
                    CNNL_CAST_FLOAT_TO_HALF,
                    sinSelectedFP16Desc, cosSelectedTable);

    // Do RoPE
    cnnlRotaryEmbedding_v2(handle, ropeDesc,
                        inDesc, destination, nullptr, nullptr,
                        sinSelectedFP16Desc, cosSelectedTable,
                        sinSelectedFP16Desc, sinSelectedTable,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                        inDesc, destination, nullptr, nullptr);
    CNRT_CHECK(cnrtQueueSync(queue));

    cnrtFree(sinSelectedTable);
    cnrtFree(cosSelectedTable);
    cnnlDestroyRotaryEmbeddingDescriptor(ropeDesc);
    cnnlDestroyTensorDescriptor(inDesc);
    cnnlDestroyTensorDescriptor(posDesc);
    cnnlDestroyTensorDescriptor(sinFullDesc);
    cnnlDestroyTensorDescriptor(sinSelectedDesc);
}
template<typename T>
void RoPECnnl(void *destination, void const *pos_ids, void const *sin_table, void const *cos_table, int *shape, int *strides, int total_seq_len)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    RoPECnnlDevice<T>(destination, pos_ids, sin_table, cos_table, shape, strides, total_seq_len, handle, queue);
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    
}
extern "C" void RoPE_cnnl_f16(void *destination, void const *pos_ids, void const *sin_table, void const *cos_table, int *shape, int *strides, int total_seq_len){
    RoPECnnl<uint16_t>(destination, pos_ids, sin_table, cos_table, shape, strides, total_seq_len);
}
