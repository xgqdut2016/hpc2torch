#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template <typename T>
void matmulCnnlDevice(void const *aData, void const *bData, void *cData,
                      int *a_shape, int *b_shape, int *c_shape,
                      int aDim, int bDim, int cDim,
                      float alpha, float beta,
                      cnnlHandle_t &handle, cnrtQueue_t &queue)
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
    int32_t transA = 0; // 默认不转置
    int32_t transB = 0;
    cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlSetTensorDescriptor(
        aDesc, layout, dataType,
        aDim, a_shape);

    cnnlCreateTensorDescriptor(&bDesc);
    cnnlSetTensorDescriptor(
        bDesc, layout, dataType,
        bDim, b_shape);

    cnnlCreateTensorDescriptor(&cDesc);
    cnnlSetTensorDescriptor(
        cDesc, layout, dataType,
        cDim, c_shape);

    cnnlMatMulDescriptor_t opDesc;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
    cnnlMatMulDescCreate(&opDesc);
    cnnlMatMulAlgoCreate(&algo);
    cnnlCreateMatMulHeuristicResult(&algoResult);
    int32_t use_stride = true;
    cnnlSetMatMulDescAttr(opDesc, CNNL_MATMUL_USE_STRIDE, &use_stride,
                          sizeof(int32_t));

    int count = 0;

    cnnlGetBatchMatMulAlgoHeuristic(handle, opDesc, aDesc,
                                    bDesc, cDesc,
                                    NULL, 1, &algoResult, &count);
    size_t wsSize;
    cnnlGetBatchMatMulHeuristicResult(algoResult, algo, &wsSize);
    void *wsData;
    cnrtMalloc(&wsData, wsSize);
    cnnlStatus_t stat = cnnlBatchMatMulBCast_v2(handle, opDesc, algo,
                                                &alpha, aDesc, aData,
                                                bDesc, bData,
                                                &beta, cDesc, cData,
                                                wsData, wsSize);

    CNRT_CHECK(cnrtQueueSync(queue));
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    cnrtFree(wsData);
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(bDesc);
    cnnlDestroyTensorDescriptor(cDesc);

    cnnlMatMulDescDestroy(opDesc);
    cnnlMatMulAlgoDestroy(algo);
    cnnlDestroyMatMulHeuristicResult(algoResult);
}
template <typename T>
void matmulCnnl(void const *aData, void const *bData, void *cData,
                int *a_shape, int *b_shape, int *c_shape,
                int aDim, int bDim, int cDim,
                float alpha, float beta)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    matmulCnnlDevice<T>(aData, bData, cData,
                        a_shape, b_shape, c_shape,
                        aDim, bDim, cDim,
                        alpha, beta, handle, queue);

    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}

extern "C" void matmul_cnnl(void const *aData, void const *bData, void *cData,
                            int *a_shape, int *b_shape, int *c_shape,
                            int aDim, int bDim, int cDim,
                            float alpha, float beta, int byteSize)
{
    if (byteSize == 2)
    {
        matmulCnnl<uint16_t>(aData, bData, cData,
                             a_shape, b_shape, c_shape,
                             aDim, bDim, cDim,
                             alpha, beta);
    }
    else if (byteSize == 4)
    {
        matmulCnnl<float>(aData, bData, cData,
                          a_shape, b_shape, c_shape,
                          aDim, bDim, cDim,
                          alpha, beta);
    }
}
