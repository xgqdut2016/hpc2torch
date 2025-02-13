#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

<<<<<<< HEAD
void setMatrixTensorEx(cnnlTensorDescriptor_t desc, cnnlTensorLayout_t layout, cnnlDataType_t dataType, int *shape, int ndim){
    int batch;
    int rows;
    int cols;
    int batch_stride;
    int row_stride;
    int col_stride;
    std::vector<int> stride(ndim, 1);
    for(int i = ndim - 2; i >= 0; i--){
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    if(ndim == 2){
        batch = 1;
        batch_stride = 0;
        rows = shape[0];
        cols = shape[1];
        row_stride = stride[0];
        col_stride = stride[1];
    }
    else if(ndim == 3){
        batch = shape[0];
        batch_stride = (batch == 1 ? 0 :stride[0]);
        rows = shape[1];
        cols = shape[2];
        row_stride = stride[1];
        col_stride = stride[2];
    }
    if (ndim == 3) {
        std::vector<int> dim_size = {batch, rows, cols};
        std::vector<int> dim_stride = {batch_stride, row_stride, col_stride};
        cnnlSetTensorDescriptorEx(desc, layout, dataType,
                                  dim_size.size(), dim_size.data(), dim_stride.data());
    } else if (ndim == 2) {
        std::vector<int> dim_size = {rows, cols};
        std::vector<int> dim_stride = {row_stride, col_stride};
        cnnlSetTensorDescriptorEx(desc, layout, dataType,
                                  dim_size.size(), dim_size.data(), dim_stride.data());
    }

}
=======
>>>>>>> faeb826 (error cnnl matmul)
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
<<<<<<< HEAD
    cnnlCreateTensorDescriptor(&bDesc);
    cnnlCreateTensorDescriptor(&cDesc);

    setMatrixTensorEx(aDesc, layout, dataType, a_shape, aDim);
    setMatrixTensorEx(bDesc, layout, dataType, b_shape, bDim);
    setMatrixTensorEx(cDesc, layout, dataType, c_shape, cDim);
    

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

=======
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
    cnnlMatMulDescriptor_t bmm_desc;
    cnnlMatMulDescCreate(&bmm_desc);
    cnnlSetMatMulDescAttr(bmm_desc, CNNL_MATMUL_DESC_TRANSA, &transA,
                          sizeof(int32_t));
    cnnlSetMatMulDescAttr(bmm_desc, CNNL_MATMUL_DESC_TRANSB, &transB,
                          sizeof(int32_t));

    cnnlMatMulAlgo_t bmm_algo;
    cnnlMatMulAlgoCreate(&bmm_algo);

    int count = 0;

    cnnlMatMulHeuristicResult_t desc;
    cnnlCreateMatMulHeuristicResult(&desc);

    cnnlGetBatchMatMulAlgoHeuristic(handle, bmm_desc, aDesc,
                                    bDesc, cDesc, NULL, 1, &desc, &count);
    size_t wsSize;
    cnnlGetBatchMatMulHeuristicResult(desc, bmm_algo, &wsSize);
    void *wsData;
    CNRT_CHECK(cnrtMalloc((void **)&wsData, wsSize));

    cnnlStatus_t stat = cnnlBatchMatMulBCast_v2(
        handle, bmm_desc, bmm_algo, &alpha, aDesc, aData,
        bDesc, bData, &beta, cDesc, cData, wsData, wsSize);
>>>>>>> faeb826 (error cnnl matmul)
    CNRT_CHECK(cnrtQueueSync(queue));
    if (stat != CNNL_STATUS_SUCCESS)
        return;
    cnrtFree(wsData);
    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(bDesc);
    cnnlDestroyTensorDescriptor(cDesc);

<<<<<<< HEAD
    cnnlMatMulDescDestroy(opDesc);
    cnnlMatMulAlgoDestroy(algo);
    cnnlDestroyMatMulHeuristicResult(algoResult);
=======
    cnnlMatMulDescDestroy(bmm_desc);
    cnnlMatMulAlgoDestroy(bmm_algo);
    cnnlDestroyMatMulHeuristicResult(desc);
>>>>>>> faeb826 (error cnnl matmul)
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
