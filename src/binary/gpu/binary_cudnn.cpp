#include <cudnn.h>
#include <cstring>
#include <stdio.h>
#include <vector>
struct BinaryMode
{
    enum Mode
    {
        // Arithmetic operations:
        Add,
        Subtract,
        Multiply,
        Divide,
        Pow,
        Mod,
        Max,
        Min,
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,
        BitwiseNot,
        // Logical operations:
        // **TODO Not currently supported**
        // Requires Boolean data type
        And,
        Or,
        Xor,
        Less,
        LessOrEqual,
        Equal,
        Greater,
        GreaterOrEqual,

        Count, ///< Number of binary operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined binary operations.
    static const size_t numBinaryMode = Count;
};
template <typename T>
void elementWiseCudnnDevice(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                            int aDim, int bDim, int cDim,
                            BinaryMode::Mode mode, float aAlpha, float bAlpha, float beta,
                            cudnnHandle_t &handle)
{
    cudnnDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CUDNN_DATA_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CUDNN_DATA_FLOAT;
    }
    if (aDim > 4 || bDim > 4 || cDim > 4)
    {
        return;
    }
    int a[4] = {1, 1, 1, 1};
    int b[4] = {1, 1, 1, 1};
    int c[4] = {1, 1, 1, 1};
    std::memcpy(a + (4 - aDim), aShape, aDim * sizeof(int));
    std::memcpy(b + (4 - bDim), bShape, bDim * sizeof(int));
    std::memcpy(c + (4 - cDim), cShape, cDim * sizeof(int));

    cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
    cudnnCreateTensorDescriptor(&aDesc);
    cudnnSetTensor4dDescriptor(
        aDesc, CUDNN_TENSOR_NCHW, dataType, a[0], a[1], a[2], a[3]);

    cudnnCreateTensorDescriptor(&bDesc);
    cudnnSetTensor4dDescriptor(
        bDesc, CUDNN_TENSOR_NCHW, dataType, b[0], b[1], b[2], b[3]);

    cudnnCreateTensorDescriptor(&cDesc);
    cudnnSetTensor4dDescriptor(
        cDesc, CUDNN_TENSOR_NCHW, dataType, c[0], c[1], c[2], c[3]);

    cudnnOpTensorDescriptor_t opDesc;
    cudnnCreateOpTensorDescriptor(&opDesc);
    if (mode == BinaryMode::Add)
    {
        cudnnSetOpTensorDescriptor(
            opDesc, CUDNN_OP_TENSOR_ADD, dataType, CUDNN_NOT_PROPAGATE_NAN);
    }
    else if (mode == BinaryMode::Multiply)
    {
        cudnnSetOpTensorDescriptor(
            opDesc, CUDNN_OP_TENSOR_MUL, dataType, CUDNN_NOT_PROPAGATE_NAN);
    }
    else if (mode == BinaryMode::Max)
    {
        cudnnSetOpTensorDescriptor(
            opDesc, CUDNN_OP_TENSOR_MAX, dataType, CUDNN_NOT_PROPAGATE_NAN);
    }
    else if (mode == BinaryMode::Min)
    {
        cudnnSetOpTensorDescriptor(
            opDesc, CUDNN_OP_TENSOR_MIN, dataType, CUDNN_NOT_PROPAGATE_NAN);
    }
    cudnnOpTensor(handle, opDesc,
                  &aAlpha, aDesc, aData, &bAlpha, bDesc,
                  bData, &beta, cDesc, cData);
    cudnnDestroyTensorDescriptor(aDesc);
    cudnnDestroyTensorDescriptor(bDesc);
    cudnnDestroyTensorDescriptor(cDesc);
    cudnnDestroyOpTensorDescriptor(opDesc);
}

template <typename T>
void elementWiseCudnn(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                      int aDim, int bDim, int cDim,
                      BinaryMode::Mode mode, float aAlpha, float bAlpha, float beta)
{
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    elementWiseCudnnDevice<T>(aData, bData, cData, aShape, bShape, cShape,
                              aDim, bDim, cDim,
                              mode, aAlpha, bAlpha, beta, handle);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    // printf("kernel time:%.4f ms\n", ker_time / 1000.);
    cudnnDestroy(handle);
}
extern "C" void add_cudnn(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCudnn<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                   aDim, bDim, cDim,
                                   BinaryMode::Add, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCudnn<float>(aData, bData, cData, aShape, bShape, cShape,
                                aDim, bDim, cDim,
                                BinaryMode::Add, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void mul_cudnn(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCudnn<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                   aDim, bDim, cDim,
                                   BinaryMode::Multiply, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCudnn<float>(aData, bData, cData, aShape, bShape, cShape,
                                aDim, bDim, cDim,
                                BinaryMode::Multiply, 1.0f, 1.0f, 0.0f);
    }
}

extern "C" void max_cudnn(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCudnn<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                   aDim, bDim, cDim,
                                   BinaryMode::Max, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCudnn<float>(aData, bData, cData, aShape, bShape, cShape,
                                aDim, bDim, cDim,
                                BinaryMode::Max, 1.0f, 1.0f, 0.0f);
    }
}
extern "C" void min_cudnn(void const *aData, void const *bData, void *cData, int *aShape, int *bShape, int *cShape,
                          int aDim, int bDim, int cDim, int byteSize)
{
    if (byteSize == 2)
    {
        elementWiseCudnn<uint16_t>(aData, bData, cData, aShape, bShape, cShape,
                                   aDim, bDim, cDim,
                                   BinaryMode::Min, 1.0f, 1.0f, 0.0f);
    }
    else if (byteSize == 4)
    {
        elementWiseCudnn<float>(aData, bData, cData, aShape, bShape, cShape,
                                aDim, bDim, cDim,
                                BinaryMode::Min, 1.0f, 1.0f, 0.0f);
    }
}
