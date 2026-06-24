#include "cnnl.h"
#include <vector>
#include <numeric>
struct ResizeMode
{
    enum Mode
    {
        // Arithmetic operations:
        Nearest,
        Bilinear,

        Count, ///< Number of resize operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined resize operations.
    static const size_t numResizeMode = Count;
};
struct CoordinateMode
{
    enum Mode
    {
        halfPixel,
        pytorchHalfPixel,
        alignCorners,
        asymmetric,
        tfCropAndResize,

        Count, ///< Number of resize operation types (marker for counting purposes).
    };

    // This static constant holds the total number of defined resize operations.
    static const size_t numCoordinateMode = Count;
};
template <typename T>
void resizeCnnlDevice(void const *input, float const *roi, void *output, 
                           int *x_shape, int *y_shape,
                           int ndim,
                           ResizeMode::Mode mode, CoordinateMode::Mode coMode,
                           cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    std::vector<int> permuteI(ndim);//从nchw做转置到nhwc
    std::vector<int> permuteO(ndim);//从nhwc转置回nchw
    for (int i = 0; i < ndim; i++) {
        permuteI[i] = i;
        permuteO[i] = i;
    }
    for (int i = 0; i < ndim; i++) {
        if(i >= 1){
            permuteI[i] = i + 1;
        }
        if(i >= 2){
            permuteO[i] = i - 1;
        }
    }
    permuteI[ndim - 1] = 1;
    permuteO[1] = ndim - 1;

    std::vector<int> inDim(ndim);//原始input的形状为[n,c,h,w]
    std::vector<int> outDim(ndim);
    int x_size = 1;//表示input的size
    int y_size = 1;//表示output的size
    for (int i = 0; i < ndim; i++) {
        inDim[i] = x_shape[i];
        outDim[i] = y_shape[i];
        x_size *= x_shape[i];
        y_size *= y_shape[i];
        
    }
    std::vector<int> x_tranDim(ndim);//tmpGdramI的形状
    std::vector<int> y_tranDim(ndim);//tmpGdramO的形状
    for(int i = 0; i < ndim; i++){
        x_tranDim[i] = x_shape[permuteI[i]];
        y_tranDim[i] = y_shape[permuteI[i]];
    }
    cnnlTensorLayout_t layoutI = CNNL_LAYOUT_NCHW;//只支持ndim=4
    cnnlTensorLayout_t layoutO = CNNL_LAYOUT_NHWC;

    cnnlDataType_t dataType;
    if (sizeof(T) == 2)
    {
        dataType = CNNL_DTYPE_HALF;
    }
    else if (sizeof(T) == 4)
    {
        dataType = CNNL_DTYPE_FLOAT;
    }
    T *tmpGdramI, *tmpGdramO;
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramI, x_size * sizeof(T)));
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramO, y_size * sizeof(T)));

    cnnlTensorDescriptor_t x_desc, y_desc, IDesc, ODesc;
    cnnlCreateTensorDescriptor(&x_desc);
    cnnlCreateTensorDescriptor(&y_desc);
    cnnlCreateTensorDescriptor(&IDesc);
    cnnlCreateTensorDescriptor(&ODesc);
    
    cnnlSetTensorDescriptor(
        x_desc, layoutI, dataType,
        inDim.size(), inDim.data());//原始input,nchw
    cnnlSetTensorDescriptor(
        IDesc, layoutO, dataType,
        x_tranDim.size(), x_tranDim.data());//转置以后的input,nhwc
    cnnlSetTensorDescriptor(
        y_desc, layoutI, dataType,
        outDim.size(), outDim.data());
    cnnlSetTensorDescriptor(
        ODesc, layoutO, dataType,
        y_tranDim.size(), y_tranDim.data());

    cnnlTransposeDescriptor_t desc;
    cnnlCreateTransposeDescriptor(&desc);
    cnnlSetTransposeDescriptor(desc, ndim, permuteI.data());
    //然后针对input做转置nchw2nhwc
    size_t tSizeI;
    cnnlGetTransposeWorkspaceSize(handle, x_desc, desc, &tSizeI);
    void *workspaceI;
    cnrtMalloc(&workspaceI, tSizeI);
    
    cnnlTranspose_v2(handle, desc, x_desc, input, IDesc,
                            tmpGdramI, workspaceI, tSizeI);
    CNRT_CHECK(cnrtQueueSync(queue));
    //下面开始做resize
    cnnlTensorDescriptor_t boxesDesc, boxesIndexDesc;
    cnnlCreateTensorDescriptor(&boxesDesc);
    auto nBatch = x_shape[0];
    std::vector<int> boxesDim = {nBatch, 4};
    cnnlSetTensorDescriptor(
        boxesDesc, CNNL_LAYOUT_ARRAY, dataType,
        boxesDim.size(), boxesDim.data());

    cnnlCreateTensorDescriptor(&boxesIndexDesc);
    std::vector<int> boxesIndexDim = {nBatch};
    cnnlSetTensorDescriptor(
        boxesIndexDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32,
        boxesIndexDim.size(), boxesIndexDim.data());
    std::vector<int32_t> boxesIndex(nBatch);
    std::iota(boxesIndex.begin(), boxesIndex.end(), 0);//boxesIndex=[0,1,2,...]
    int32_t *boxesIndexData;
    cnrtMalloc((void**)&boxesIndexData, nBatch * sizeof(int32_t));
    cnrtMemcpy(boxesIndexData, boxesIndex.data(), nBatch * sizeof(int32_t), cnrtMemcpyHostToDev);
     
    cnnlCropAndResizeMode_t resizeOp;
    if (mode == ResizeMode::Nearest){
        resizeOp = CNNL_CROP_AND_RESIZE_NEAREST;
    }
    else if (mode == ResizeMode::Bilinear){
        resizeOp = CNNL_CROP_AND_RESIZE_BILINEAR;
    }
    
    std::vector<float> box = {0, 0, 1.0, 1.0};
    if (coMode == CoordinateMode::tfCropAndResize){
        box = {roi[2], roi[3], roi[6], roi[7]};
    }
    float *boxesData;
    cnrtMalloc((void**)&boxesData, nBatch * box.size() * sizeof(float));
    for(int i = 0; i < nBatch; i++){
        cnrtMemcpy(boxesData + i * box.size(),
                    box.data(), box.size() * sizeof(float), cnrtMemcpyHostToDev);
    }
    cnnlCropAndResize(
            handle, IDesc, tmpGdramI, boxesDesc, boxesData,
            boxesIndexDesc, boxesIndexData, resizeOp, 0.0, ODesc, tmpGdramO);
    //------------------------------------------------------------ 
    //下面开始提前对output做转置：nhwc2nchw
    size_t tSizeO;
    cnnlGetTransposeWorkspaceSize(handle, ODesc, desc, &tSizeO);
    void *workspaceO;
    cnrtMalloc(&workspaceO, tSizeO);
    cnnlSetTransposeDescriptor(desc, ndim, permuteO.data());
    cnnlTranspose_v2(handle, desc, ODesc, tmpGdramO, y_desc,
                            output, workspaceO, tSizeO);
    CNRT_CHECK(cnrtQueueSync(queue));  
    
    cnrtFree(tmpGdramI);
    cnrtFree(tmpGdramO);

    cnrtFree(boxesIndexData);
    cnrtFree(boxesData);

    cnrtFree(workspaceI);
    cnrtFree(workspaceO);

    cnnlDestroyTensorDescriptor(IDesc);
    cnnlDestroyTensorDescriptor(ODesc);
    cnnlDestroyTransposeDescriptor(desc);

    cnnlDestroyTensorDescriptor(x_desc);
    cnnlDestroyTensorDescriptor(y_desc);
    cnnlDestroyTensorDescriptor(boxesDesc);
    cnnlDestroyTensorDescriptor(boxesIndexDesc);
    
}
template <typename T>
void resizeCnnl(void const *input, float const *roi, void *output, 
                           int *x_shape, int *y_shape,
                           int ndim,
                           ResizeMode::Mode mode, CoordinateMode::Mode coMode)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。
    
    resizeCnnlDevice<T>(input, roi, output, 
                           x_shape, y_shape,
                           ndim,
                           mode, coMode, handle, queue);
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));
}
extern "C" void nearest_cnnl(void const *input, float const *roi, void *output, 
                           int *x_shape, int *y_shape,
                           int ndim, int byteSize)
{
    if (byteSize == 2)
    {
        resizeCnnl<uint16_t>(input, roi, output, 
                           x_shape, y_shape,
                           ndim,
                           ResizeMode::Nearest, CoordinateMode::tfCropAndResize);
    }
    else if (byteSize == 4)
    {
        resizeCnnl<float>(input, roi, output, 
                           x_shape, y_shape,
                           ndim,
                           ResizeMode::Nearest, CoordinateMode::tfCropAndResize);
    }
}
extern "C" void bilinear_cnnl(void const *input, float const *roi, void *output, 
                           int *x_shape, int *y_shape,
                           int ndim, int byteSize)
{
    if (byteSize == 2)
    {
        resizeCnnl<uint16_t>(input, roi, output, 
                           x_shape, y_shape,
                           ndim,
                           ResizeMode::Bilinear, CoordinateMode::tfCropAndResize);
    }
    else if (byteSize == 4)
    {
        resizeCnnl<float>(input, roi, output, 
                           x_shape, y_shape,
                           ndim,
                           ResizeMode::Bilinear, CoordinateMode::tfCropAndResize);
    }
}
