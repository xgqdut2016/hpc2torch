#include "cnnl.h"
#include <vector>

template<typename T>
void convolutionCnnlDevice(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, cnnlHandle_t &handle, cnrtQueue_t &queue){
    //nDim = len(w_shape) = len(x_shape) = len(y_shape)
    std::vector<int> inDim(nDim);//原始input的形状为[n,c,h,w]
    std::vector<int> wDim(nDim);
    std::vector<int> outDim(nDim);
    int x_size = 1;//表示input的size
    int w_size = 1;//表示scale的size
    int y_size = 1;//表示output的size
    for (int i = 0; i < nDim; i++) {
        inDim[i] = x_shape[i];
        outDim[i] = y_shape[i];
        wDim[i] = w_shape[i];
        x_size *= x_shape[i];
        w_size *= w_shape[i];
        y_size *= y_shape[i];
        
    }
    
    cnnlTensorDescriptor_t x_desc, w_desc, y_desc;
    cnnlCreateTensorDescriptor(&x_desc);
    cnnlCreateTensorDescriptor(&w_desc);
    cnnlCreateTensorDescriptor(&y_desc);
    cnnlTensorLayout_t layout;//cnnlConv只支持nDim=4,5
    
    if(nDim == 4){
        layout = CNNL_LAYOUT_NHWC;
    }
    else if(nDim == 5){
        layout = CNNL_LAYOUT_NDHWC;
    }
    cnnlDataType_t dataType;
    if(sizeof(T) == 2){
        dataType = CNNL_DTYPE_HALF;
    }
    else if(sizeof(T) == 4){
        dataType = CNNL_DTYPE_FLOAT;
    }
    //由于cnnl支持的操作是nhwc，所以下面需要提前对数据做permute，下面开始nchw2nhwc
    T *tmpGdramI, *tmpGdramS, *tmpGdramO;//conv库函数只能处理[n,h,w,c],tmpGdramI作为转置来变换input
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramI, x_size * sizeof(T)));
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramS, w_size * sizeof(T)));
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramO, y_size * sizeof(T)));
    cnnlTransposeDescriptor_t desc;
    cnnlCreateTransposeDescriptor(&desc);
    
    std::vector<int> permuteI(nDim);
    std::vector<int> permuteO(nDim);
    for (int i = 0; i < nDim; i++) {
        permuteI[i] = i;
        permuteO[i] = i;
    }
    for (int i = 0; i < nDim; i++) {
        if(i >= 1){
            permuteI[i] = i + 1;
        }
        if(i >= 2){
            permuteO[i] = i - 1;
        }
    }
    permuteI[nDim - 1] = 1;
    permuteO[1] = nDim - 1;
    
    cnnlSetTransposeDescriptor(desc, nDim, permuteI.data());

    std::vector<int> x_tranDim(nDim);//tmpGdramI的形状
    std::vector<int> w_tranDim(nDim);//tmpGdramS的形状
    std::vector<int> y_tranDim(nDim);//tmpGdramO的形状
    for(int i = 0; i < nDim; i++){
        x_tranDim[i] = x_shape[permuteI[i]];
        w_tranDim[i] = w_shape[permuteI[i]];
        y_tranDim[i] = y_shape[permuteI[i]];
    }
    //下面先对input做转置nchw2nhwc
    cnnlTensorDescriptor_t aDesc, cDesc;
    
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&cDesc);
    //下面需要针对input和scale的不同shape不断修改aDesc,cDesc的数据
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        inDim.size(), inDim.data());
    
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        x_tranDim.size(), x_tranDim.data());

    
    size_t tSizeI;
    cnnlGetTransposeWorkspaceSize(handle, aDesc, desc, &tSizeI);
    void *workspaceI;
    cnrtMalloc(&workspaceI, tSizeI);
    
    cnnlTranspose_v2(handle, desc, aDesc, input, cDesc,
                            tmpGdramI, workspaceI, tSizeI);
    CNRT_CHECK(cnrtQueueSync(queue));  
    //然后针对scale做转置nchw2nhwc
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        wDim.size(), wDim.data());
    
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        w_tranDim.size(), w_tranDim.data());

    
    size_t tSizeS;
    cnnlGetTransposeWorkspaceSize(handle, aDesc, desc, &tSizeS);
    void *workspaceS;
    cnrtMalloc(&workspaceS, tSizeS);
    
    cnnlTranspose_v2(handle, desc, aDesc, scale, cDesc,
                            tmpGdramS, workspaceS, tSizeS);
    CNRT_CHECK(cnrtQueueSync(queue));  
    //------------------------------------------------------------               
    //上面成功对input, scale做好了nchw2nhwc，下面开始正式计算conv
    cnnlSetTensorDescriptor(
        x_desc, layout, dataType,
        x_tranDim.size(), x_tranDim.data());
    cnnlSetTensorDescriptor(
        w_desc, layout, dataType,
        w_tranDim.size(), w_tranDim.data());
    cnnlSetTensorDescriptor(
        y_desc, layout, dataType,
        y_tranDim.size(), y_tranDim.data());
    // for(int i = 0; i < nDim; i++){
    //     printf("%d ", y_tranDim[i]);
    // }
    // printf("\n");
    
    cnnlConvolutionDescriptor_t convDesc;
    cnnlCreateConvolutionDescriptor(&convDesc);
    cnnlSetConvolutionDescriptor(convDesc, nDim, pads, strides, dilations, 1,
                                         dataType);
    cnnlConvolutionForwardAlgo_t algo;   
    cnnlGetConvolutionForwardAlgorithm(handle, convDesc,
                                           x_desc, w_desc, y_desc,
                                           CNNL_CONVOLUTION_FWD_FASTEST, &algo);                                  
    size_t convSize;                                     
    cnnlGetConvolutionForwardWorkspaceSize(handle,
                                       x_desc,
                                       w_desc,
                                       y_desc,
                                       nullptr,
                                       convDesc,
                                       algo,
                                       &convSize);   
    void *workspaceConv;
    cnrtMalloc(&workspaceConv, convSize);  
    cnnlConvolutionForward(
            handle, convDesc, algo, NULL, x_desc, tmpGdramI, w_desc,
            tmpGdramS, NULL, NULL, workspaceConv, convSize, NULL, y_desc, tmpGdramO);                                                                 
    //------------------------------------------------------------ 
    //下面开始提前对output做转置：nhwc2nchw，此时需要重新设置aDesc和cDesc,desc
    cnnlSetTensorDescriptor(
        aDesc, CNNL_LAYOUT_ARRAY, dataType,
        y_tranDim.size(), y_tranDim.data());
    cnnlSetTensorDescriptor(
        cDesc, CNNL_LAYOUT_ARRAY, dataType,
        outDim.size(), outDim.data());
    size_t tSizeO;
    cnnlGetTransposeWorkspaceSize(handle, aDesc, desc, &tSizeO);
    void *workspaceO;
    cnrtMalloc(&workspaceO, tSizeO);
    cnnlSetTransposeDescriptor(desc, nDim, permuteO.data());
    cnnlTranspose_v2(handle, desc, aDesc, tmpGdramO, cDesc,
                            output, workspaceO, tSizeO);
    CNRT_CHECK(cnrtQueueSync(queue));  
    cnrtFree(tmpGdramI);
    cnrtFree(tmpGdramS);
    cnrtFree(tmpGdramO);

    cnrtFree(workspaceI);
    cnrtFree(workspaceConv);
    cnrtFree(workspaceS);
    cnrtFree(workspaceO);

    cnnlDestroyTensorDescriptor(aDesc);
    cnnlDestroyTensorDescriptor(cDesc);
    cnnlDestroyTransposeDescriptor(desc);

    cnnlDestroyTensorDescriptor(x_desc);
    cnnlDestroyTensorDescriptor(w_desc);
    cnnlDestroyTensorDescriptor(y_desc);
    cnnlDestroyConvolutionDescriptor(convDesc);
}
template<typename T>
void convolutionCnnl(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    convolutionCnnlDevice<T>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim, handle, queue);
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    
}
extern "C" void convolution_cnnl_f32(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim){
    convolutionCnnl<float>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
}
extern "C" void convolution_cnnl_f16(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim){
    convolutionCnnl<uint16_t>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
}
