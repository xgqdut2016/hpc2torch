#include "acl/acl.h"
// #include "aclnnop/level2/aclnn_convolution.h"
#include "aclnnop/aclnn_convolution.h"
#include <iostream>
#include <vector>
#include "npu/common_npu.h"

template <typename T>
void convolutionAclnnDevice(void *input, void *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, aclrtStream &stream)
{
    aclDataType dataType;
    if (sizeof(T) == 2)
    {
        dataType = aclDataType::ACL_FLOAT16;
    }
    else if (sizeof(T) == 4)
    {
        dataType = aclDataType::ACL_FLOAT;
    }
    aclFormat format;
    if (nDim == 3)
    {
        format = aclFormat::ACL_FORMAT_NCL;
    }
    else if (nDim == 4)
    {
        format = aclFormat::ACL_FORMAT_NCHW;
    }
    else if (nDim == 5)
    {
        format = aclFormat::ACL_FORMAT_NCDHW;
    }
    std::vector<int64_t> inputDim(nDim);       // aclCreateTensor只支持int64_t的数组
    std::vector<int64_t> inputStride(nDim, 1); // 初始化为1
    std::vector<int64_t> weightDim(nDim);
    std::vector<int64_t> weightStride(nDim, 1);
    std::vector<int64_t> outputDim(nDim);
    std::vector<int64_t> outputStride(nDim, 1);

    for (int i = nDim - 1; i >= 0; i--)
    {
        inputDim[i] = int64_t(x_shape[i]);
        outputDim[i] = int64_t(y_shape[i]);
        weightDim[i] = int64_t(w_shape[i]);
        if (i < nDim - 1)
        {
            inputStride[i] = inputDim[i + 1] * inputStride[i + 1];
            weightStride[i] = weightDim[i + 1] * weightStride[i + 1];
            outputStride[i] = outputDim[i + 1] * outputStride[i + 1];
        }
    }
    auto inputTensor =
        aclCreateTensor(inputDim.data(), inputDim.size(), dataType,
                        inputStride.data(), 0, format,
                        inputDim.data(), inputDim.size(), input); // const aclTensor *inputTensor
    auto weightTensor =
        aclCreateTensor(weightDim.data(), weightDim.size(), dataType,
                        weightStride.data(), 0, format,
                        weightDim.data(), weightDim.size(), scale);
    auto outputTensor =
        aclCreateTensor(outputDim.data(), outputDim.size(), dataType,
                        outputStride.data(), 0, format,
                        outputDim.data(), outputDim.size(), output);
    // 下面开始正式计算
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);只能接受int64_t的数组
    std::vector<int64_t> pad(nDim - 2);
    std::vector<int64_t> stride(nDim - 2);
    std::vector<int64_t> dilation(nDim - 2);
    std::vector<int64_t> outputPadding(nDim - 2, 0); // 这个数组元素默认全是0
    for (int i = 0; i < nDim - 2; i++)
    {
        pad[i] = int64_t(pads[i]);
        stride[i] = int64_t(strides[i]);
        dilation[i] = int64_t(dilations[i]);
    }

    aclIntArray *convpads = aclCreateIntArray(pad.data(), pad.size());
    aclIntArray *convstride =
        aclCreateIntArray(stride.data(), stride.size());
    aclIntArray *convdilation =
        aclCreateIntArray(dilation.data(), dilation.size());
    aclIntArray *convOutputpadding =
        aclCreateIntArray(outputPadding.data(), outputPadding.size());
    int groups = 1;
    int8_t cubeMathType = 1;
    auto ret = aclnnConvolutionGetWorkspaceSize(
        inputTensor, weightTensor, nullptr, convstride, convpads,
        convdilation, false, convOutputpadding, int64_t(groups), outputTensor,
        cubeMathType, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret);
    }
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

        if (ret != ACL_SUCCESS)
        {
            printf("allocate workspace failed. ERROR: %d\n", ret);
        }
    }

    ret = aclnnConvolution(workspaceAddr, workspaceSize, executor,
                           stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclnnConvolution failed. ERROR: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);

    if (ret != ACL_SUCCESS)
    {
        printf("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    aclDestroyTensor(inputTensor);
    aclDestroyTensor(weightTensor);
    aclDestroyTensor(outputTensor);

    aclDestroyIntArray(convstride);
    aclDestroyIntArray(convpads);
    aclDestroyIntArray(convOutputpadding);
    aclDestroyIntArray(convdilation);

    // aclDestroyAclOpExecutor(executor);//似乎不支持destroy，一旦destroy测试报错
}
template <typename T>
void convolutionAclnn(void *input, void *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim)
{
    // static int count = 0;
    // printf("count is %d \n", count);
    int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS)
    {
        printf("Init acl failed. ERROR: %d\n", ret);
    }

    convolutionAclnnDevice<T>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim, stream);
    Finalize(deviceId, stream);
}
extern "C" void convolution_aclnn_f32(void *input, void *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim)
{
    convolutionAclnn<float>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
}
extern "C" void convolution_aclnn_f16(void *input, void *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim)
{
    convolutionAclnn<uint16_t>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
}
