#include <xpu/refactor/context/xpu_act_type.h>
#include "kunlun/common_kunlun.h"
#include <vector>

template <typename T>
void convolutionXdnnDevice(void const *x, void const *w, void *y, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, xdnn::Context *handle)
{
    switch (nDim)
    {
    case 3:
    {
        int64_t ksize = (int64_t)w_shape[2];
        int64_t stride = (int64_t)strides[0];
        std::initializer_list<int64_t> pad = {(int64_t)pads[0]};
        int64_t dilation = (int64_t)dilations[0];
        if (sizeof(T) == 2)
        {
            xdnn::conv1d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)x_shape[0], (int64_t)x_shape[1], (int64_t)x_shape[2],
                                                                    (int64_t)w_shape[0], ksize,
                                                                    stride, pad,
                                                                    dilation, 1, nullptr,
                                                                    nullptr, nullptr, true, nullptr,
                                                                    nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                    nullptr);
        }
        else if (sizeof(T) == 4)
        {
            xdnn::conv1d_fusion<float, float, float, int16_t>(handle, (float *)x, (float *)w, (float *)y, (int64_t)x_shape[0], (int64_t)x_shape[1], (int64_t)x_shape[2],
                                                              (int64_t)w_shape[0], ksize,
                                                              stride, pad,
                                                              dilation, 1, nullptr,
                                                              nullptr, nullptr, true, nullptr,
                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                              nullptr);
        }
    }
    case 4:
    {
        std::vector<int64_t> ksize = {(int64_t)w_shape[2], (int64_t)w_shape[3]};
        std::vector<int64_t> stride = {(int64_t)strides[0], (int64_t)strides[1]};
        std::vector<int64_t> pad = {
            (int64_t)pads[0],
            (int64_t)pads[1]};
        std::vector<int64_t> dilation = {(int64_t)dilations[0], (int64_t)dilations[1]};
        if (sizeof(T) == 2)
        {
            xdnn::conv2d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)x_shape[0], (int64_t)x_shape[1], (int64_t)x_shape[2],
                                                                    (int64_t)x_shape[3], (int64_t)w_shape[0], ksize,
                                                                    stride, pad,
                                                                    dilation, 1, nullptr,
                                                                    nullptr, nullptr, true, nullptr,
                                                                    nullptr, baidu::xpu::api::Activation_t::LINEAR, nullptr,
                                                                    nullptr, -1);
        }
        else if (sizeof(T) == 4)
        {
            xdnn::conv2d_fusion<float, float, float, int16_t>(handle, (float *)x, (float *)w, (float *)y, (int64_t)x_shape[0], (int64_t)x_shape[1], (int64_t)x_shape[2],
                                                              (int64_t)x_shape[3], (int64_t)w_shape[0], ksize,
                                                              stride, pad,
                                                              dilation, 1, nullptr,
                                                              nullptr, nullptr, true, nullptr,
                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR, nullptr,
                                                              nullptr, -1);
        }
    }
    case 5:
    {
        std::vector<int64_t> ksize = {(int64_t)w_shape[2], (int64_t)w_shape[3], (int64_t)w_shape[4]};
        std::vector<int64_t> stride = {(int64_t)strides[0], (int64_t)strides[1], (int64_t)strides[2]};
        std::vector<int64_t> pad = {(int64_t)pads[0],
                                    (int64_t)pads[1],
                                    (int64_t)pads[2]};
        std::vector<int64_t> dilation = {(int64_t)dilations[0], (int64_t)dilations[1], (int64_t)dilations[2]};
        if (sizeof(T) == 2)
        {
            xdnn::conv3d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)x_shape[0], (int64_t)x_shape[1], (int64_t)x_shape[2],
                                                                    (int64_t)x_shape[3], (int64_t)x_shape[4], (int64_t)w_shape[0], ksize,
                                                                    stride, pad,
                                                                    dilation, 1, nullptr,
                                                                    nullptr, nullptr, true, nullptr,
                                                                    nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                    nullptr);
        }
        else if (sizeof(T) == 4)
        {
            xdnn::conv3d_fusion<float, float, float, int16_t>(handle, (float *)x, (float *)w, (float *)y, (int64_t)x_shape[0], (int64_t)x_shape[1], (int64_t)x_shape[2],
                                                              (int64_t)x_shape[3], (int64_t)x_shape[4], (int64_t)w_shape[0], ksize,
                                                              stride, pad,
                                                              dilation, 1, nullptr,
                                                              nullptr, nullptr, true, nullptr,
                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                              nullptr);
        }
    }
    default:
        break;
    }
}
template <typename T>
void convolutionXdnn(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim)
{
    auto handle = xdnn::create_context();
    convolutionXdnnDevice<T>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim, handle);
    // xpu_wait(handle->xpu_stream);
    destroy_context(handle);
}
extern "C" void convolution_xdnn(void const *input, void const *scale, void *output, int *pads, int *strides, int *dilations, int *x_shape, int *w_shape, int *y_shape, int nDim, int byteSize)
{
    if (byteSize == 4)
    {
        convolutionXdnn<float>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
    }
    else if (byteSize == 2)
    {
        convolutionXdnn<uint16_t>(input, scale, output, pads, strides, dilations, x_shape, w_shape, y_shape, nDim);
    }
}
