import torch
import ctypes
import torch.nn.functional as F
from functools import partial
import argparse
from typing import List, Tuple
import math

from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def convTran(x, w, stride, padding, dilation):
    if(len(x.shape) == 3): 
        return F.conv_transpose1d(
            x, w, stride=stride, padding=padding, groups = 1, dilation=dilation
        )
    elif (len(x.shape) == 4): 
        return F.conv_transpose2d(
                x, w, stride=stride, padding=padding, groups = 1, dilation=dilation
            )
    elif (len(x.shape) == 5): 
        return F.conv_transpose3d(
                x, w, stride=stride, padding=padding, groups = 1, dilation=dilation
            )
    else:
        print("Error: Pytorch -> Unsupported tensor dimension")
        return None


def inferShape(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[int, ...]:
    # 检查维度一致性
    assert (
        len(x_shape) == len(w_shape) == len(pads) + 2 == len(dilations) + 2 == len(strides) + 2
    ), "x and w should have the same length; pads, strides, and dilations should have the same length; the length of pads should be that of x - 2"
    
    # 初始化输出维度列表
    output_dims = []
    
    for i in range(len(pads)):
        # 有效卷积核大小
        effective_kernel_size = (w_shape[i + 2] - 1) * dilations[i] + 1
        # 输出尺寸计算
        output_dim = strides[i] * (x_shape[i + 2] - 1) + effective_kernel_size - pads[i] * 2
        output_dims.append(output_dim)
    # 返回输出形状
    return (x_shape[0], w_shape[1]) + tuple(output_dims)


def test(x_shape, w_shape, pads, strides, dilations, device):
    byteSize = 2
    if byteSize == 2:
        test_dtype = torch.float16
    elif byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing convTranspose on {device} with x_shape:{x_shape}, w_shape:{w_shape}, pads: {pads}, strides: {strides}, dilations: {dilations}, dtype:{test_dtype}"
    )           
    ndim = len(x_shape) 

    x = torch.rand(x_shape, dtype=test_dtype).to(device)
    w = torch.rand(w_shape, dtype=test_dtype).to(device)
    y_shape = inferShape(x.shape, w.shape, pads, strides, dilations)
    y = torch.zeros(y_shape, dtype=test_dtype).to(device)
    tmpa = convTran(x, w, strides, pads, dilations)
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    w_ptr = ctypes.cast(w.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    import numpy as np
    x_array = np.array(x_shape, dtype=np.int32)
    xShape = x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    w_array = np.array(w_shape, dtype=np.int32)
    wShape = w_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    y_array = np.array(y_shape, dtype=np.int32)
    yShape = y_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    p_array = np.array(pads, dtype=np.int32)
    pData = p_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    s_array = np.array(strides, dtype=np.int32)
    sData = s_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    d_array = np.array(dilations, dtype=np.int32)
    dData = d_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    outpads = [0 for i in range(len(pads))]
    outpads_array = np.array(outpads, dtype=np.int32)
    oData = outpads_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_convTranspose_time = performance.BangProfile((convTran, (x, w, strides, pads, dilations))) 
        lib.convTranspose_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),#pads
            ctypes.POINTER(ctypes.c_int),#strides
            ctypes.POINTER(ctypes.c_int),#dilations
            ctypes.POINTER(ctypes.c_int),#x_shape
            ctypes.POINTER(ctypes.c_int),#w_shape
            ctypes.POINTER(ctypes.c_int),#y_shape
            ctypes.c_int,
            ctypes.c_int
        ]           
        custom_convTranspose_time = \
        performance.BangProfile((lib.convTranspose_cnnl, (x_ptr, w_ptr, y_ptr, pData, sData, dData, xShape, wShape, yShape, ndim, byteSize)))
    elif device == "npu":
        torch_convTranspose_time = performance.AscendProfile((convTran, (x, w, strides, pads, dilations))) 
        lib.convTranspose_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),#pads
            ctypes.POINTER(ctypes.c_int),#strides
            ctypes.POINTER(ctypes.c_int),#dilations
            ctypes.POINTER(ctypes.c_int),#outpads
            ctypes.POINTER(ctypes.c_int),#x_shape
            ctypes.POINTER(ctypes.c_int),#w_shape
            ctypes.POINTER(ctypes.c_int),#y_shape
            ctypes.c_int,
            ctypes.c_int
        ]           
        custom_convTranspose_time = \
        performance.AscendProfile((lib.convTranspose_aclnn, (x_ptr, w_ptr, y_ptr, pData, sData, dData, oData, xShape, wShape, yShape, ndim, byteSize)))
            
    performance.logBenchmark(torch_convTranspose_time, custom_convTranspose_time)
    

    # 将结果转换回 PyTorch 张量以进行比较
    
    tmpa = convTran(x, w, strides, pads, dilations).to('cpu').detach().numpy().flatten()
    
    tmpb = y.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test convTranspose on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [   
        ((20, 16, 50),
            (16, 33, 5),
            (1,),
            (1,),
            (1,)),    
        ((32, 64, 128, 128),
            (64, 3, 5, 5),
            (2, 2),
            (2, 2),
            (1, 1)), 
        ((1, 2, 4, 4, 4), #当ndim=5时，昇腾机器不支持f32计算，应该是昇腾机器的torch.convtranspose不支持，torch.conv3d有同样问题
            (2, 1, 5, 5, 5),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1)), 
        ((32, 3, 32, 32, 32),
            (3, 3, 5, 5, 5),
            (3, 2, 2),
            (4, 3, 3),
            (2, 2, 1)),   
]


if args.device == 'mlu':
    import torch_mlu
elif args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for x_shape, w_shape, pads, strides, dilations in test_cases:
    test(x_shape, w_shape, pads, strides, dilations, args.device)

