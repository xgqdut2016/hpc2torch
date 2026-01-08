import torch
import ctypes
import numpy as np
from functools import partial
import argparse

from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)
def padFunction(input, pad, value):#暂时不支持axis参数，寒武纪目前只支持constant
    return torch.nn.functional.pad(input, pad, 'constant', value)#torch.pad只支持value是标量
def inferShape(input, pad):
    ndim = input.ndimension()
    outputShape = []
    
    for i in range(ndim):
        tmp = input.shape[i] + pad[2 * (ndim - 1 - i)] + pad[2 * (ndim - 1 - i) + 1]
        outputShape.append(tmp)
    return outputShape
def test(shape, Pad, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing pad on {device} with shape:{shape}, pad:{Pad}, dtype:{test_dtype}"
    )
    
    ndim = len(shape)
    input = torch.rand(shape, device=device, dtype=test_dtype, requires_grad=False)
    #特别注意,hostPad,hostValue不要放到device上
    hostPad = torch.tensor(Pad, dtype=torch.int32, requires_grad=False)
    hostValue = torch.rand(1, dtype=test_dtype, requires_grad=False)
    
    
    yShape = inferShape(input, Pad)
    output = torch.zeros(yShape, device=device, dtype=test_dtype, requires_grad=False)

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    hostPad_ptr = ctypes.cast(hostPad.data_ptr(), ctypes.POINTER(ctypes.c_int))
    hostValue_ptr = ctypes.cast(hostValue.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    x_shape = np.array(shape, dtype=np.int32)
    inputShape = x_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_shape = np.array(yShape, dtype=np.int32)
    outputShape = y_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_pad_time = performance.BangProfile((padFunction, (input, Pad, hostValue[0])))  # 以毫秒为单位
        lib.pad_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_pad_time = \
        performance.BangProfile((lib.pad_cnnl, (input_ptr, hostPad_ptr, hostValue_ptr, output_ptr, 
        inputShape, outputShape, 
        ndim, byteSize)))
    elif device == "npu":
        torch_pad_time = performance.AscendProfile((padFunction, (input, Pad, hostValue[0])))  # 以毫秒为单位
        lib.pad_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_pad_time = \
        performance.AscendProfile((lib.pad_aclnn, (input_ptr, hostPad_ptr, hostValue_ptr, output_ptr, 
        inputShape, outputShape, 
        ndim, byteSize)))
    
    performance.logBenchmark(torch_pad_time, custom_pad_time)
    # 将结果转换回 PyTorch 张量以进行比较
    
    tmpa = padFunction(input, Pad, hostValue[0]).to('cpu').numpy().flatten()
    tmpb = output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test pad on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # shape, Pad
        ((700, 10), (1, 0, -1, 3)),
        ((700, 10, 24), (1, 0, 0, -2, 1, 2)),
        ((700, 10, 24, 6), (1, 0, 0, 3, -2, -2, -3, 1)),     
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for shape, Pad in test_cases:
    test(shape, Pad, args.device)
