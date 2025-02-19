import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)
def expandFunction(x, yShape):
    return x.expand(yShape)

def test(xShape, yShape, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing expand on {device} with xShape:{xShape}, yShape:{yShape}, dtype:{test_dtype}"
    )
    
    ndim = len(xShape)
    input = torch.rand(xShape, device=device, dtype=test_dtype, requires_grad=False)
    output = torch.zeros(yShape, device=device, dtype=test_dtype, requires_grad=False)

    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    x_shape = np.array(xShape, dtype=np.int32)
    inputShape = x_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_shape = np.array(yShape, dtype=np.int32)
    outputShape = y_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if device == "mlu":
        torch_expand_time = performance.BangProfile((expandFunction, (input, yShape)))  # 以毫秒为单位
        lib.expand_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_expand_time = \
        performance.BangProfile((lib.expand_cnnl, (input_ptr, output_ptr, inputShape, outputShape, ndim, byteSize)))
    if device == "npu":
        torch_expand_time = performance.AscendProfile((expandFunction, (input, yShape)))  # 以毫秒为单位
        lib.expand_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int
        ]
        custom_expand_time = \
        performance.AscendProfile((lib.expand_aclnn, (input_ptr, output_ptr, inputShape, outputShape, ndim, byteSize)))
    performance.logBenchmark(torch_expand_time, custom_expand_time)
    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = expandFunction(input, yShape).to('cpu').numpy().flatten()
    tmpb = output.to('cpu').numpy().flatten()

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test expand on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # x_shape, y_shape, dtype
        ((700, 1, 24), (700, 1200, 24)),
        ((1, 1200, 24), (700, 1200, 24)),
        ((700, 1200, 1), (700, 1200, 24)),       
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for xShape, yShape in test_cases:
    test(xShape, yShape, args.device)
