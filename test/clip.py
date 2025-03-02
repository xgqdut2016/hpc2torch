import torch
import ctypes
import numpy as np
import torch.nn.functional as F
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def test(test_shape, minValue, maxValue, device):
    ndim = len(test_shape)
    byteSize = 2
    if byteSize == 2:
        tensor_dtype = torch.float16
    elif byteSize == 4:
        tensor_dtype = torch.float32
    print(
        f"Testing Clip on {device} with x_shape:{test_shape}, min:{minValue}, max:{maxValue}, dtype:{tensor_dtype}"
    )
    
    aData = torch.rand(test_shape, device=device, dtype=tensor_dtype, requires_grad=False)
    cData = torch.rand(test_shape, device=device, dtype=tensor_dtype, requires_grad=False)

    n = 1
    for i in range(len(test_shape)):
        n *= test_shape[i]

    input_ptr = ctypes.cast(aData.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(cData.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    aShape = np.array(test_shape, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    if device == "cpu":
        torch_clip_time = performance.CpuProfile((torch.clip, (aData, minValue, maxValue)))  # 可以替换为mul, div
        lib.clip_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int

        ]
        custom_clip_time = \
        performance.CpuProfile((lib.clip_cpu, (input_ptr, output_ptr, minValue, maxValue, n, byteSize)))
    elif device == "cuda":
        torch_clip_time = performance.CudaProfile((torch.clip, (aData, minValue, maxValue)))  # 可以替换为mul, div
        lib.clip_cuda.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int

        ]
        custom_clip_time = \
        performance.CudaProfile((lib.clip_cuda, (input_ptr, output_ptr, minValue, maxValue, n, byteSize)))
    elif device == "mlu":
        torch_clip_time = performance.BangProfile((torch.clip, (aData, minValue, maxValue)))  # 可以替换为mul, div
        lib.clip_cnnl.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int

        ]
        custom_clip_time = \
        performance.BangProfile((lib.clip_cnnl, (input_ptr, output_ptr, aShape, len(test_shape), minValue, maxValue, byteSize)))
    elif device == "npu":
        torch_clip_time = performance.AscendProfile((torch.clip, (aData, minValue, maxValue)))  # 可以替换为mul, div
        lib.clip_aclnn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int

        ]
        custom_clip_time = \
        performance.AscendProfile((lib.clip_aclnn, (input_ptr, output_ptr, aShape, len(test_shape), minValue, maxValue, byteSize)))
    performance.logBenchmark(torch_clip_time, custom_clip_time)
    tmpa = torch.clip(aData, minValue, maxValue).to('cpu').numpy().flatten()
    tmpb = cData.to('cpu').numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test clip on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # 
        ((6, ), 0.5, 1),
        ((33, 109), 0.1, 0.5),
        # ((502, 768, 5), -0.5, 0.5),
        ((50257, 768, 5), 0.5, 1),
         
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for test_shape, minValue, maxValue in test_cases:
    test(test_shape, minValue, maxValue, args.device)