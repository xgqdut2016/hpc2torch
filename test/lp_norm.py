import torch
import ctypes
import numpy as np
import torch.nn as nn
from functools import partial
import argparse

from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def lp_norm(x, axis, p, eps):
    return torch.nn.functional.normalize(
        x.to(torch.float32), dim=axis, p=p, eps=eps
    ).to(x.dtype)

def test(test_shape, axis, p, eps, device):
    byteSize = 2
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    print(
        f"Testing LPNorm on {device} with test_shape:{test_shape}, axis:{axis} p:{p} , dtype:{test_dtype}, eps:{eps}"
    )
    ndim = len(test_shape)
    
    input = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)
    output = torch.rand(test_shape, device=device, dtype=test_dtype, requires_grad=False)
    input_stride = np.array(input.stride(), dtype=np.int32)
    output_stride = np.array(output.stride(), dtype=np.int32)
    
    input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    

    
    np_array = np.array(test_shape, dtype=np.int32)
    ctypes_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    input_array = input_stride.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    output_array = output_stride.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    if device == "cuda":
        torch_lp_norm_time = performance.CudaProfile((lp_norm, (input, axis, p, eps)))  # 以毫秒为单位
        lib.lp_norm_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_int
        ]
        custom_lp_norm_time = \
        performance.CudaProfile((lib.lp_norm_nv, (output_ptr, input_ptr, output_array, input_array, ctypes_array, ndim, axis, p, eps, True, byteSize)))
    
    performance.logBenchmark(torch_lp_norm_time, custom_lp_norm_time)

    # 将结果转换回 PyTorch 张量以进行比较
    tmpa = lp_norm(input, axis, p, eps).to('cpu').detach().numpy().flatten()
    
    tmpb = output.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test lp_norm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
        # shape, axis, p, eps
    ((2, 1, 512), -1, 2, 1e-12),
    ((2, 1, 1024), -1, 2, 1e-12),
    ((2, 1, 2048), -1, 2, 1e-12),
    ((2048, 2050), 0, 1, 1e-12),
    ((2048, 2050), 1, 1, 1e-12),
    ((12, 16, 512, 512), 0, 2, 1e-12),
    ((12, 16, 512, 512), 1, 2, 1e-12),
    ((12, 16, 512, 512), 2, 1, 1e-12),
    ((12, 16, 512, 512), 3, 2, 1e-12),
    ((1, 16, 512, 512), 0, 2, 1e-12),
    ((1, 16, 512, 512), 1, 1, 1e-12),
    ((1, 16, 512, 512), 2, 2, 1e-12),
    ((1, 16, 512, 512), 3, 2, 1e-12),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for shape, axis, p, eps in test_cases:
    test(shape, axis, p, eps, args.device)