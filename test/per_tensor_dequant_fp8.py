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
FP8_E4M3_MAX = 448.0


def dequantize_per_tensor_sym(x_packed, x_scale, dtype):
    fake_qweight = x_packed.to(dtype)
    dq_weight = fake_qweight * x_scale
    return dq_weight
    
def test(
    x_shape,
    symmetric,
    device
    ):
    if symmetric == False:
        return
    byteSize = 4
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    atol = 1e-3; rtol = 5e-2
    if test_dtype == torch.float16 or test_dtype == torch.bfloat16:
        atol = 1e-3; rtol = 5e-2
    elif test_dtype == torch.float32:
        atol = 3e-5; rtol = 5e-3
    print(
        f"Testing Per Tensor Dequant Fp8 on {device} with x_shape:{x_shape}, symmetric:{symmetric}, dtype:{test_dtype}"
    )
    
    x = torch.zeros(x_shape, device=device, dtype=test_dtype, requires_grad=False)
    num_elements = x_shape[0] * x_shape[1]
    x_scale = torch.rand((1, ), device=device, dtype=torch.float32, requires_grad=False)
    x_packed = torch.rand(x_shape, device=device, dtype=torch.float32, requires_grad=False).to(dtype=torch.float8_e4m3fn)

    ans = dequantize_per_tensor_sym(x_packed, x_scale, test_dtype)
    
    x_packed_ptr = ctypes.cast(x_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_scale_ptr = ctypes.cast(x_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if symmetric == True:
        x_zero_ptr = None

    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    
    if device == "cuda":
        torch_quant_linear_time = performance.CudaProfile((dequantize_per_tensor_sym, (x_packed, x_scale, test_dtype))) 
        lib.PerTensorDequantF8_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int
        ]
        
        custom_quant_linear_time = \
        performance.CudaProfile((lib.PerTensorDequantF8_nv, (
                      x_ptr, x_packed_ptr, x_scale_ptr, x_zero_ptr, 
                      num_elements, byteSize)))
        
    performance.logBenchmark(torch_quant_linear_time, custom_quant_linear_time)
    # 将结果转换回 PyTorch 张量以进行比较
    
    assert torch.allclose(x, ans, atol=atol, rtol=rtol)
    
    tmpa = x.float().detach().to('cpu').numpy().flatten()
    tmpb = ans.float().to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test layernorm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
    # x_shape, symmetric
    ((8, 8), True),
    ((8, 128), True),
    ((256, 1024), True),
    ((1024, 2048), True),
    ((2048, 2048), True),
    ((4096, 2048), True),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for x_shape, symmetric in test_cases:
    test(x_shape, symmetric, args.device)
