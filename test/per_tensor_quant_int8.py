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



def per_tensor_quant_int8_torch(x, symmetric):
    if symmetric == False:
        return
    else:
        x = x.float()
        absmax = x.flatten().abs().max()
        if absmax == 0:
            scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            q = torch.zeros_like(x, dtype=torch.int8)
            return q, scale, None
        scale_x = absmax / 127
        x_q = x.mul(127 / absmax)
        x_q = torch.round(x_q).to(torch.int8)

        return x_q, scale_x, None
    
def test(
    x_shape,
    symmetric,
    device
    ):
    
    byteSize = 4
    test_dtype = torch.float16
    if byteSize == 4:
        test_dtype = torch.float32
    elif byteSize == 2:
        test_dtype = torch.float16
    atol = 1e-3; rtol = 5e-2
    if test_dtype == torch.float16 or test_dtype == torch.bfloat16:
        atol = 1e-3; rtol = 5e-2
    elif test_dtype == torch.float32:
        atol = 3e-5; rtol = 5e-3
    print(
        f"Testing Per Tensor Quant Int8 on {device} with x_shape:{x_shape}, symmetric:{symmetric}, dtype:{test_dtype}"
    )
    x = torch.rand(x_shape, device=device, dtype=test_dtype, requires_grad=False)
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    num_elements = x_shape[0] * x_shape[1]

    x_packed = torch.zeros(x_shape, device=device, dtype=torch.int8, requires_grad=False)
    x_packed_ptr = ctypes.cast(x_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_scale = torch.rand((1, ), device=device, dtype=x.dtype, requires_grad=False)
    x_scale_ptr = ctypes.cast(x_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_zero = None
    x_zero_ptr = None
    if symmetric == False:
        x_zero = torch.rand((1, ), device=device, dtype=x.dtype, requires_grad=False)
        x_zero_ptr = ctypes.cast(x_zero.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    x_p, x_s, x_z = per_tensor_quant_int8_torch(x, symmetric)
    if device == "cuda":
        torch_quant_linear_time = performance.CudaProfile((per_tensor_quant_int8_torch, (x, symmetric))) 
        lib.PerTensorQuantI8_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int
        ]
        
        custom_quant_linear_time = \
        performance.CudaProfile((lib.PerTensorQuantI8_nv, (x_packed_ptr, x_scale_ptr, x_zero_ptr, x_ptr, num_elements, byteSize)))
        
    performance.logBenchmark(torch_quant_linear_time, custom_quant_linear_time)
    # 将结果转换回 PyTorch 张量以进行比较
    
    assert torch.allclose(x_p.float(), x_packed.float(), atol=1, rtol=0) 
    assert torch.allclose(x_s, x_scale, atol=atol, rtol=rtol)
    tmpa = x_p.float().detach().to('cpu').numpy().flatten()
    tmpb = x_packed.float().to('cpu').detach().numpy().flatten()
    
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
    ((128, 512), True),
    ((128, 128), True),
    ((256, 1024), True),
    ((256, 2048), True),
    ((1024, 2048), True),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for x_shape, symmetric in test_cases:
    test(x_shape, symmetric, args.device)
