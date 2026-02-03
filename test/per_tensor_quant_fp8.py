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


def per_tensor_quant_fp8_torch(x, symmetric):
    if symmetric == False:
        return
    else:
        absmax = x.flatten().abs().max()

        if absmax == 0:
            scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
            return q, scale, None

        # 2. scale = absmax / FP8_MAX
        scale = absmax / FP8_E4M3_MAX

        # 3. 量化（注意：CUDA 里用的是 x * (1 / scale)）
        inv_scale = 1.0 / scale
        x_scaled = x * inv_scale

        # 4. clip 到 FP8 可表示范围
        x_clamped = torch.clamp(x_scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)

        # 5. cast to fp8 e4m3
        q = x_clamped.to(torch.float8_e4m3fn)

        return q, scale.float(), None
    
def test(
    x_shape,
    symmetric,
    is_static,
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
        f"Testing Per Tensor Quant Fp8 on {device} with x_shape:{x_shape}, symmetric:{symmetric}, dtype:{test_dtype}"
    )
    
    x = torch.rand(x_shape, device=device, dtype=test_dtype, requires_grad=False)
    num_elements = x_shape[0] * x_shape[1]
    x_scale = torch.zeros((1, ), device=device, dtype=torch.float32, requires_grad=False)
    x_packed = torch.rand(x_shape, device=device, dtype=torch.float32, requires_grad=False).to(dtype=torch.float8_e4m3fn)

    x_p, x_s, x_z = per_tensor_quant_fp8_torch(x, symmetric)
    
    x_packed_ptr = ctypes.cast(x_packed.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_scale_ptr = ctypes.cast(x_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    if symmetric == True:
        x_zero_ptr = None

    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    
    if device == "cuda":
        torch_quant_linear_time = performance.CudaProfile((per_tensor_quant_fp8_torch, (x, symmetric))) 
        lib.PerTensorQuantF8_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int
        ]
        
        custom_quant_linear_time = \
        performance.CudaProfile((lib.PerTensorQuantF8_nv, (
                      x_packed_ptr, x_scale_ptr, x_zero_ptr, x_ptr,
                      num_elements, is_static, byteSize)))
        
    performance.logBenchmark(torch_quant_linear_time, custom_quant_linear_time)
    # 将结果转换回 PyTorch 张量以进行比较
    
    assert torch.allclose(x_p.float(), x_packed.float(), atol=2, rtol=2) 
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
    # x_shape, symmetric, is_static
    ((8, 8), True, False),
    ((8, 128), True, False),
    ((256, 1024), True, False),
    ((1024, 2048), True, False),
    ((2048, 2048), True, False),
    ((4096, 2048), True, False),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for x_shape, symmetric, is_static in test_cases:
    test(x_shape, symmetric, is_static, args.device)
