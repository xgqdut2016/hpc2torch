import torch
import ctypes
import torch.nn.functional as F
import argparse
import numpy as np
from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os



lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if bias is not None:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1) + bias
    else:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1)
    return o.to(out_dtype)
    
def test(a_shape, b_shape, c_shape, device):
    dataType = 0
    if (dataType == 0):
        test_dtype = torch.float16
    elif (dataType == 1):
        test_dtype = torch.bfloat16
    
    print(
        f"Testing int8 scaled_gemm on {device} with a_shape:{a_shape} b_shape:{b_shape} c_shape:{c_shape} , test_dtype:{test_dtype}"
    )
    M, K = a_shape
    N = b_shape[1]
    A = to_int8(torch.randn(a_shape, device=device) * 5)
    B = to_int8(torch.randn((N, K), device="cuda").t() * 5) #cutlass使用epilogue针对B是列优先计算的，必须使用这个才能保证正确性
    C = torch.randn(c_shape, device=device, dtype=test_dtype)
    x_scale = torch.randn((M,), device=device, dtype=torch.float32)
    weights_scale = torch.randn((N,), device=device, dtype=torch.float32)
    bias = torch.randn((N,), device=device, dtype=test_dtype) * 10

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    x_scale_ptr = ctypes.cast(x_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    weights_scale_ptr = ctypes.cast(weights_scale.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    if device == "cuda":
        torch_matmul_time = performance.CudaProfile((torch_scaled_mm, (A, B, x_scale, weights_scale, test_dtype, bias))) 
        lib.int8_scaled_gemm_cutlass.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]           
        custom_matmul_time = \
        performance.CudaProfile((lib.int8_scaled_gemm_cutlass, 
        (C_ptr, bias_ptr, A_ptr, x_scale_ptr, B_ptr, weights_scale_ptr, M, K, N, dataType)))
    performance.logBenchmark(torch_matmul_time, custom_matmul_time)

    ans = torch_scaled_mm(A, B, x_scale, weights_scale, test_dtype, bias)
    tmpa = ans.to('cpu').detach().numpy().flatten()
    tmpb = C.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

    
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test int8 scaled_gemm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()   


test_cases = [
    # a_shape, b_shape, c_shape
    ((128, 512), (512, 1024), (128, 1024)),
    ((256, 1024), (1024, 2048), (256, 2048)),
    ((1024, 2048), (2048, 1024), (1024, 1024)),
    
]
if args.device == "mlu":
    import torch_mlu
elif args.device == "npu":
    import torch_npu
for a_shape, b_shape, c_shape in test_cases:
    test(a_shape, b_shape, c_shape, args.device)
