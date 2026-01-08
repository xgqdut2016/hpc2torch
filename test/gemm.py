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

def gemm(
    x_packed: torch.Tensor,
    w_packed: torch.Tensor,
    y_packed: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    out_dtype: torch.dtype | None = None,
):
    """
    Compute:
        Y = alpha * (X @ W) + beta * Y

    Supports int8 GEMM with int32 accumulation.

    Args:
        x_packed: [M, K]
        w_packed: [K, N]
        y_packed: [M, N]
        alpha, beta: float
        acc_dtype: accumulation dtype (default int32)
        out_dtype: output dtype (default same as y_packed)
    """

    assert x_packed.dim() == 2
    assert w_packed.dim() == 2
    assert y_packed.dim() == 2

    if out_dtype is None:
        out_dtype = y_packed.dtype

    # ---- 1. Cast inputs to accumulation type ----
    x_acc = x_packed.float()
    w_acc = w_packed.float()
   
    prod = torch.matmul(x_acc, w_acc)  # [M, N]

    # ---- 3. alpha scaling ----
    if alpha != 1.0:
        prod = prod * alpha

    # ---- 4. beta * Y ----
    if beta != 0.0:
        y_acc = y_packed.to(prod.dtype)
        prod = prod + beta * y_acc

    # ---- 5. Cast to output dtype ----
    return prod.to(out_dtype)

    
def test(a_shape, b_shape, c_shape, alpha, beta, device):
    byteSize = 2
    if (byteSize == 4):
        input_dtype = torch.float32
        output_dtype = torch.float32
    elif (byteSize == 2):
        input_dtype = torch.float16
        output_dtype = torch.float32
    elif (byteSize == 1):
        input_dtype = torch.int8
        output_dtype = torch.int32
    
    print(
        f"Testing gemm on {device} with alpha:{alpha}, beta:{beta}, a_shape:{a_shape} b_shape:{b_shape} c_shape:{c_shape} , input_dtype:{input_dtype}, output_dtype:{output_dtype}"
    )
    A = torch.randint(-128, 127, a_shape, dtype=input_dtype, device=device)
    B = torch.randint(-128, 127, b_shape, dtype=input_dtype, device=device)
    C = torch.randint(-128, 127, c_shape, device=device, dtype=output_dtype)
    if (byteSize !=1):
        A = torch.randn(a_shape, dtype=input_dtype, device=device)
        B = torch.randn(b_shape, dtype=input_dtype, device=device)
        C = torch.rand(c_shape, device=device, dtype=output_dtype)
    
    C_clone = C.clone()
    M, K = a_shape
    N = b_shape[1]

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))


    if device == "cuda":
        torch_matmul_time = performance.CudaProfile((gemm, (A, B, C_clone, alpha, beta, output_dtype))) 
        lib.gemm_cutlass.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]           
        custom_matmul_time = \
        performance.CudaProfile((lib.gemm_cutlass, 
        (A_ptr, B_ptr, C_ptr, M, K, N, alpha, beta, byteSize)))
    performance.logBenchmark(torch_matmul_time, custom_matmul_time)

    for i in range(40): #对于alpha , beta > 0的情况，此时需要特别注意
        C_clone = gemm(A, B, C_clone, alpha, beta, output_dtype)
    tmpa = C_clone.to('cpu').detach().numpy().flatten()
    
    tmpb = C.to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

    
# 解析命令行参数
parser = argparse.ArgumentParser(description="Test gemm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()   


test_cases = [
    # alpha, beta, a_shape, b_shape, c_shape
    (1.0, 2.0, (6, 2048), (2048, 2048), (6, 2048)),
    (1.0, 0.0, (4, 2050), (2050, 2048), (4, 2048)),
    (1.0, 0.0, (128, 512), (512, 1024), (128, 1024)),
    (1.0, 4.5, (256, 1024), (1024, 2048), (256, 2048)),
    (3.0, 0.5, (1024, 2048), (2048, 1040), (1024, 1040)),
    (1.0, 0.0, (1, 512), (512, 8), (1, 8)),
    (1.0, 0.0, (2048, 256), (256, 4096), (2048, 4096)),
    
]
if args.device == "mlu":
    import torch_mlu
elif args.device == "npu":
    import torch_npu
for alpha, beta, a_shape, b_shape, c_shape in test_cases:
    test(a_shape, b_shape, c_shape, alpha, beta, args.device)
