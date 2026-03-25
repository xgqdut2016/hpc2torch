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

def gptq_gemm_torch(A, B, scale, zero, b_g_idx, group_size, quant_bit):
    """
    vLLM-style W4A16 / W8A16 (with b_g_idx)

    A: [M, K]
    B:
        - int4: [K/2, N] uint8
        - int8: [K,   N] uint8
    scale: [num_groups, N]
    zero:  [num_groups, N] int32
    b_g_idx: [K] int32   ⭐ group index for each K
    group_size: int
    quant_bit: 4 or 8

    return:
        out: [M, N]
    """

    M, K = A.shape
    device = A.device
    dtype = A.dtype

    # ------------------------
    # 1. 解码权重
    # ------------------------
    if quant_bit == 4:
        Bf = B.view(-1)

        low  = (Bf & 0xF).to(torch.int8)
        high = ((Bf >> 4) & 0xF).to(torch.int8)

        vals = torch.stack([low, high], dim=1).view(-1)

        # signed int4
        vals = vals - ((vals >= 8).to(torch.int8) * 16)

        N = B.shape[1]
        W_int = vals[:K * N].view(K, N)

    elif quant_bit == 8:
        # uint8 -> int8
        tmp = B.to(torch.int16)
        tmp = (tmp - 256) * (tmp >= 128) + tmp * (tmp < 128)
        W_int = tmp.to(torch.int8)

        N = B.shape[1]

    else:
        raise ValueError("quant_bit must be 4 or 8")

    # ------------------------
    # 2. Dequant（核心：b_g_idx）
    # ------------------------
    if group_size == -1:
        # 特殊情况：所有K共用一个group
        s = scale[0]
        z = zero[0].to(dtype)

        W = s * (W_int.to(dtype) - z)

    else:
        # ⭐ 用 b_g_idx 替代 k//group_size
        group_ids = b_g_idx.to(torch.long)   # [K]

        s = scale[group_ids]                 # [K,N]
        z = zero[group_ids].to(dtype)

        W = s * (W_int.to(dtype) - z)

    # ------------------------
    # 3. GEMM
    # ------------------------
    out = A @ W

    return out

def test(M, K, N, use_exllama, quant_bit, group_size, device):
    test_dtype = torch.float16
    print(
        f"Testing Gptq Gemm on {device} with M-K-N:{M, K, N}, use_exllama:{use_exllama}, quant_bit:{quant_bit}, group_size:{group_size}, dtype:{test_dtype}"
    )
    A = torch.randn((M, K), dtype=test_dtype, device=device)
    if quant_bit == 4:
        B = torch.randint(0, 16, (N, k // 2), dtype=torch.uint8, device=device)
    elif quant_bit == 8:
        B = torch.randint(0, 256, (N, K // 4), dtype=torch.uint8, device=device)
    C = torch.randn((M, N), dtype=test_dtype, device=device)
    num_groups = 1 if group_size == -1 else K // group_size
    block_size = 1 if group_size == -1 else group_size
    b_scales = torch.randn((num_groups, N), dtype=test_dtype, device=device)
    b_zeros = torch.randint(-2000000000, 2000000000, (num_groups, N), dtype=torch.int32, device=device)
    b_g_idx = torch.randint(0, K // block_size, (K, ), dtype=torch.int32, device=device)

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_scales_ptr = ctypes.cast(b_scales.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_zeros_ptr = ctypes.cast(b_zeros.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_g_idx_ptr = ctypes.cast(b_g_idx.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    if device == "cuda":
        torch_gptq_gemm_time = performance.CudaProfile((gptq_gemm_torch, (A, B, b_scales, b_zeros, b_g_idx, group_size, quant_bit))) 
        lib.gptq_gemm_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int
        ]
        
        custom_gptq_gemm_time = \
        performance.CudaProfile((lib.gptq_gemm_nv, (
                      C_ptr, A_ptr, B_ptr, b_scales_ptr, b_zeros_ptr, b_g_idx_ptr, M, K, N, num_groups, K //2, use_exllama, quant_bit)))
        
    performance.logBenchmark(torch_gptq_gemm_time, custom_gptq_gemm_time)

    atol = 1e-3; rtol = 5e-2
    ans = gptq_gemm_torch(A, B, b_scales, b_zeros, b_g_idx, group_size, quant_bit)
    tmpa = C.float().detach().to('cpu').numpy().flatten()
    tmpb = ans.float().to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test gptq gemm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
    # M, K, N, use_exllama, quant_bit, group_size
    (16, 1024, 512, False, 4, 128),
    (128, 256, 32, False, 4, 128),
    (512, 2048, 128, True, 4, 128),
    (1024, 1024, 128, False, 8, 128),
    (1024, 1024, 128, True, 8, 128),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for M, K, N, use_exllama, quant_bit, group_size in test_cases:
    test(M, K, N, use_exllama, quant_bit, group_size, args.device)