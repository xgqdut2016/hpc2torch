import torch
import ctypes
import torch.nn.functional as F
import argparse
import numpy as np
import numpy
from utils import performance
# 添加上一层目录到模块搜索路径
import sys
import os



lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits

def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def pack_rows(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res

def torch_dequantize(q_weight, q_zeros, scales, g_idx, use_shuffle, bit, K, N):
    assert bit == 4, "Reference dequantization only supports 4-bit"
    group_size = K // scales.shape[0]
    pack_factor = 32 // bit

    # unpack q_weight: (K//pack_factor, N) -> (K, N)
    unpacked_q_weight = torch.empty(
        q_weight.shape[0] * pack_factor,
        q_weight.shape[1],
        dtype=torch.uint8,
        device=q_weight.device,
    )
    for i in range(pack_factor):
        unpacked_q_weight[i::pack_factor, :] = (q_weight >> (i * 4)) & 0x0F

    # unpack q_zeros: (num_groups, N//pack_factor) -> (num_groups, N)
    unpacked_q_zeros = torch.empty(
        q_zeros.shape[0],
        q_zeros.shape[1] * pack_factor,
        dtype=torch.uint8,
        device=q_zeros.device,
    )
    for i in range(pack_factor):
        unpacked_q_zeros[:, i::pack_factor] = (q_zeros >> (i * 4)) & 0x0F

    unpacked_q_zeros += 1
    unpacked_q_zeros = unpacked_q_zeros.to(scales.dtype)

    scale_zeros = unpacked_q_zeros * scales  # (num_groups, N)

    current_g_idx = torch.tensor(
        [i // group_size for i in range(K)], dtype=torch.int32, device=q_weight.device
    )

    scale_mat = scales[current_g_idx]  # (K, N)
    scale_zeros_mat = scale_zeros[current_g_idx]  # (K, N)

    # dequant: weight * scale - scale_zeros
    dequantized_b = unpacked_q_weight.to(scales.dtype) * scale_mat - scale_zeros_mat

    return dequantized_b.reshape(K, N)


def torch_gptq_gemm(
    a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit
):
    K, N = a.shape[1], b_q_weight.shape[1]

    b_dequant = torch_dequantize(
        b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit, K, N
    )
    c = torch.matmul(a, b_dequant)
    return c

def test(M, K, N, use_exllama, quant_bit, group_size, device):
    test_dtype = torch.float16
    print(
        f"Testing Gptq Gemm on {device} with M-K-N:{M, K, N}, use_exllama:{use_exllama}, quant_bit:{quant_bit}, group_size:{group_size}, dtype:{test_dtype}"
    )
    b_fp = torch.randn(K, N, dtype=test_dtype, device=device)

    assert K % group_size == 0, "K must be divisible by group_size"
    num_groups = K // group_size
    use_shuffle = use_exllama

    if use_shuffle:
        print(f"not support use_shuffle:{use_shuffle}")
        return
    else:
        g_idx = torch.tensor(
            [i // group_size for i in range(K)], dtype=torch.int32, device=device
        )
        b_shuffled = b_fp[g_idx]

    b_grouped = b_shuffled.reshape(num_groups, group_size, N)

    b_max = torch.max(b_grouped, dim=1, keepdim=True)[0]
    b_min = torch.min(b_grouped, dim=1, keepdim=True)[0]

    scales = (b_max - b_min) / (2**quant_bit - 1)
    scales = scales.clamp(min=1e-6)

    zeros_float = (-b_min / scales).round()

    q_b = (
        (b_grouped / scales + zeros_float).round().clamp(0, 2**quant_bit - 1).to(torch.uint8)
    )

    q_zeros_unpacked = zeros_float.to(torch.uint8) - 1

    b_q_weight = pack_rows(q_b.reshape(K, N), quant_bit, K, N)

    q_zeros_unpacked = q_zeros_unpacked.reshape(num_groups, N)
    b_gptq_qzeros = pack_cols(q_zeros_unpacked, quant_bit, num_groups, N)
    b_gptq_scales = scales.squeeze(1)

    a = torch.randn(M, K, dtype=test_dtype, device=device)
    C = torch.randn((M, N), dtype=test_dtype, device=device)

    A_ptr = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    B_ptr = ctypes.cast(b_q_weight.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_scales_ptr = ctypes.cast(b_gptq_scales.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_zeros_ptr = ctypes.cast(b_gptq_qzeros.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_g_idx_ptr = ctypes.cast(g_idx.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(ctypes.c_void_p))

    if device == "cuda":
        torch_gptq_gemm_time = performance.CudaProfile((torch_gptq_gemm, (a, b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx, use_shuffle, quant_bit))) 
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

    ans = torch_gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx, use_shuffle, quant_bit)
    rtol = 4e-2
    atol = 4e-2
    assert torch.allclose(C, ans, atol=atol, rtol=rtol)
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
    (1, 2048, 2048, True, 4, 128),
    (1, 2048, 4096, False, 4, 128),
    (1, 4096, 2048, False, 4, 128),
    (8, 2048, 2048, False, 4, 128),
    (8, 2048, 4096, False, 4, 128),
    (8, 4096, 2048, False, 4, 128),
    (128, 2048, 2048, False, 4, 128),
    (128, 2048, 4096, False, 4, 128),
    (128, 4096, 2048, False, 4, 128),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for M, K, N, use_exllama, quant_bit, group_size in test_cases:
    test(M, K, N, use_exllama, quant_bit, group_size, args.device)