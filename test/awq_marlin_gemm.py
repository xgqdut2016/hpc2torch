import torch
import ctypes
import numpy as np
import numpy
from functools import partial
import argparse
import itertools
import warnings

from utils import performance
from utils.scalar_type import scalar_types, ScalarType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def ops_marlin_int4_fp8_preprocess(qweights, qzeros=None, group_size=128):
    size_k, size_n = qweights.shape
    num_groups = size_k // group_size if qzeros is not None else -1
    dev = qweights.device
    output = torch.zeros_like(qweights).to(torch.int32)
    
    qweight_ptr = ctypes.cast(qweights.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    qzeros_ptr = ctypes.cast(qzeros.data_ptr(), ctypes.POINTER(ctypes.c_void_p)) if qzeros is not None else None
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    lib.marlin_int4_fp8_preprocess_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
    lib.marlin_int4_fp8_preprocess_nv(output_ptr, qweight_ptr, qzeros_ptr, size_k, size_n, num_groups)
    return output

def rand_data(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)

def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single

def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()

def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int | None,
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    assert quant_type.is_integer(), (
        "Floating point quantization may work but has not been tested"
    )
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided "
        "(-1 group_size is channelwise)"
    )

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = (
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
            )
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )

SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: torch.Tensor | None = None,
):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size,), dtype=torch.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert quant_type in SUPPORTED_GPTQ_QUANT_TYPES, (
        f"Unsupported gptq type = {quant_type}"
    )
    assert group_size in SUPPORTED_GROUP_SIZES + [size_k], (
        f"Unsupported groupsize = {group_size}"
    )

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert group_size < size_k, (
            "For act_order, groupsize = {} must be less than size_k = {}".format(
                group_size, size_k
            )
        )

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)

    return w_ref, w_q, w_s, g_idx, rand_perm
def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )

def get_weight_perm(num_bits: int, is_a_8bit: bool = False):
    perm_list: list[int] = []
    if is_a_8bit:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                    4 * (i % 4 + 4),
                    4 * (i % 4 + 4) + 1,
                    4 * (i % 4 + 4) + 2,
                    4 * (i % 4 + 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(2):
                perm_list.extend([p + 512 * j for p in perm1])
    else:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        if is_a_8bit:  # noqa: SIM108
            interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        if is_a_8bit:  # noqa: SIM108
            interleave = np.array([0, 1, 2, 3])
        else:
            interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

def marlin_permute_weights(
    q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE, is_a_8bit=False
):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    if is_a_8bit:
        # Permute weights to 32x32 marlin tiles
        q_w = q_w.reshape((size_k // (tile * 2), tile * 2, size_n // tile, tile))
    else:
        # Permute weights to 16x64 marlin tiles
        q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w
def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def marlin_weights(q_w, size_k, size_n, num_bits, perm, is_a_8bit=False):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm, is_a_8bit=is_a_8bit)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed

def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s
    
def marlin_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
):
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    size_k, size_n = w.shape
    num_bits = quant_type.size_bits

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        w, quant_type, group_size, act_order, test_perm
    )

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Reformat to marlin
    weight_perm = get_weight_perm(num_bits, is_a_8bit)
    marlin_q_w = marlin_weights(
        q_w, size_k, size_n, num_bits, weight_perm, is_a_8bit=is_a_8bit
    )
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size, is_a_8bit=is_a_8bit)

    if input_dtype == torch.float8_e4m3fn and quant_type == scalar_types.uint4b8:
        marlin_q_w = ops_marlin_int4_fp8_preprocess(marlin_q_w)
        marlin_s = marlin_s * 512

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list

def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )
    
def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )
    
def awq_marlin_gemm_torch(a_input, w_ref, b_bias):
    if b_bias == None:
        return torch.matmul(a_input, w_ref)
    else:
        return torch.matmul(a_input, w_ref) + b_bias.view(1, -1)

def test_marlin_gemm_with_bias(size_m, size_k, size_n, group_size, device):
    dataType = 1
    if dataType == 0:
        test_dtype = torch.float16
    elif dataType == 1:
        test_dtype = torch.bfloat16
    quant_type = scalar_types.uint4b8
    group_size = 128
    print(
        f"Testing awq_marlin_gemm_with_bias on {device} with M-K-N:({size_m, size_k, size_n}), group_size:{group_size}, test_dtype:{test_dtype}"
    )
    a_input = rand_data((size_m, size_k), test_dtype, device)
    b_weight = rand_data((size_k, size_n), test_dtype, device)
    b_bias = rand_data((size_n,), test_dtype, device) * 10

    marlin_bias = marlin_permute_bias(b_bias)

    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight, quant_type, group_size, False
    )

    marlin_zp = marlin_make_empty_g_idx(marlin_s.device)
    ans = awq_marlin_gemm_torch(a_input, w_ref, b_bias)
    
    output = torch.zeros_like(ans)
    
    a_ptr = ctypes.cast(a_input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    c_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_q_weight_ptr = ctypes.cast(marlin_q_w.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_bias_ptr = ctypes.cast(marlin_bias.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_scales_ptr = ctypes.cast(marlin_s.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    a_scales_ptr = None
    global_scales_ptr = None
    b_zeros_ptr = ctypes.cast(marlin_zp.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    g_idx_ptr = ctypes.cast(g_idx.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    perm_ptr = ctypes.cast(sort_indices.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_type_id = quant_type.id
    is_k_full=True
    use_atomic_add=False
    use_fp32_reduce=True
    is_zp_float=False
    b_q_size_0, b_q_size_1 = marlin_q_w.shape
    a_stride_0 = a_input.stride()[0]
    if marlin_zp is None or marlin_zp.numel() == 0:
        b_zeros_size_1 = -1
    else:
        b_zeros_size_1 = marlin_zp.shape[1]
    num_groups = size_k // group_size if group_size > 1 else -1
    
    
    if device == "cuda":
        torch_awq_marlin_gemm_time = performance.CudaProfile((awq_marlin_gemm_torch, (a_input, w_ref, b_bias)))  # 以毫秒为单位
        lib.awq_marlin_gemm_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), # c
            ctypes.POINTER(ctypes.c_void_p), # a
            ctypes.POINTER(ctypes.c_void_p), # b_q_weight
            ctypes.POINTER(ctypes.c_void_p), # b_bias
            ctypes.POINTER(ctypes.c_void_p), # b_scales
            ctypes.POINTER(ctypes.c_void_p), # a_scales
            ctypes.POINTER(ctypes.c_void_p), # global_scale
            ctypes.POINTER(ctypes.c_void_p), # b_zeros
            ctypes.POINTER(ctypes.c_void_p), # g_idx
            ctypes.POINTER(ctypes.c_void_p), # perm
            ctypes.c_int64, # b_type_id
            ctypes.c_bool, # is_k_full
            ctypes.c_bool, # use_atomic_add
            ctypes.c_bool, # use_fp32_reduce
            ctypes.c_bool, # is_zp_float
            ctypes.c_int, # size_m
            ctypes.c_int, # size_k
            ctypes.c_int, # size_n
            ctypes.c_int, # b_q_size_0
            ctypes.c_int, # b_q_size_1
            ctypes.c_int, # a_stride_0
            ctypes.c_int, # b_zeros_size_1
            ctypes.c_int, # num_groups
            ctypes.c_int # dataType
        ]
        custom_awq_marlin_gemm_time = \
        performance.CudaProfile((lib.awq_marlin_gemm_nv, 
                                 (c_ptr, a_ptr, b_q_weight_ptr, 
                                  b_bias_ptr, b_scales_ptr, a_scales_ptr,
                                  global_scales_ptr, b_zeros_ptr, g_idx_ptr, perm_ptr,
                                  b_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float,
                                  size_m, size_k, size_n, b_q_size_0, b_q_size_1,
                                  a_stride_0, b_zeros_size_1, num_groups, dataType)))
    
    performance.logBenchmark(torch_awq_marlin_gemm_time, custom_awq_marlin_gemm_time)
    

    max_diff = compute_max_diff(output, ans)

    assert max_diff < 0.04
    tmpa = output.float().detach().to('cpu').numpy().flatten()
    tmpb = ans.float().to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))
    rtol = atol / (max(abs(tmpb)) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
    
def test_marlin_gemm_subset_input(device):
    dataType = 0
    if dataType == 0:
        test_dtype = torch.float16
    elif dataType == 1:
        test_dtype = torch.bfloat16
    
    quant_type = scalar_types.uint4b8
    group_size = 128

    size_m, size_k, size_n = 32, 1024, 2048
    print(
        f"Testing awq_marlin_gemm_subset_input on {device} with M-K-N:({size_m, size_k, size_n}), group_size:{group_size}, test_dtype:{test_dtype}"
    )
    big_m = size_m * 2
    big_k = size_k * 2

    a_input = rand_data((big_m, big_k), test_dtype, device)[8 : size_m + 8, 8 : size_k + 8]
    b_weight = rand_data((size_k, size_n), test_dtype, device)

    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight, quant_type, group_size, False
    )

    marlin_zp = marlin_make_empty_g_idx(marlin_s.device)
    
    ans = awq_marlin_gemm_torch(a_input, w_ref, None)
    
    output = torch.zeros_like(ans)
    
    a_ptr = ctypes.cast(a_input.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    c_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_q_weight_ptr = ctypes.cast(marlin_q_w.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_bias_ptr = None
    b_scales_ptr = ctypes.cast(marlin_s.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    a_scales_ptr = None
    global_scales_ptr = None
    b_zeros_ptr = ctypes.cast(marlin_zp.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    g_idx_ptr = ctypes.cast(g_idx.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    perm_ptr = ctypes.cast(sort_indices.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_type_id = quant_type.id
    is_k_full=True
    use_atomic_add=False
    use_fp32_reduce=True
    is_zp_float=False
    b_q_size_0, b_q_size_1 = marlin_q_w.shape
    a_stride_0 = a_input.stride()[0]
    if marlin_zp is None or marlin_zp.numel() == 0:
        b_zeros_size_1 = -1
    else:
        b_zeros_size_1 = marlin_zp.shape[1]
    num_groups = size_k // group_size if group_size > 1 else -1
    
    if device == "cuda":
        torch_awq_marlin_gemm_time = performance.CudaProfile((awq_marlin_gemm_torch, (a_input, w_ref, None)))  # 以毫秒为单位
        lib.awq_marlin_gemm_nv.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), # c
            ctypes.POINTER(ctypes.c_void_p), # a
            ctypes.POINTER(ctypes.c_void_p), # b_q_weight
            ctypes.POINTER(ctypes.c_void_p), # b_bias
            ctypes.POINTER(ctypes.c_void_p), # b_scales
            ctypes.POINTER(ctypes.c_void_p), # a_scales
            ctypes.POINTER(ctypes.c_void_p), # global_scale
            ctypes.POINTER(ctypes.c_void_p), # b_zeros
            ctypes.POINTER(ctypes.c_void_p), # g_idx
            ctypes.POINTER(ctypes.c_void_p), # perm
            ctypes.c_int64, # b_type_id
            ctypes.c_bool, # is_k_full
            ctypes.c_bool, # use_atomic_add
            ctypes.c_bool, # use_fp32_reduce
            ctypes.c_bool, # is_zp_float
            ctypes.c_int, # size_m
            ctypes.c_int, # size_k
            ctypes.c_int, # size_n
            ctypes.c_int, # b_q_size_0
            ctypes.c_int, # b_q_size_1
            ctypes.c_int, # a_stride_0
            ctypes.c_int, # b_zeros_size_1
            ctypes.c_int, # num_groups
            ctypes.c_int # dataType
        ]
        custom_awq_marlin_gemm_time = \
        performance.CudaProfile((lib.awq_marlin_gemm_nv, 
                                 (c_ptr, a_ptr, b_q_weight_ptr, 
                                  b_bias_ptr, b_scales_ptr, a_scales_ptr,
                                  global_scales_ptr, b_zeros_ptr, g_idx_ptr, perm_ptr,
                                  b_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float,
                                  size_m, size_k, size_n, b_q_size_0, b_q_size_1,
                                  a_stride_0, b_zeros_size_1, num_groups, dataType)))
    
    performance.logBenchmark(torch_awq_marlin_gemm_time, custom_awq_marlin_gemm_time)
    

    max_diff = compute_max_diff(output, ans)

    assert max_diff < 0.04
    tmpa = output.float().detach().to('cpu').numpy().flatten()
    tmpb = ans.float().to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))
    rtol = atol / (max(abs(tmpb)) + 1e-8)

    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))


    max_diff = compute_max_diff(output, ans)

    assert max_diff < 0.04
    

    
parser = argparse.ArgumentParser(description="Test marlin gemm on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = [
    # size_m, size_k, size_n, group_size
    (1, 1024, 2048, 128),
    (256, 1024, 2048, 128),
]

if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for size_m, size_k, size_n, group_size in test_cases:
    test_marlin_gemm_with_bias(size_m, size_k, size_n, group_size, args.device)
    
test_marlin_gemm_subset_input(args.device)
    