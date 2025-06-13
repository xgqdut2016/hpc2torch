import torch
import ctypes
import numpy as np
import torch.nn as nn
import math
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def quantize(x, scale, zero, minq, maxq):
    if scale.shape[1] == 1:
        q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
        return scale * (q - zero)
    else:
        group_size = x.shape[1] // scale.shape[1]
        y = torch.zeros_like(x)
        for j in range(scale.shape[1]):
            q = torch.clamp(
                torch.round(
                    x[:, j * group_size : (j + 1) * group_size] / scale[:, j : j + 1]
                )
                + zero[:, j : j + 1],
                minq,
                maxq,
            )
            y[:, j * group_size : (j + 1) * group_size] = scale[:, j : j + 1] * (
                q - zero[:, j : j + 1]
            )
        return y


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("minq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits=4,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        sign_ed=False,
    ):
        if sign_ed:  # 有符号量化，范围是[-8,7]
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
            self.minq = -torch.tensor(2 ** (bits - 1))
        else:  # 无符号量化，范围是[0,15]
            self.maxq = torch.tensor(2**bits - 1)
            self.minq = -torch.tensor(0)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        self.minq = self.minq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / (self.maxq - self.minq)
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + self.minq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / (self.maxq - self.minq)
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(
                    x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.minq, self.maxq
                )
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            # self.scale = self.scale.unsqueeze(1)
            # self.zero = self.zero.unsqueeze(1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.minq, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, weight):
        self.weight = weight
        self.dev = self.weight.device

        self.rows = self.weight.shape[0]
        self.columns = self.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, group_size=-1):
        W = self.weight.clone()

        W = W.float()

        # tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H.to("cpu")).to(
            H.device
        )  # 对于CUDA来说，这个地方直接在CUDA上做cholesky分解可能会失败
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        num_groups = self.columns // group_size
        if group_size == -1:
            scale = self.quantizer.scale.clone()
            zero = self.quantizer.zero.clone()
        else:
            scale = torch.zeros(self.rows, num_groups)
            zero = torch.zeros(self.rows, num_groups)
        for index in range(self.columns // blocksize):
            i1 = index * blocksize
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if (i1 + i) % group_size == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i) : (i1 + i + group_size)], weight=True
                        )
                        ind = index * blocksize // group_size + i // group_size

                        scale[:, ind : ind + 1] = self.quantizer.scale
                        zero[:, ind : ind + 1] = self.quantizer.zero

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.minq,
                    self.quantizer.maxq,
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        print("error", torch.sum(Losses).item())

        self.weight = Q.reshape(self.weight.shape).to(self.weight.dtype)
        self.scale = scale.to(self.weight.dtype)
        self.zero = zero.to(self.weight.dtype)


def get_scale_zero(b, a, c, group_size, bits, sym, sign_ed):
    weight = b.clone()
    inp = a.clone()
    out = c.clone()
    gptq = GPTQ(weight)
    gptq.quantizer = Quantizer()
    gptq.quantizer.configure(
        bits=bits, perchannel=True, sym=sym, mse=False, sign_ed=sign_ed
    )
    gptq.add_batch(inp, out)
    gptq.fasterquant(group_size=group_size)

    return (
        gptq.weight.to(weight.device),
        gptq.scale.to(weight.device),
        gptq.zero.to(weight.device),
    )
def pack(weight, scale, zero, minq, maxq):
    intweight = torch.clamp(torch.round(weight / scale + zero), minq, maxq).to(
        torch.int32
    )
    qweight = torch.zeros(
        [weight.shape[0], weight.shape[1] // 8], dtype=torch.int32, device=weight.device
    )
    for i in range(intweight.shape[1]):
        qweight[:, i // 8] |= intweight[:, i] << (4 * (i % 8))
    return qweight

def _get_perms():
    perm = []
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
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()
class MarlinLayer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError("`infeatures` must be divisible by `groupsize`.")
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer(
            "B", torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int)
        )
        self.register_buffer(
            "s", torch.empty((self.k // groupsize, self.n), dtype=torch.half)
        )

    def forward(self, A):
        C = torch.empty(
            A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device
        )
        marlin_matmul(
            A.view((-1, A.shape[-1])),
            self.B,
            C.view((-1, C.shape[-1])),
            self.s,
        )
        return C

    def pack(self, linear, scales):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        tile = 16
        maxq = 2**4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
def gen_quant4(m, n, groupsize=-1):
    DEV = torch.device("cuda:0")
    tile = 16
    maxq = 2**4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = MarlinLayer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B.reshape(m // 8, n)
    s = layer.s
    return ref, q, s

# PyTorch implementation for matrix multiplication
def quantize_gptq(a, b):  # 昇腾芯片的CPU不支持转置计算
    ans = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(b.dtype)
    return ans

def test(torch_device,
    M,
    K,
    N):
    dtype = torch.float16
    print(
        f"Testing MatmulGptq on {torch_device}" f" M:{M}, K:{K}, N:{N}, dtype:{dtype}"
    )
    torch.manual_seed(12)
    a = 1e0 * torch.randn([K, M], dtype=dtype).to(torch_device)
    layer = nn.Linear(K, N)
    b = 1e0 * layer.weight.data.to(dtype).to(torch_device)
    c = torch.zeros([N, M], dtype=dtype).to(torch_device)
    is_weight_transposed = False
    sign_ed = False
    sym = False
    if torch_device != "cpu":
        is_weight_transposed = True

    group_size = -1
    num_groups = 1
    if group_size == -1:
        num_groups = 1
    else:
        num_groups = K // group_size
    
    packed_weights = torch.zeros([N, K // 8], dtype=torch.int32).to(torch_device)
    s = torch.zeros([N, num_groups], dtype=dtype).to(torch_device)
    z = torch.zeros([N, num_groups], dtype=dtype).to(torch_device)

    bits = 4
    maxq = 2**bits - 1
    minq = 0
    if sign_ed:  # 有符号量化，范围是[-8,7]
        maxq = 2 ** (bits - 1) - 1
        minq = -(2 ** (bits - 1))

    if torch_device == "cuda":
        B_ref, packed_weights, s = gen_quant4(K, N, groupsize=group_size)
        b = B_ref.t()
        packed_weights = packed_weights.t()
        s = s.t()
        print(a.shape, b.shape, packed_weights.shape, s.shape)
    if is_weight_transposed:
        ans = quantize_gptq(a.t(), b.t())
    else:
        ans = quantize_gptq(b, a)
    if torch_device == "cpu":
        b_ref, s, z = get_scale_zero(
            b, a.t(), c, group_size, bits, sym, sign_ed=sign_ed
        )  # 无符号量化

        packed_weights = pack(b_ref, s, z, minq, maxq)

    if is_weight_transposed:
        output_ptr = ctypes.cast(c.t().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        inputA_ptr = ctypes.cast(a.t().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        inputB_ptr = ctypes.cast(b.t().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        packed_weights_ptr = ctypes.cast(packed_weights.t().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        scale_ptr = ctypes.cast(s.t().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        zero_ptr = ctypes.cast(z.t().data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    else:
        output_ptr = ctypes.cast(c.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        inputA_ptr = ctypes.cast(a.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        inputB_ptr = ctypes.cast(b.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        packed_weights_ptr = ctypes.cast(packed_weights.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        scale_ptr = ctypes.cast(s.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
        zero_ptr = ctypes.cast(z.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    
    if torch_device == "cpu":
        torch_gptq_time = performance.CpuProfile((quantize_gptq, (b, a)))  # 可以替换为mul, div
        lib.quant_cpu.argtypes = [
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
        quant_time = 0
        # quant_time = \
        # performance.CpuProfile((lib.quant_cpu, 
        # (inputA_ptr, inputB_ptr, packed_weights_ptr, scale_ptr, zero_ptr, M, K, N, group_size)))
        lib.caculate_cpu.argtypes = [
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
        caculate_time = \
        performance.CpuProfile((lib.caculate_cpu, 
        (output_ptr, inputA_ptr, packed_weights_ptr, scale_ptr, zero_ptr, M, K, N, group_size)))
        custom_gptq_time = quant_time + caculate_time
    if torch_device == "cuda":
        torch_gptq_time = performance.CpuProfile((quantize_gptq, (a.t(), b.t())))  # 可以替换为mul, div
        lib.caculate_cuda.argtypes = [
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
        custom_gptq_time = \
        performance.CudaProfile((lib.caculate_cuda, 
        (output_ptr, inputA_ptr, packed_weights_ptr, scale_ptr, zero_ptr, M, K, N, group_size)))
    
    performance.logBenchmark(torch_gptq_time, custom_gptq_time)
    atol = 1e-3
    rtol = 1e-3
    if is_weight_transposed:
        tmpa = quantize_gptq(a.t(), b.t()).to('cpu').numpy().flatten()
        tmpc = c.t().to('cpu').numpy().flatten()
        for i in range(tmpa.shape[0]):
            if abs(tmpa[i] - tmpc[i]) > atol + rtol * abs(tmpa[i]):
                print(tmpa[i], tmpc[i], abs(tmpa[i] - tmpc[i]), rtol * abs(tmpa[i]))
                break
        torch.allclose(c.t(), quantize_gptq(a.t(), b.t()), atol=atol, rtol=rtol)
    else:
        tmpa = quantize_gptq(b, a).to('cpu').numpy().flatten()
        tmpc = c.to('cpu').numpy().flatten()
        for i in range(tmpa.shape[0]):
            if abs(tmpa[i] - tmpc[i]) > atol + rtol * abs(tmpa[i]):
                print(tmpa[i], tmpc[i], abs(tmpa[i] - tmpc[i]), rtol * abs(tmpa[i]))
                break
        torch.allclose(c, quantize_gptq(b, a), atol=atol, rtol=rtol)
    atol = max(abs(tmpa - tmpc))

    rtol = atol / max(abs(tmpc) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test gptq on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu', 'npu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    

test_cases = []

MODELS = {
    "7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    # "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    # "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    # "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
}

# Loop through models and layers to generate the new _TEST_CASES
for _, layers in MODELS.items():
    for layer in layers:
        for batch in [1, 16]:
            test_cases.append(((batch, layer[0], layer[1])))
if args.device == 'mlu':
    import torch_mlu
if args.device == 'npu':
    import torch_npu
# 执行过滤后的测试用例
for M, K, N in test_cases:
    test(args.device, M, K, N)