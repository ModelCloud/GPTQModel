# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import copy  # noqa: E402
import unittest  # noqa: E402

# isort: off
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import gptqmodel_marlin_cuda  # noqa: E402
# isort: on
from gptqmodel.nn_modules.qlinear.qlinear_exllama import ExllamaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear, _get_perms, dequantize_weight  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_tritonv2 import TritonV2QuantLinear  # noqa: E402


def gen_quant4(k, n, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((k, n), dtype=torch.half, device="cpu")

    original_w = w.clone()

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq

    # Quantize.
    w = torch.round(w / s).int()

    # Unsigned storage.
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)

    # Dequantize.
    ref = (w - (maxq + 1) // 2).half() * s

    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()

    return original_w, linear, s


class TestRepacking(unittest.TestCase):
    def test_triton_compare_exllama(self):
        k = 2048
        n = 1024
        group_size = 128

        _, linear, s = gen_quant4(k, n, group_size)
        zeros = torch.full((k // group_size, n), 8, dtype=torch.int32)

        exllama_linear = ExllamaQuantLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=True,
            infeatures=k,
            outfeatures=n,
            bias=False)

        exllama_linear.pack(linear, s.T, zeros.T, g_idx=None)

        dequantized_weight, dequantized_qzeros = dequantize_weight(exllama_linear)
        dequantized_weight = dequantized_weight.to(torch.float16)

        self.assertTrue(torch.equal(dequantized_weight, linear.weight))
        self.assertTrue(torch.all(dequantized_qzeros == 8))

        triton_v2_linear = TritonV2QuantLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=True,
            infeatures=k,
            outfeatures=n,
            bias=False)

        triton_v2_linear.pack(linear, s.T, zeros.T, g_idx=None)

        dequantized_weight, dequantized_qzeros = dequantize_weight(triton_v2_linear)
        dequantized_weight = dequantized_weight.to(torch.float16)

        self.assertTrue(torch.equal(dequantized_weight, linear.weight))
        self.assertTrue(torch.all(dequantized_qzeros == 8))

        self.assertTrue(torch.allclose(exllama_linear.qweight, triton_v2_linear.qweight))
        self.assertTrue(torch.allclose(exllama_linear.scales, triton_v2_linear.scales))
        self.assertTrue(torch.allclose(exllama_linear.qzeros, triton_v2_linear.qzeros))


    def test_marlin_gptq_repack(self):
        k = 2048
        n = 1024
        group_size = 128

        _, linear, s = gen_quant4(k, n, group_size)
        zeros = torch.full((k // group_size, n), 8, dtype=torch.int32)

        triton_v2_linear = TritonV2QuantLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            infeatures=k,
            outfeatures=n,
            bias=False)

        triton_v2_linear.pack(linear, s.T, zeros.T, g_idx=None)

        dequantized_weight, dequantized_qzeros = dequantize_weight(triton_v2_linear)
        dequantized_weight = dequantized_weight.to(torch.float16)

        self.assertTrue(torch.equal(dequantized_weight, linear.weight))
        self.assertTrue(torch.all(dequantized_qzeros == 8))

        marlin_linear = MarlinQuantLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            infeatures=k,
            outfeatures=n,
            bias=False,
        )

        marlin_linear.pack(linear.to("cuda"), scales=copy.deepcopy(triton_v2_linear.scales.data.t()).to("cuda"))

        # make sure triton_v2_linear and marlin_linear on same device
        triton_v2_linear.to("cuda")
        marlin_linear.to("cuda")

        # repack the qweight using triton_v2_linear pack, then check if it is allclose to B from the marlin pack.
        weight_repacked = gptqmodel_marlin_cuda.gptq_repack(triton_v2_linear.qweight)

        _, _scale_perm, _scale_perm_single = _get_perms()

        # adjust the scales of triton_v2_linear to make them compatible with marlin, then check if they are allclose to the scales after marlin pack.
        s = triton_v2_linear.scales.data.clone()
        if group_size != k:
            s = s.reshape((1, -1))
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, n)).contiguous()

        self.assertTrue(torch.allclose(weight_repacked, marlin_linear.B))
        self.assertTrue(torch.allclose(s, marlin_linear.s))
