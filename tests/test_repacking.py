# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import copy  # noqa: E402
import unittest  # noqa: E402

# isort: off
import torch # noqa: E402
import torch.nn as nn # noqa: E402
import gptqmodel_marlin_cuda # noqa: E402
# isort: on
from gptqmodel.nn_modules.qlinear.qlinear_exllama import ExllamaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.qlinear_marlin import _get_perms, dequantize_weight  # noqa: E402


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
    def test_marlin_fast_repacking(self):
        k = 2048
        n = 1024
        m = 5
        group_size = 128

        _, linear, s = gen_quant4(k, n, group_size)
        use_act_order = False
        exllama_linear = ExllamaQuantLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=use_act_order,
            infeatures=k,
            outfeatures=n,
            bias=False)

        exllama_linear._use_act_order = use_act_order

        zeros = torch.full((k // group_size, n), 8, dtype=torch.int32)

        exllama_linear.pack(linear, s.T, zeros.T, g_idx=None)

        exllama_linear = exllama_linear.to("cuda")

        exllama_linear.post_init()

        # Adapted from utils.marlin_utils.convert_to_marlin
        dequantized_weight, dequantized_qzeros = dequantize_weight(exllama_linear)
        dequantized_weight = dequantized_weight.to(torch.float16)

        self.assertTrue(torch.all(dequantized_qzeros == 8))

        linear_module = torch.nn.Linear(
            in_features=k,
            out_features=n,
            bias=False,
            dtype=torch.float16,
            device="cuda",
        )
        linear_module.weight.data.copy_(linear.weight.data)  # Not using dequantized_weight to avoid approx

        # Create new linear method and copy to model.
        marlin_linear = MarlinQuantLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            infeatures=k,
            outfeatures=n,
            bias=False,
        )

        marlin_linear.pack(linear_module.to("cuda"), scales=copy.deepcopy(exllama_linear.scales.data.t()).to("cuda"))

        inp = torch.rand(m, k, dtype=torch.float16, device="cuda")

        exllama_linear = exllama_linear.to("cuda")
        marlin_linear = marlin_linear.to("cuda")
        with torch.no_grad():
            res_exllama = exllama_linear(inp)
            res_marlin = marlin_linear(inp)

        reldiff = (res_exllama - res_marlin).abs() / (res_exllama.abs() + 1e-12)
        print(f"reldiff = {reldiff}, ",torch.mean(reldiff))
        # torch.mean(reldiff) 100 times:
        # Max: 0.010498046875
        # Min: 0.00415802001953125
        # Average: 0.006191991990612399
        self.assertLess(torch.mean(reldiff).item(), 0.0068)

        weight_repacked = gptqmodel_marlin_cuda.gptq_repack(exllama_linear.qweight)
        self.assertTrue(torch.allclose(weight_repacked, marlin_linear.B))

        _, _scale_perm, _scale_perm_single = _get_perms()

        s = exllama_linear.scales.data.clone()
        if group_size != k:
            s = s.reshape((1, -1))
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, n)).contiguous()

        self.assertTrue(torch.allclose(s, marlin_linear.s))
