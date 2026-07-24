"""Time two GPTQ layer quantizes to separate compile overhead."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn as nn
import gptqmodel.quantization.gptq as gptq_module
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.torch import torch_compile


def make_and_quantize():
    mod = nn.Linear(2048, 8192, bias=False, device="cuda:0")
    qcfg = QuantizeConfig(bits=4, group_size=128, damp_percent=0.05, desc_act=False, sym=True)
    g = GPTQ(mod, qcfg=qcfg)
    g.quantizer.configure(perchannel=True, grid=100, maxshrink=0.8, trits=False)

    inp = torch.randn(64 * 128, 2048, device="cuda:0", dtype=torch.bfloat16)
    out = torch.randn(64 * 128, 8192, device="cuda:0", dtype=torch.bfloat16)
    g.add_batch(inp, out)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    g.quantize()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main():
    gptq_module._HESSIAN_INVERSE_TRY = torch_compile(
        gptq_module._hessian_inverse_try_cholesky, backend="inductor", fullgraph=False
    )
    gptq_module._HESSIAN_INVERSE_FACTOR = torch_compile(
        gptq_module._hessian_inverse_factor, backend="inductor", fullgraph=False
    )

    t1 = make_and_quantize()
    print(f"first quantize (compile incl): {t1:.3f} ms")
    t2 = make_and_quantize()
    print(f"second quantize (cached): {t2:.3f} ms")


if __name__ == "__main__":
    main()
