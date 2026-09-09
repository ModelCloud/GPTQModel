"""Install the existing F6 snapshot as canonical FP32 QVQ operators."""

import json


def install_f6(model, snapshot):
    import torch
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((snapshot / "quantize_config.json").read_text())

    def read(name):
        with safe_open(str(snapshot / index[name]), framework="pt") as handle:
            return handle.get_tensor(name).cuda()

    class CanonicalQVQ(torch.nn.Module):
        def __init__(self, inner, su, sv):
            super().__init__()
            self.register_buffer("inner", inner.float())
            self.register_buffer("su", su.float())
            self.register_buffer("sv", sv.float())

        def forward(self, inputs):
            return matmul_hadU(matmul_hadU(inputs.float()*self.su) @ self.inner)*self.sv

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in index:
                param.copy_(read(name))
    if cfg.get("activation") or cfg.get("incoherence") != "rht":
        raise ValueError("Unsupported F6 transform")
    for name in sorted(index):
        if not name.endswith(".trellis"):
            continue
        prefix = name[:-8]
        trellis, su, sv = [read(prefix + "." + suffix) for suffix in ("trellis", "SU", "SV")]
        p32 = prefix + ".bank_alt_id" in index
        bank = read(prefix + ".bank_ids") if prefix + ".bank_ids" in index else None
        alt = read(prefix + ".bank_alt_id") if p32 else None
        inner = reconstruct_qvq_inner_weight(
            trellis, bits=trellis.shape[-1]/8, in_features=su.numel(), out_features=sv.numel(),
            bank_ids=bank, bank_alt_id=alt, codebook_version=cfg["codebook"], v2b2_p32=p32)
        parent, leaf = prefix.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, CanonicalQVQ(inner, su, sv))
    return {"snapshot_quantized_modules": sum(n.endswith(".trellis") for n in index),
            "snapshot_p32_modules": sum(n.endswith(".bank_alt_id") for n in index)}
