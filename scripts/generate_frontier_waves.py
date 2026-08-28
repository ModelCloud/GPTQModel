#!/usr/bin/env python3
"""Generate matched-BPW frontier configs for the Llama-3.2-1B campaign."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "configs"
BASE = {
    "bits": 2,
    "format": "qvq_v2b2_p32",
    "bank_count": 2,
    "rounding": "yaqa",
    "yaqa": {"seed": 0, "regularization": 0.15},
    "device": "cuda:0",
    "offload_to_disk": False,
}

def emit(name, bits, dynamic):
    cfg = dict(BASE)
    cfg["bits"] = bits
    entries = []
    if bits == 2:
        entries.extend([
            ("[0-9]+", "self_attn.q_proj|self_attn.k_proj", 2.5),
            ("[0-9]+", "self_attn.v_proj|self_attn.o_proj", 3.5),
            ("[0-9]+", "mlp.gate_proj|mlp.down_proj", 3.0),
            ("[0-9]+", "mlp.up_proj", 3.5),
        ])
    for pat, mod, rate in dynamic:
        entries.append((pat, mod, rate))
    cfg["dynamic"] = {"+:^model\\.layers\\.%s\\.(%s)$" % (pat, mod): ({"bits": rate, **({"format": "qvq"} if rate > 3.5 else {})})
                      for pat, mod, rate in entries}
    path = OUT / ("llama32_1b_frontier_" + name + ".json")
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(Path.cwd()))

def main():
    wave1 = [
        ("w1_up4_l12_15", 2, [("(12|13|14|15)", "mlp.up_proj", 4)]),
        ("w1_up4_l10_13", 2, [("(10|11|12|13)", "mlp.up_proj", 4)]),
        ("w1_up4_l8_11", 2, [("(8|9|10|11)", "mlp.up_proj", 4)]),
        ("w1_up4_interleaved", 2, [("(8|10|12|14)", "mlp.up_proj", 4)]),
        ("w1_up4_down35_l14_15", 2, [("(14|15)", "mlp.up_proj", 4), ("(14|15)", "mlp.down_proj", 3.5)]),
        ("w1_up4_gate35_l14_15", 2, [("(14|15)", "mlp.up_proj", 4), ("(14|15)", "mlp.gate_proj", 3.5)]),
        ("w1_o4_all", 2, [("[0-9]+", "self_attn.o_proj", 4)]),
        ("w1_v4_up4_l13_15", 2, [("[0-9]+", "self_attn.v_proj", 4), ("(13|14|15)", "mlp.up_proj", 4)]),
    ]
    wave2 = [
        ("w2_flat35_up4_l8_15", 3.5, [("(8|9|10|11|12|13|14|15)", "mlp.up_proj", 4)]),
        ("w2_flat35_up4_l6_15", 3.5, [("(6|7|8|9|10|11|12|13|14|15)", "mlp.up_proj", 4)]),
        ("w2_flat35_up4_l4_15", 3.5, [("(4|5|6|7|8|9|10|11|12|13|14|15)", "mlp.up_proj", 4)]),
        ("w2_flat35_up4_down4_l12_15", 3.5, [("(12|13|14|15)", "mlp.up_proj", 4), ("(12|13|14|15)", "mlp.down_proj", 4)]),
        ("w2_flat35_up4_gate4_l12_15", 3.5, [("(12|13|14|15)", "mlp.up_proj", 4), ("(12|13|14|15)", "mlp.gate_proj", 4)]),
        ("w2_flat35_up4_o4_all", 3.5, [("(12|13|14|15)", "mlp.up_proj", 4), ("[0-9]+", "self_attn.o_proj", 4)]),
        ("w2_flat35_up4_v4_l9_15", 3.5, [("(9|10|11|12|13|14|15)", "mlp.up_proj", 4), ("[0-9]+", "self_attn.v_proj", 4)]),
        ("w2_flat35_seed1", 3.5, []),
    ]
    # Dynamic patterns are converted below to the repository's full module regex form.
    for wave, arms in ((1, wave1), (2, wave2)):
        manifest = []
        for name, bits, entries in arms:
            dynamic = []
            for layer_pat, module, rate in entries:
                dynamic.append((layer_pat, module, rate))
            rel = emit(name, bits, dynamic)
            manifest.append({"id": name, "config": rel, "wave": wave})
        (OUT / ("llama32_1b_frontier_wave%d.json" % wave)).write_text(
            json.dumps({"wave": wave, "arms": manifest}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
