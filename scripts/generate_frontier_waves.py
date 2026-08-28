#!/usr/bin/env python3
"""Generate matched-BPW frontier configs for the Llama-3.2-1B campaign."""
import json
import hashlib
import re
from copy import deepcopy
from pathlib import Path

OUT = Path(__file__).parent / "configs"
BASE = {
    "bits": 2,
    "format": "qvq_v2b2_p32",
    "bank_count": 2,
    "rounding": "yaqa",
    "yaqa": {
        "seed": 0,
        "regularization": 0.15,
        "regularization_by_rate": [[2.0, 0.15], [2.5, 0.02], [3.0, 0.02], [3.5, 0.02], [4.0, 0.02]],
        "minimum_sequences": 182,
        "batch_size": 1,
        "sequence_sort": "desc",
        "activation_checkpointing": True,
        "v2b2_family_mode": "reselect",
        "sample_strategy": "full",
    },
    "device": "cuda:0",
    "offload_to_disk": False,
}

def emit(name, bits, dynamic, seed=0):
    cfg = deepcopy(BASE)
    cfg["bits"] = bits
    cfg["yaqa"]["seed"] = seed
    # The resolver is first-match-wins: arm-specific rules must precede
    # broad defaults or the special allocation is silently shadowed.
    entries = list(dynamic)
    if bits == 2:
        entries.extend([
            ("[0-9]+", "self_attn.q_proj|self_attn.k_proj", 2.5),
            ("[0-9]+", "self_attn.v_proj|self_attn.o_proj", 3.5),
            ("[0-9]+", "mlp.gate_proj|mlp.down_proj", 3.0),
            ("[0-9]+", "mlp.up_proj", 3.5),
        ])
    cfg["dynamic"] = {"+:^model\\.layers\\.%s\\.(%s)$" % (pat, mod): ({"bits": rate, **({"format": "qvq"} if rate > 3.5 else {})})
                      for pat, mod, rate in entries}
    path = OUT / ("llama32_1b_frontier_" + name + ".json")
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(Path.cwd()))


def resolved_fingerprints(path):
    """Return fingerprints of the effective per-module formats/rates."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [
        f"model.layers.{layer}.{role}"
        for layer in range(16)
        for role in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    ]
    resolved = {}
    for name in names:
        value = {"format": cfg["format"], "bits": cfg["bits"]}
        for pattern, override in cfg.get("dynamic", {}).items():
            regex = pattern[3:] if pattern.startswith("+:") else pattern
            if re.search(regex, name):
                value = {
                    "format": override.get("format", cfg["format"]),
                    "bits": override.get("bits", cfg["bits"]),
                }
                break
        resolved[name] = value
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    effective_payload = json.dumps(
        {"seed": cfg.get("yaqa", {}).get("seed", 0), "resolved": resolved},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return (
        hashlib.sha256(payload).hexdigest()[:12],
        hashlib.sha256(effective_payload).hexdigest()[:12],
    )

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
        seen = {}
        for name, bits, entries in arms:
            dynamic = []
            for layer_pat, module, rate in entries:
                dynamic.append((layer_pat, module, rate))
            seed = 1 if name.endswith("seed1") else 0
            rel = emit(name, bits, dynamic, seed=seed)
            resolved_fp, effective_fp = resolved_fingerprints(OUT / Path(rel).name)
            # Duplicate effective maps are almost always a shadowed rule bug.
            # The seed1 arm is an intentional same-map reproducibility control.
            if resolved_fp in seen:
                previous_name, previous_seed = seen[resolved_fp]
                intentional_seed_control = (
                    previous_seed != seed
                    and (name.endswith("seed1") or previous_name.endswith("seed1"))
                )
                if intentional_seed_control:
                    pass
                else:
                    raise RuntimeError(
                        f"duplicate effective allocation fingerprint {resolved_fp}: "
                        f"{previous_name} and {name}"
                    )
            seen[resolved_fp] = (name, seed)
            manifest.append({
                "id": name, "config": rel, "wave": wave,
                "resolved_fingerprint": resolved_fp,
                "effective_fingerprint": effective_fp,
                "yaqa_seed": seed,
            })
        (OUT / ("llama32_1b_frontier_wave%d.json" % wave)).write_text(
            json.dumps({"wave": wave, "arms": manifest}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
