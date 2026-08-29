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

def emit(name, bits, dynamic, seed=0, defaults=None):
    cfg = deepcopy(BASE)
    cfg["bits"] = bits
    cfg["yaqa"]["seed"] = seed
    # The resolver is first-match-wins: arm-specific rules must precede
    # broad defaults or the special allocation is silently shadowed.
    entries = list(dynamic)
    if defaults is None and bits == 2:
        defaults = [
            ("[0-9]+", "self_attn.q_proj|self_attn.k_proj", 2.5),
            ("[0-9]+", "self_attn.v_proj|self_attn.o_proj", 3.5),
            ("[0-9]+", "mlp.gate_proj|mlp.down_proj", 3.0),
            ("[0-9]+", "mlp.up_proj", 3.5),
        ]
    if defaults:
        entries.extend(defaults)
    # Preserve first-match semantics even when a special rule has the same
    # regex as a broad fallback (for example V4-all overriding the anchor's
    # V3.5 rule).  A dict comprehension would silently overwrite the special
    # value while retaining the original insertion position.
    dynamic_map = {}
    for pat, mod, rate in entries:
        key = "+:^model\\.layers\\.%s\\.(%s)$" % (pat, mod)
        if key in dynamic_map:
            continue
        dynamic_map[key] = {
            "bits": rate,
            **({"format": "qvq"} if rate > 3.5 else {}),
        }
    cfg["dynamic"] = dynamic_map
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
    wave3 = [
        # Current-code replication of the historical 2ae00f allocation.
        ("w3_control_qk25_vo35_g3_u35_d3", 2, []),
        # Exact-budget reallocations: O4 is funded by early Up precision.
        ("w3_o4_early_up3", 2, [
            ("[0-9]+", "self_attn.o_proj", 4),
            ("[0-3]", "mlp.up_proj", 3),
        ]),
        ("w3_o4_early_gate25", 2, [
            ("[0-9]+", "self_attn.o_proj", 4),
            ("[0-3]", "mlp.gate_proj", 2.5),
        ]),
        ("w3_o4_q2", 2, [
            ("[0-9]+", "self_attn.o_proj", 4),
            ("[0-9]+", "self_attn.q_proj", 2),
        ]),
        ("w3_flat35_up4_l12_15_seed1", 3.5, [
            ("(12|13|14|15)", "mlp.up_proj", 4),
        ]),
        ("w3_flat35_up4_l8_15", 3.5, [
            ("(8|9|10|11|12|13|14|15)", "mlp.up_proj", 4),
        ]),
        ("w3_flat35_up4_l7_15", 3.5, [
            ("(7|8|9|10|11|12|13|14|15)", "mlp.up_proj", 4),
        ]),
        ("w3_flat35_up4_l12_15_o4", 3.5, [
            ("(12|13|14|15)", "mlp.up_proj", 4),
            ("[0-9]+", "self_attn.o_proj", 4),
        ]),
    ]
    wave4_defaults = [
        ("[0-9]+", "self_attn.q_proj", 2),
        ("[0-9]+", "self_attn.k_proj", 2.5),
        ("[0-9]+", "self_attn.v_proj", 3.5),
        ("[0-9]+", "self_attn.o_proj", 4),
        ("[0-9]+", "mlp.gate_proj|mlp.down_proj", 3),
        ("[0-9]+", "mlp.up_proj", 3.5),
    ]
    wave4 = [
        ("w4_anchor_seed1", 2, [], wave4_defaults),
        ("w4_anchor_v4_all", 2, [("[0-9]+", "self_attn.v_proj", 4)], wave4_defaults),
        ("w4_anchor_up4_l6", 2, [("6", "mlp.up_proj", 4)], wave4_defaults),
        ("w4_anchor_up4_l6_7", 2, [("(6|7)", "mlp.up_proj", 4)], wave4_defaults),
        ("w4_anchor_v4_up4_l6", 2, [
            ("[0-9]+", "self_attn.v_proj", 4),
            ("6", "mlp.up_proj", 4),
        ], wave4_defaults),
        ("w4_anchor_up4_l6_8", 2, [("(6|7|8)", "mlp.up_proj", 4)], wave4_defaults),
        ("w4_anchor_up4_l6_9", 2, [("(6|7|8|9)", "mlp.up_proj", 4)], wave4_defaults),
        ("w4_anchor_v4_up4_l6_8", 2, [
            ("[0-9]+", "self_attn.v_proj", 4),
            ("(6|7|8)", "mlp.up_proj", 4),
        ], wave4_defaults),
    ]
    wave5_defaults = wave4_defaults
    wave5 = [
        ("w5_anchor_up4_l9", 2, [("9", "mlp.up_proj", 4)], wave5_defaults),
        ("w5_anchor_up4_l6_l9", 2, [("(6|9)", "mlp.up_proj", 4)], wave5_defaults),
        ("w5_anchor_up4_l7_l8", 2, [("(7|8)", "mlp.up_proj", 4)], wave5_defaults),
        ("w5_anchor_up4_l6_l8", 2, [("(6|8)", "mlp.up_proj", 4)], wave5_defaults),
        ("w5_anchor_up4_l6_l10", 2, [("(6|10)", "mlp.up_proj", 4)], wave5_defaults),
        ("w5_anchor_up4_l5_l9", 2, [("(5|9)", "mlp.up_proj", 4)], wave5_defaults),
        ("w5_anchor_up4_l6_l9_l12", 2, [("(6|9|12)", "mlp.up_proj", 4)], wave5_defaults),
        ("w5_anchor_up4_l6_l9_l12_l15", 2, [("(6|9|12|15)", "mlp.up_proj", 4)], wave5_defaults),
    ]
    wave6_defaults = wave4_defaults
    wave6 = [
        ("w6_anchor_up4_l8", 2, [("8", "mlp.up_proj", 4)], wave6_defaults),
        ("w6_anchor_up4_l5_l8", 2, [("(5|8)", "mlp.up_proj", 4)], wave6_defaults),
        ("w6_anchor_up4_l8_l9", 2, [("(8|9)", "mlp.up_proj", 4)], wave6_defaults),
        ("w6_anchor_up4_l8_l10", 2, [("(8|10)", "mlp.up_proj", 4)], wave6_defaults),
        ("w6_anchor_up4_l8_l12", 2, [("(8|12)", "mlp.up_proj", 4)], wave6_defaults),
        ("w6_anchor_up4_l8_l15", 2, [("(8|15)", "mlp.up_proj", 4)], wave6_defaults),
        ("w6_anchor_up4_l6_l8_l9", 2, [("(6|8|9)", "mlp.up_proj", 4)], wave6_defaults),
        ("w6_anchor_up4_l6_l8_l12", 2, [("(6|8|12)", "mlp.up_proj", 4)], wave6_defaults),
    ]
    # Full single-layer Up4 sensitivity sweep from the corrected W3.2 anchor.
    # Each arm changes exactly one Up projection (W3.5 -> W4), so all arms
    # have the same effective payload budget and are directly comparable.
    wave7_defaults = wave4_defaults
    wave7 = [
        (f"w7_anchor_up4_l{layer}", 2, [(str(layer), "mlp.up_proj", 4)], wave7_defaults)
        for layer in range(16)
    ]
    # Pair-interaction mapping from the corrected W3.2 anchor.  Every arm
    # promotes exactly two Up projections from W3.5 to W4 so the payload budget
    # is matched at 3.178340 BPW; repeats are deliberate positive/negative
    # interaction controls.
    wave8_defaults = wave4_defaults
    wave8_pairs = [
        ("l5_l12", "5|12"), ("l5_l15", "5|15"), ("l12_l15", "12|15"),
        ("l9_l12", "9|12"), ("l9_l15", "9|15"), ("l6_l8", "6|8"),
        ("l8_l12", "8|12"), ("l5_l6", "5|6"), ("l6_l12", "6|12"),
        ("l6_l15", "6|15"), ("l5_l7", "5|7"), ("l7_l12", "7|12"),
        ("l7_l15", "7|15"), ("l4_l5", "4|5"), ("l4_l12", "4|12"),
        ("l4_l8", "4|8"),
    ]
    wave8 = [
        (f"w8_anchor_up4_{suffix}", 2, [(f"({layers})", "mlp.up_proj", 4)], wave8_defaults)
        for suffix, layers in wave8_pairs
    ]
    # Local marginal sweep around the two strongest Wave-8 roots.  The first
    # four arms add one Up4 or Down3.5 promotion (+0.008621 BPW); the final
    # four test the corresponding two-module Down upgrades (+0.017241 BPW).
    wave9_defaults = wave4_defaults
    wave9 = [
        ("w9_anchor_up4_l7_l8_l12", 2, [("(7|8|12)", "mlp.up_proj", 4)], wave9_defaults),
        ("w9_anchor_up4_l6_l7_l8", 2, [("(6|7|8)", "mlp.up_proj", 4)], wave9_defaults),
        ("w9_anchor_up4_l8_l12_down35_l12", 2, [
            ("(8|12)", "mlp.up_proj", 4), ("12", "mlp.down_proj", 3.5)
        ], wave9_defaults),
        ("w9_anchor_up4_l6_l8_down35_l8", 2, [
            ("(6|8)", "mlp.up_proj", 4), ("8", "mlp.down_proj", 3.5)
        ], wave9_defaults),
        ("w9_anchor_up4_l8_l11_l12", 2, [("(8|11|12)", "mlp.up_proj", 4)], wave9_defaults),
        ("w9_anchor_up4_l6_l8_l11", 2, [("(6|8|11)", "mlp.up_proj", 4)], wave9_defaults),
        ("w9_anchor_up4_l8_l12_down35_l8_l12", 2, [
            ("(8|12)", "mlp.up_proj", 4), ("(8|12)", "mlp.down_proj", 3.5)
        ], wave9_defaults),
        ("w9_anchor_up4_l6_l8_down35_l6_l8", 2, [
            ("(6|8)", "mlp.up_proj", 4), ("(6|8)", "mlp.down_proj", 3.5)
        ], wave9_defaults),
    ]
    # Dynamic patterns are converted below to the repository's full module regex form.
    for wave, arms in ((1, wave1), (2, wave2), (3, wave3), (4, wave4), (5, wave5), (6, wave6), (7, wave7), (8, wave8), (9, wave9)):
        manifest = []
        seen = {}
        for spec in arms:
            name, bits, entries = spec[:3]
            defaults = spec[3] if len(spec) > 3 else None
            dynamic = []
            for layer_pat, module, rate in entries:
                dynamic.append((layer_pat, module, rate))
            seed = 1 if name.endswith("seed1") else 0
            rel = emit(name, bits, dynamic, seed=seed, defaults=defaults)
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
