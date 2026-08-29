"""Generate the eight matched-BPW high-rate QVQ experiment configs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "configs"
COMMON = {
    "device": "cuda:0", "offload_to_disk": False,
    "yaqa": {"seed": 0, "regularization": 0.15,
             "regularization_by_rate": [[2.0, 0.15], [2.5, 0.02], [3.0, 0.02], [3.5, 0.02], [4.0, 0.02], [4.5, 0.02], [5.0, 0.02], [5.5, 0.02], [6.0, 0.02], [7.0, 0.02]],
             "minimum_sequences": 182, "batch_size": 1, "sequence_sort": "desc",
             "activation_checkpointing": True, "v2b2_family_mode": "reselect", "sample_strategy": "full"},
}
ARMS = [
    ("up4_l8_15", 2, {r"+:^model\.layers\.(8|9|10|11|12|13|14|15)\.mlp\.up_proj$": {"bits": 4, "format": "qvq"}}),
    ("up5_l12_15", 2, {r"+:^model\.layers\.(12|13|14|15)\.mlp\.up_proj$": {"bits": 5, "format": "qvq"}}),
    ("up7_l14_15", 2, {r"+:^model\.layers\.(14|15)\.mlp\.up_proj$": {"bits": 7, "format": "qvq"}}),
    ("up4_l12_13_up6_l14_15", 2, {r"+:^model\.layers\.(12|13)\.mlp\.up_proj$": {"bits": 4, "format": "qvq"}, r"+:^model\.layers\.(14|15)\.mlp\.up_proj$": {"bits": 6, "format": "qvq"}}),
    ("flat35_up4_l12_15", 3.5, {r"+:^model\.layers\.(12|13|14|15)\.mlp\.up_proj$": {"bits": 4, "format": "qvq"}}),
    ("flat35_up45_l14_15", 3.5, {r"+:^model\.layers\.(14|15)\.mlp\.up_proj$": {"bits": 4.5, "format": "qvq"}}),
    ("flat35_up55_l15", 3.5, {r"+:^model\.layers\.15\.mlp\.up_proj$": {"bits": 5.5, "format": "qvq"}}),
    ("flat35_up4_l14_up5_l15", 3.5, {r"+:^model\.layers\.14\.mlp\.up_proj$": {"bits": 4, "format": "qvq"}, r"+:^model\.layers\.15\.mlp\.up_proj$": {"bits": 5, "format": "qvq"}}),
]

for arm_id, bits, special in ARMS:
    dynamic = dict(special)
    if bits == 2:
        dynamic.update({r"+:.*\.self_attn\.(q_proj|k_proj)": {"bits": 2.5}, r"+:.*\.self_attn\.(v_proj|o_proj)": {"bits": 3.5}, r"+:.*\.mlp\.(gate_proj|up_proj|down_proj)": {"bits": 3.0}})
    payload = {"bits": bits, "format": "qvq_v2b2_p32" if bits == 2 else "qvq_v2b2_p32", "bank_count": 2, "rounding": "yaqa", "dynamic": dynamic, **COMMON}
    (ROOT / f"llama32_1b_highrate_{arm_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
