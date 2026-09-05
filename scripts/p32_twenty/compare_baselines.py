import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/f6-snapshot-pr")
import torch

from scripts.p32_twenty.scorecard import logits_metrics

root = Path("/root/p32-model-baseline")
report = {
    "scope": "16 C4 documents; per-token KL and top-k overlap, not downstream tasks",
    "rows": [],
}
for i in range(16):
    teacher = torch.load(
        root / "canonical" / f"logits-{i:03d}.pt", weights_only=True
    ).squeeze(0)
    window = torch.load(
        root / "window" / f"logits-{i:03d}.pt", weights_only=True
    ).squeeze(0)
    row = {"row": i, "canonical_window_logits_equal": torch.equal(teacher, window)}
    del window
    bf16 = torch.load(root / "bf16" / f"logits-{i:03d}.pt", weights_only=True).squeeze(
        0
    )
    row["bf16_vs_p32_teacher"] = logits_metrics(bf16, teacher)
    report["rows"].append(row)
    print(
        i,
        row["canonical_window_logits_equal"],
        row["bf16_vs_p32_teacher"]["kl_teacher_candidate"],
        flush=True,
    )
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
report["complete"] = True
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
