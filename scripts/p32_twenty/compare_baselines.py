"""Compare saved logits without running or modifying the checkpoint."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from scripts.p32_twenty.scorecard import logits_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/p32-model-baseline"))
    parser.add_argument("--candidate", default="production")
    parser.add_argument("--teacher", default="canonical")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / (args.candidate + "-comparison.json")
    if output.exists():
        parser.error("Refusing to overwrite an existing comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher_root = args.root / args.teacher
    candidate_root = args.root / args.candidate
    files = sorted(teacher_root.glob("logits-*.pt"))
    if not files:
        raise ValueError("No teacher logits")
    for root in (teacher_root, candidate_root):
        if not json.loads((root / "report.json").read_text()).get("complete"):
            raise ValueError("Baseline run is not complete: " + str(root))
    report = {
        "scope": "bounded C4 per-token logits; no downstream-task claim",
        "candidate": args.candidate,
        "teacher": args.teacher,
        "rows": [],
    }
    for path in files:
        teacher = torch.load(path, weights_only=True).squeeze(0)
        candidate = torch.load(candidate_root / path.name, weights_only=True).squeeze(0)
        metrics = logits_metrics(candidate, teacher)
        report["rows"].append(
            {
                "file": path.name,
                "equal_logits": torch.equal(candidate, teacher),
                "metrics": metrics,
            }
        )
        print(path.name, metrics["kl_teacher_candidate"], flush=True)
        output.write_text(
            json.dumps(report, indent=2) + "\n"
        )
    report["complete"] = True
    count = sum(r["metrics"]["tokens"] for r in report["rows"])
    report["aggregate"] = {
        key: sum(r["metrics"][key] * r["metrics"]["tokens"] for r in report["rows"])
        / count
        for key in (
            "kl_teacher_candidate",
            "top1_agreement",
            "top5_agreement",
            "top10_agreement",
        )
    }
    output.write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
