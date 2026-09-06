"""Summarize experiment 36 without treating missing or partial fits as passes."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/p32-stability"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "scope": "Experiment 36 local fit summary; broader/model acceptance is separate",
        "expected_fits": 15,
        "completed_fits": 0,
        "fits": [],
    }
    for seed in (71, 72, 73):
        for tokens in (2048, 4096, 8192, 16384, 32768):
            original = args.root / f"seed{seed}/fit{tokens}"
            retries = sorted(
                (
                    p
                    for p in original.parent.glob(original.name + "-retry*")
                    if p.name.rsplit("-retry", 1)[-1].isdigit()
                ),
                key=lambda p: int(p.name.rsplit("-retry", 1)[-1]),
            )
            attempts = []
            for directory in [original, *retries]:
                report_path = directory / "report.json"
                entry = {"path": str(report_path), "status": "missing"}
                if report_path.exists():
                    attempt_raw = report_path.read_bytes()
                    attempt = json.loads(attempt_raw)
                    entry.update(
                        status="complete" if attempt.get("complete") else "partial",
                        sha256=hashlib.sha256(attempt_raw).hexdigest(),
                    )
                attempts.append(entry)
            # The newest started attempt is authoritative, even while partial.
            path = (retries[-1] if retries else original) / "report.json"
            item = {
                "sampling_seed": seed,
                "tokens": tokens,
                "path": str(path),
                "attempts": attempts,
            }
            result["fits"].append(item)
            if not path.exists():
                item["status"] = "missing"
                continue
            raw = path.read_bytes()
            report = json.loads(raw)
            item["sha256"] = hashlib.sha256(raw).hexdigest()
            if not report.get("complete"):
                item["status"] = "partial"
                continue
            if not (
                report.get("source_export_unchanged")
                and report.get("teacher_read_shards_unchanged")
            ):
                raise ValueError(f"Read-only source verification failed: {path}")
            candidates = report["candidates"]
            expected = {
                (r, f, d)
                for r in (2, 4, 6, 8, 12, 16)
                for f in ("l2", "tail")
                for d in ("float32", "float16")
            }
            actual = {(c["rank"], c["fit"], c["factor_dtype"]) for c in candidates}
            if actual != expected or len(candidates) != len(expected):
                raise ValueError(f"Incomplete candidate grid: {path}")
            item.update(status="complete", candidates=[])
            result["completed_fits"] += 1
            for candidate in candidates:
                rows = candidate["rows"]
                if sorted(x["M"] for x in rows) != [1, 2, 4, 8, 16, 32, 128, 512, 2048]:
                    raise ValueError(f"Incomplete row grid: {path}")
                passes = [
                    x["reload_equal"]
                    and x["teacher_metrics"]["local_tolerance_pass"]
                    and x["window_metrics"]["local_tolerance_pass"]
                    for x in rows
                ]
                item["candidates"].append(
                    {
                        "rank": candidate["rank"],
                        "fit": candidate["fit"],
                        "factor_dtype": candidate["factor_dtype"],
                        "local_passes": sum(passes),
                        "reload_equal": all(x["reload_equal"] for x in rows),
                        "teacher_max": max(
                            x["teacher_metrics"]["max_abs"] for x in rows
                        ),
                        "window_max": max(x["window_metrics"]["max_abs"] for x in rows),
                        "bpw": candidate["serialized_operator_bpw"],
                        "exception_review_cases": [
                            {
                                "M": x["M"],
                                "speedup_vs_window": x["speedup_vs_window"],
                                "teacher_metrics": x["teacher_metrics"],
                                "window_metrics": x["window_metrics"],
                            }
                            for x, passed in zip(rows, passes)
                            if not passed and x["speedup_vs_window"] > 1.25
                        ],
                    }
                )
    result["all_local_fits_complete"] = (
        result["completed_fits"] == result["expected_fits"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Completed local fits: {result['completed_fits']}/{result['expected_fits']}")


if __name__ == "__main__":
    main()
