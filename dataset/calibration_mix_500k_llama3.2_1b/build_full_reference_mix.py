"""Build the larger disjoint YAQA mix by adding the held-out scan reference."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "calibration.parquet"
REFERENCE = ROOT / "_shards" / "reference.parquet"
OUT = ROOT / "calibration_full_reference.parquet"
DATASET_INFO = ROOT / "dataset_info.json"


def _series_for(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series([None] * len(data), index=data.index, dtype="object")


def main():
    base = pd.read_parquet(BASE)
    ref = pd.read_parquet(REFERENCE)

    base_source = _series_for(base, "shard")
    if base_source.isna().all():
        base_source = _series_for(base, "source")
    ref_source = _series_for(ref, "source")
    if ref_source.isna().all():
        ref_source = _series_for(ref, "shard")

    out = pd.concat(
        [
            pd.DataFrame({
                "messages": base["messages"].reset_index(drop=True),
                "source": base_source.reset_index(drop=True),
                "split": ["base"] * len(base),
            }),
            pd.DataFrame({
                "messages": ref["messages"].reset_index(drop=True),
                "source": ref_source.reset_index(drop=True),
                "split": ["reference"] * len(ref),
            }),
        ],
        ignore_index=True,
    )
    out.to_parquet(OUT, index=False)

    dataset_info = json.loads(DATASET_INFO.read_text(encoding="utf-8")) if DATASET_INFO.is_file() else {}
    content_hash_intersections = dataset_info.get("content_hash_intersections", {})
    disjointness_evidence = {
        "selected_vs_reference": content_hash_intersections.get(
            "selected_vs_reference",
            content_hash_intersections.get("candidate_vs_reference"),
        ),
        "reference_vs_benchmark_or_yaqa": content_hash_intersections.get("reference_vs_benchmark_or_yaqa"),
    }

    info = {
        "base": str(BASE),
        "reference": str(REFERENCE),
        "output": str(OUT),
        "base_rows": len(base),
        "reference_rows": len(ref),
        "output_rows": len(out),
        "reference_provenance": str(DATASET_INFO),
        "dataset_info": dataset_info,
        "disjointness_evidence": disjointness_evidence,
        "source_counts": out["source"].value_counts().to_dict(),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    (ROOT / "calibration_full_reference.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
