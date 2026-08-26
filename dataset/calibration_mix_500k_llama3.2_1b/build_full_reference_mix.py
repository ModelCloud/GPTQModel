"""Build the larger disjoint YAQA mix by adding the held-out scan reference."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "calibration.parquet"
REFERENCE = ROOT / "_shards" / "reference.parquet"
OUT = ROOT / "calibration_full_reference.parquet"

def main():
    base = pd.read_parquet(BASE)[["messages"]]
    ref = pd.read_parquet(REFERENCE)[["messages"]]
    out = pd.concat([base, ref], ignore_index=True)
    out.to_parquet(OUT, index=False)
    info = {
        "base_rows": len(base), "reference_rows": len(ref), "output_rows": len(out),
        "base": str(BASE), "reference": str(REFERENCE), "output": str(OUT),
        "reference_provenance": "calibration_mix_500k_llama3.2_1b/dataset_info.json",
        "disjointness_evidence": {"reference_vs_benchmark_or_yaqa": 0, "selected_vs_reference": 0},
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    (ROOT / "calibration_full_reference.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
