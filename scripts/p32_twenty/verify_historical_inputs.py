"""Fail closed unless the historical F6 artifacts and seed-only config match."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = Path("/root/work/qvq-f6-historical")
REVISION = "3ebcf9a307231187178e29ae7ad91e156a84d6ef"
BINDINGS = {
    "/monster/data/model/dataset/nm-calibration/llm.parquet": "26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef",
    "/root/qvq-data/calibration-fisher-scaling-v2/yaqa182_nm10000.parquet": "5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39",
    "/root/QvQ/dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet": "2140541facb66112428212b3a36d51a7735393b28c79db59c2429f6e51ed57ef",
    str(
        ROOT / "dataset/calibration-fisher-scaling-v2/yaqa182_nm10000.disjointness.json"
    ): "f283eca649cbf1d2dcc160c202bc131c2462d4b3485b34c0d4bb5237d85a9d4e",
    "/monster/data/model/Llama-3.2-1B-Instruct/model.safetensors": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
}


def main():
    revision = subprocess.check_output(
        ["git", "-C", str(HISTORICAL), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise SystemExit("Historical quantizer revision mismatch")
    base = json.loads(
        (
            HISTORICAL
            / "scripts/configs/llama32_1b_fisher_scaling_yaqa182_nm10000.json"
        ).read_text()
    )
    base["yaqa"]["seed"] = 7
    candidate = json.loads((ROOT / "scripts/p32_twenty/f6_seed7.json").read_text())
    if candidate != base:
        raise SystemExit("F6 config differs in more than YAQA seed=7")
    failures = []
    for name, expected in BINDINGS.items():
        path = Path(name)
        if not path.is_file():
            failures.append("MISSING: " + name)
            continue
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            failures.append("HASH MISMATCH: " + name)
        else:
            print("VERIFIED:", name, actual, flush=True)
    if failures:
        raise SystemExit("\n".join(failures))
    print("Historical F6 inputs verified; sole config change is YAQA seed=7.")


if __name__ == "__main__":
    main()
