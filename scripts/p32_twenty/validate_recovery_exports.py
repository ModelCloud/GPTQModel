"""GPU export reload and identical-input FP16-boundary checks."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    for _ in range(3):
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + args.uuid,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).split(",")
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if int(row[0]) > 8 or int(row[1]) or args.uuid in procs:
            raise RuntimeError("GPU not idle")
        time.sleep(1)
    import torch
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from scripts.p32_twenty.native_layer_recovery import SNAPSHOT
    from scripts.p32_twenty.recovered_linear import RecoveredLinear
    from scripts.p32_twenty.scorecard import layer_metrics

    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path("/root/p32-native-recovery") / args.name
    report = json.loads((root / "report.json").read_text())
    module = report["module"]
    data = torch.load(
        Path("/root/p32-timing-activations") / (module + ".pt"), weights_only=True
    )
    x = data["input"].cuda().float().reshape(-1, data["input"].shape[-1])
    y = (
        data["teacher_output"]
        .cuda()
        .float()
        .reshape(-1, data["teacher_output"].shape[-1])
    )
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(s):
        key = module + "." + s
        with safe_open(str(SNAPSHOT / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key).cuda()

    t, su, sv, bank, alt = [
        read(k) for k in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]
    ]
    inner = reconstruct_qvq_inner_weight(
        t,
        bits=t.shape[-1] / 8,
        in_features=su.numel(),
        out_features=sv.numel(),
        bank_ids=bank,
        bank_alt_id=alt,
        v2b2_p32=True,
    ).float()
    result = {
        "scope": "trusted export reload, original FP32 measurements and identical FP16 inputs versus FP32 canonical operator",
        "module": module,
        "uuid": args.uuid,
        "rows": [],
    }
    with torch.no_grad():
        for rank in [16, 32, 64, 128]:
            candidate = RecoveredLinear(root / f"rank{rank}.pt")
            for row in report["rows"]:
                m = row["M"]
                actual = layer_metrics(candidate(x[:m]), y[:m])
                expected = next(r for r in row["ranks"] if r["requested_rank"] == rank)[
                    "metrics"
                ]
                drift = max(
                    abs(actual[k] - expected[k]) for k in ["mean_abs", "max_abs"]
                )
                if drift > 1e-7:
                    raise AssertionError(
                        ("Reload changed measured errors", rank, m, drift)
                    )
                xh = x[:m].half()
                teacher = (
                    matmul_hadU(matmul_hadU(xh.float() * su.float()) @ inner)
                    * sv.float()
                )
                result["rows"].append(
                    {
                        "rank": rank,
                        "M": m,
                        "reload_metric_drift": drift,
                        "fp16_boundary_metrics": layer_metrics(candidate(xh), teacher),
                        "teacher_fp16_rounding_metrics": layer_metrics(teacher.half(), teacher),
                    }
                )
    result["complete"] = True
    (root / "reload-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.name, "complete", flush=True)


if __name__ == "__main__":
    main()
