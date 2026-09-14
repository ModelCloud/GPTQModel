"""Check prior GPTQ/GEMM GSQ behavior against the pre-GEMV implementation."""

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization import AWQConfig, GSQConfig
from gptqmodel.quantization.gsq_scalar import refine_affine_scalar


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an earlier audit")
    reference = "22d98963a2bf83cb4f3e2d3047b1a9706bb60cbd"
    source = subprocess.check_output(["git", "show", reference+":gptqmodel/quantization/gsq_scalar.py"], text=True)
    module = types.ModuleType("gptqmodel.quantization._gsq_pre_gemv")
    module.__package__ = "gptqmodel.quantization"
    sys.modules[module.__name__] = module
    exec(compile(source, reference+":gsq_scalar.py", "exec"), module.__dict__)
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    target = data["weight"][:8, :64].contiguous()
    inputs = torch.cat(data["train"])[:128, :64].contiguous()
    rows = []
    for packing in ("gptq", "awq_gemm"):
        for dtype in (torch.float16, torch.bfloat16):
            weight = target.to(dtype)
            cfg = AWQConfig(bits=4, group_size=32, sym=False)
            baseline, scales, zeros = AWQProcessor.pseudo_quantize_tensor(types.SimpleNamespace(qcfg=cfg), weight)
            if packing == "gptq":
                scales, zeros = scales.float(), zeros.float()
            groups = torch.arange(64, dtype=torch.int32)//32
            for candidates in (3, 33):
                for learn in (False, True):
                    kw = dict(target=weight.float(), bits=4, inputs=inputs, packing=packing,
                              config=GSQConfig(enabled=True, steps=100, candidates=candidates, seed=7, learn_scales=learn),
                              scale_dtype=dtype if packing == "awq_gemm" else torch.float16)
                    old_error = None
                    try:
                        old = module.refine_affine_scalar(baseline, scales, zeros, groups, **kw)
                    except ValueError as exc:
                        if str(exc) != "non-finite scalar GSQ relaxed objective":
                            raise
                        old_error = str(exc)
                    new = refine_affine_scalar(baseline, scales, zeros, groups, **kw)
                    exact = old_error is None and old.history == new.history and old.before == new.before and old.after == new.after
                    if packing == "awq_gemm" and candidates == 3:
                        if not torch.isfinite(torch.tensor(new.history)).all() or new.after > new.before:
                            raise ValueError("Local-grid FP32 optimizer failed its finite baseline guard")
                        rows.append(dict(packing=packing, dtype=str(dtype), candidates=candidates,
                                         learn_scales=learn, exact=exact, old_error=old_error,
                                         status="FP32 optimizer correction", before=new.before, after=new.after))
                        continue
                    if not exact:
                        raise ValueError("Changed fitting history")
                    if any(not torch.equal(getattr(old, field), getattr(new, field)) for field in ("weight", "scales", "zeros", "g_idx")):
                        raise ValueError("Changed fitted tensors")
                    rows.append(dict(packing=packing, dtype=str(dtype), candidates=candidates,
                                     learn_scales=learn, exact=True, before=new.before, after=new.after))
    result = {"reference_commit": reference, "reference_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "inputs": str(args.inputs), "inputs_sha256": sha(args.inputs),
              "current_code": {str(p): sha(p) for p in (Path(__file__), Path("gptqmodel/quantization/gsq_scalar.py"))},
              "scope": "8x64 real block-0 K slice; first 128 real calibration rows; identity regression only, not model quality",
              "seed": 7, "steps": 100, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Checked {len(rows)} cases: {sum(r['exact'] for r in rows)} exact histories; "
          "local AWQ grids separately checked for finite baseline-guarded fitting")


if __name__ == "__main__":
    main()
