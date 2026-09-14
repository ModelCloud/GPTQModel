"""Reproduce Hessian-reference failures against the pre-GSQ GPTAQ class."""

import argparse
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh report path")
    sys.path.insert(0, str(Path("tests").resolve()))
    spec = importlib.util.spec_from_file_location("gptaq_regression_cases", "tests/test_gptaq.py")
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    current = cases.GPTAQ
    revision = "11993f31d8b84119c5d03e5dc9c820fced2a3f1f"
    source = subprocess.check_output(["git", "show", revision+":gptqmodel/quantization/gptaq.py"], text=True)
    previous = types.ModuleType("gptqmodel.quantization._gptaq_before_gsq")
    previous.__package__ = "gptqmodel.quantization"
    exec(compile(source, revision+":gptaq.py", "exec"), previous.__dict__)
    original_factory = cases._make_gptaq_for_random
    rows = []
    for seed in (1114, 2229, 2679, 4143, 4627, 6373, 6903, 7592, 8027, 9625, 9666):
        states, failed = [], []
        for cls in (previous.GPTAQ, current):
            captured = []

            def factory(*a, **kw):
                obj = original_factory(*a, **kw)
                captured.append(obj)
                return obj

            cases.GPTAQ = cls
            cases._make_gptaq_for_random = factory
            try:
                cases.test_gptaq_hessian_randomized(seed)
                failed.append(False)
            except AssertionError:
                failed.append(True)
            states.append((captured[0].H.clone(), captured[0].dXXT.clone()))
        exact = all(torch.equal(a, b) for a, b in zip(*states, strict=True))
        if failed != [True, True] or not exact:
            raise ValueError(f"Seed {seed} does not reproduce identically: {failed}, exact={exact}")
        rows.append({"seed": seed, "previous_failed": failed[0], "current_failed": failed[1],
                     "hessian_and_cross_exact": exact})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reference_commit": revision, "rows": rows,
        "scope": "CPU Hessian accumulation only; both classes use current shared GPTQ base; no quantization invoked"},
        indent=2)+"\n")
    print("All 11 existing reference failures reproduce with exact old/new H and cross moments")


if __name__ == "__main__":
    main()
