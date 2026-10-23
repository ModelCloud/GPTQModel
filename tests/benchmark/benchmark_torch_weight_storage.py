"""Compare uncached TorchLinear with a historical revision and its prohibited dense cache."""

import argparse
import json
import linecache
import subprocess
import sys
import types
from functools import partial
from pathlib import Path

import torch
from benchmark_cpu_method_kernels import _bench, _relative_error

from tests.test_torch_weight_storage import build_case, state_tensors


def historical_class(revision):
    path = "gptqmodel/nn_modules/qlinear/torch.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    filename = f"<TorchLinear baseline {revision}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    name = "gptqmodel.nn_modules.qlinear._storage_baseline"
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "gptqmodel.nn_modules.qlinear"
    sys.modules[name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module.TorchLinear


def storage_bytes(module, excluded):
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for tensor in state_tensors(module)}
    return sum(size for pointer, size in storages.items() if pointer not in excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="1b5a8ca4")
    parser.add_argument("--shapes", default="512x512,4096x4096")
    parser.add_argument("--tokens", default="1,7,25,128")
    parser.add_argument("--bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    baseline_class = historical_class(args.baseline_ref)
    rows = []
    with torch.inference_mode():
        for shape in args.shapes.split(","):
            in_features, out_features = map(int, shape.split("x"))
            for dtype in (torch.float16, torch.bfloat16):
                current, weight = build_case(args.bits, dtype, args.group_size,
                                             in_features=in_features, out_features=out_features)
                baseline = baseline_class(bits=args.bits, group_size=args.group_size, sym=False, desc_act=True,
                                          in_features=in_features, out_features=out_features, bias=False,
                                          format=current.format)
                baseline._init_wf_unsqueeze_buffers()
                baseline.load_state_dict(current.state_dict())
                baseline.eval().to(dtype).enable_weight_cache(False).enable_lookahead(False)
                if args.compiled:
                    current.optimize()
                    baseline.optimize()
                current_storage = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(current)}
                baseline_storage = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(baseline)}
                for tokens in map(int, args.tokens.split(",")):
                    activation = torch.arange(tokens * in_features).reshape(tokens, -1).remainder(9).sub(4).to(dtype) / 16
                    raw = activation.float() @ weight.to(dtype).float()
                    ideal = activation.float() @ weight
                    actual = current(activation)
                    baseline.enable_weight_cache(False)
                    torch.testing.assert_close(actual, baseline(activation), rtol=0, atol=0)
                    torch.testing.assert_close(actual, raw.to(dtype), rtol=0, atol=0)
                    baseline_ms, _ = _bench(partial(baseline, activation), warmup=args.warmup, iters=args.iters)
                    current_ms, _ = _bench(partial(current, activation), warmup=args.warmup, iters=args.iters)
                    baseline.enable_weight_cache(True)
                    torch.testing.assert_close(actual, baseline(activation), rtol=0, atol=0)
                    cached_ms, _ = _bench(partial(baseline, activation), warmup=args.warmup, iters=args.iters)
                    rows.append({
                        "shape": shape, "tokens": tokens, "dtype": str(dtype), "compiled": args.compiled,
                        "baseline_uncached_ms": baseline_ms, "current_ms": current_ms, "historical_dense_ms": cached_ms,
                        "current_retained_cache_bytes": storage_bytes(current, current_storage),
                        "historical_retained_cache_bytes": storage_bytes(baseline, baseline_storage),
                        "transient_dense_operand_bytes": in_features * out_features * actual.element_size(),
                        "visible_max_abs": (actual.float() - raw.to(dtype).float()).abs().max().item(),
                        "raw_max_abs": (actual.float() - raw).abs().max().item(),
                        "raw_relative": _relative_error(actual, raw),
                        "ideal_max_abs": (actual.float() - ideal).abs().max().item(),
                        "ideal_relative": _relative_error(actual, ideal),
                    })
                    print(json.dumps(rows[-1]), flush=True)
                del current, baseline, weight
    result = {"torch": torch.__version__, "threads": args.threads, "baseline": args.baseline_ref, "rows": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
