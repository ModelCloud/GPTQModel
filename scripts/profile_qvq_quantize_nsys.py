#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Run the real QVQ quantization harness (``scripts/qvq_quantize.py``) with NVTX attribution.

This script is meant to run *under* ``nsys profile`` (see ``scripts/profile_qvq_quantize_nsys.sh``).
It does not modify any kernel or quantization code.  Before delegating to ``qvq_quantize.main()`` it
monkeypatches, at the Python module boundary only:

* every ``gptqmodel.utils.qvq_cuda._qvq_cuda_*_op`` resolver so the returned ``torch.ops`` callable is
  wrapped in an NVTX range named ``qvq_cuda.<op>`` (one range per kernel-variant launch site; the
  variants are the ``required_ops`` tuple in ``qvq_cuda.py``),
* the public ``qvq_cuda_gemv`` / ``qvq_cuda_hadamard`` / ``qvq_cuda_viterbi*`` wrappers
  (``qvq_api.<name>``) so codec/GEMV paths are attributable even when the op is reached through them,
* the ``gptqmodel.utils.qvq_cpu`` viterbi/hadamard/yaqa resolvers (``qvq_cpu.<op>``) so CPU fallbacks
  show up as host-side NVTX ranges with call counts,
* ``QVQProcessor.prepare_yaqa`` / ``preprocess`` / ``process`` / ``submodule_finalize`` / ``finalize``
  (``stage.<stage>:<module full name>``) for per-layer / per-module stage ranges.

Independently of nsys, every wrapped call is also counted and wall-timed on the host (without forcing
CUDA synchronisation) and the summary is written to ``--attribution-json`` so the report can cross-check
call counts against ``nsys stats``.

All unknown CLI arguments are forwarded verbatim to ``scripts/qvq_quantize.py``; ``--max-layers N`` is an
alias for its ``--layers N``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

_nvtx = torch.cuda.nvtx


class _Stats:
    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.host_seconds: dict[str, float] = defaultdict(float)
        self.max_seconds: dict[str, float] = defaultdict(float)

    def record(self, name: str, seconds: float) -> None:
        self.calls[name] += 1
        self.host_seconds[name] += seconds
        if seconds > self.max_seconds[name]:
            self.max_seconds[name] = seconds

    def to_dict(self) -> dict:
        return {
            name: {
                "calls": self.calls[name],
                "host_seconds_total": self.host_seconds[name],
                "host_seconds_avg": self.host_seconds[name] / self.calls[name],
                "host_seconds_max": self.max_seconds[name],
            }
            for name in sorted(self.calls, key=lambda n: -self.host_seconds[n])
        }


STATS = _Stats()


def _ranged(name: str, fn):
    """Wrap ``fn`` in an NVTX range ``name`` and host-side timing; does not sync the device."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _nvtx.range_push(name)
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            STATS.record(name, time.perf_counter() - start)
            _nvtx.range_pop()

    wrapper.__qvq_profile_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_resolver(module, attr: str, prefix: str) -> None:
    """Patch ``module.attr`` (an op *resolver*) so the op it returns is NVTX ranged.

    The resolvers (``_qvq_cuda_viterbi_v4_op`` ...) are called at launch time through function-local
    imports in ``gptqmodel/quantization/qvq.py``, so patching the module attribute is sufficient and
    no kernel/quantization source is touched.
    """

    resolver = getattr(module, attr)
    op_name = attr[len("_qvq_cuda_") if attr.startswith("_qvq_cuda_") else len("_qvq_cpu_") : -len("_op")]
    if op_name == "":  # `_qvq_cuda_op` is the gemv op resolver
        op_name = "gemv"
    range_name = f"{prefix}.{op_name}"
    cache: dict[int, object] = {}

    @functools.wraps(resolver)
    def patched():
        op = resolver()
        key = id(op)
        wrapped = cache.get(key)
        if wrapped is None:
            wrapped = _ranged(range_name, op)
            cache[key] = wrapped
        return wrapped

    setattr(module, attr, patched)


def install_instrumentation() -> list[str]:
    from gptqmodel.looper import qvq_processor as qvq_processor_module
    from gptqmodel.utils import qvq_cpu, qvq_cuda

    installed: list[str] = []
    for attr in dir(qvq_cuda):
        if attr.startswith("_qvq_cuda_") and attr.endswith("_op") and callable(getattr(qvq_cuda, attr)):
            _wrap_resolver(qvq_cuda, attr, "qvq_cuda")
            installed.append(f"qvq_cuda.{attr}")
    for attr in dir(qvq_cpu):
        if attr.startswith("_qvq_cpu_") and attr.endswith("_op") and callable(getattr(qvq_cpu, attr)):
            _wrap_resolver(qvq_cpu, attr, "qvq_cpu")
            installed.append(f"qvq_cpu.{attr}")
    for attr in (
        "qvq_cuda_gemv",
        "qvq_cuda_hadamard",
        "qvq_cuda_viterbi",
        "qvq_cuda_viterbi_banked",
        "qvq_cuda_viterbi_v2_segment_banked",
        "_qvq_cuda_viterbi_trusted",
    ):
        if hasattr(qvq_cuda, attr):
            setattr(qvq_cuda, attr, _ranged(f"qvq_api.{attr}", getattr(qvq_cuda, attr)))
            installed.append(f"qvq_cuda.{attr}")
    for attr in ("qvq_cpu_viterbi", "qvq_cpu_viterbi_banked"):
        if hasattr(qvq_cpu, attr):
            setattr(qvq_cpu, attr, _ranged(f"qvq_api.{attr}", getattr(qvq_cpu, attr)))
            installed.append(f"qvq_cpu.{attr}")

    processor_cls = qvq_processor_module.QVQProcessor

    def _stage(method_name: str, label_from_args):
        original = getattr(processor_cls, method_name)

        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            label = label_from_args(args, kwargs)
            name = f"stage.{method_name}:{label}" if label else f"stage.{method_name}"
            _nvtx.range_push(name)
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                STATS.record(f"stage.{method_name}", time.perf_counter() - start)
                _nvtx.range_pop()

        setattr(processor_cls, method_name, wrapper)
        installed.append(f"QVQProcessor.{method_name}")

    def _module_label(args, kwargs):
        module = args[0] if args else kwargs.get("module")
        return getattr(module, "full_name", None) or getattr(module, "name", "")

    _stage("prepare_yaqa", lambda a, k: "")
    _stage("prepare_module_granular_replay", lambda a, k: "")
    _stage("preprocess", _module_label)
    _stage("process", _module_label)
    _stage("submodule_finalize", _module_label)
    _stage("finalize", lambda a, k: "")
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-layers", type=int, default=None, help="Alias for qvq_quantize.py --layers N.")
    parser.add_argument(
        "--attribution-json",
        type=Path,
        default=None,
        help="Where to write host-side call counts/timings (default: <output>/qvq_nsys_host_attribution.json).",
    )
    args, forwarded = parser.parse_known_args()
    if args.max_layers is not None:
        forwarded += ["--layers", str(args.max_layers)]

    installed = install_instrumentation()
    print(f"[profile_qvq_quantize_nsys] instrumented {len(installed)} call sites", flush=True)

    import importlib.util

    spec = importlib.util.spec_from_file_location("qvq_quantize", REPO_ROOT / "scripts" / "qvq_quantize.py")
    qvq_quantize = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["qvq_quantize"] = qvq_quantize  # dataclass() resolves annotations via sys.modules
    spec.loader.exec_module(qvq_quantize)

    sys.argv = [str(REPO_ROOT / "scripts" / "qvq_quantize.py"), *forwarded]
    output_dir = None
    if "--output" in forwarded:
        output_dir = Path(forwarded[forwarded.index("--output") + 1])

    _nvtx.range_push("qvq_quantize.main")
    wall_start = time.perf_counter()
    exit_code = 0
    try:
        result = qvq_quantize.main()
        if isinstance(result, int):
            exit_code = result
    except SystemExit as exc:  # argparse / harness exits
        exit_code = int(exc.code or 0)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - wall_start
        _nvtx.range_pop()

    payload = {
        "wall_seconds_qvq_quantize_main": wall,
        "instrumented": installed,
        "forwarded_args": forwarded,
        "ranges": STATS.to_dict(),
        "pid": os.getpid(),
    }
    target = args.attribution_json
    if target is None and output_dir is not None:
        target = output_dir / "qvq_nsys_host_attribution.json"
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2))
        print(f"[profile_qvq_quantize_nsys] host attribution -> {target}", flush=True)
    print(f"[profile_qvq_quantize_nsys] qvq_quantize.main wall = {wall:.1f}s exit={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
