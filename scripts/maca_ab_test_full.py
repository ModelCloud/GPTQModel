#!/usr/bin/env python3
"""Full all-layer Llama-3.2-1B MaCa A/B test on GPU 3.

Based on tests/models/test_llama3_2.py. Runs six variants and reports
arc_challenge, gsm8k_platinum_cot, mmlu_stem, mmlu_history and
mmlu_chemistry scores.

  A: DATASET_CONCAT_SIZE=2048, length_aware=False
  B: DATASET_CONCAT_SIZE=0,    length_aware=False
  C: DATASET_CONCAT_SIZE=0,    length_aware=True (per-sequence)
  D: DATASET_CONCAT_SIZE=0,    length_aware=EqualPerBucketWeight(p=1.0)
  E: DATASET_CONCAT_SIZE=0,    length_aware=SINGLE
  F: DATASET_CONCAT_SIZE=0,    length_aware=EqualPerBucketWeight(p=0.4)
"""

# ruff: noqa: E402

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

# Pin to a GPU before any torch/transformers import, but allow the
# gpu_allocator to override CUDA_VISIBLE_DEVICES for parallel sweeps.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "models"))

import torch  # noqa: E402
from gptqmodel import BACKEND  # noqa: E402
from gptqmodel.quantization.config import (  # noqa: E402
    HessianConfig,
    LengthAwareConfig,
    LengthAwareMode,
)
from gptqmodel.utils.calibration import prepare_calibration_dataset  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from model_test import ModelTest  # noqa: E402
from tests.eval import evaluate as eval_run, get_eval_task_results  # noqa: E402

MMLU_HISTORY_SUBSETS = [
    "humanities.high_school_european_history",
    "humanities.high_school_us_history",
    "humanities.high_school_world_history",
    "humanities.prehistory",
]

MMLU_CHEMISTRY_SUBSETS = [
    "stem.college_chemistry",
    "stem.high_school_chemistry",
]

GPU_MEMORY_MB_ALLOWANCE = 512
PREFLIGHT_MIN_FREE_MB = 12_000
PREFLIGHT_MAX_UTIL_PCT = 5.0
PREFLIGHT_SAMPLES = 3
PREFLIGHT_INTERVAL_S = 1.0


def _requested_physical_gpu() -> int:
    """Parse the requested physical GPU index from CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if visible.startswith("GPU-"):
        # UUID target: look it up in the nvidia-smi inventory each poll.
        return -1
    try:
        return int(visible)
    except ValueError:
        return 0


def _gpu_preflight() -> None:
    """Require the requested GPU to be idle and have enough free memory."""
    target_label = f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    target_index = _requested_physical_gpu()
    target_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    print(
        f"Preflight: waiting for {PREFLIGHT_SAMPLES} consecutive idle samples "
        f"(util<={PREFLIGHT_MAX_UTIL_PCT}%, free>={PREFLIGHT_MIN_FREE_MB} MiB) "
        f"on {target_label} ..."
    )
    idle_count = 0
    last_info: Dict[str, Any] = {}
    while idle_count < PREFLIGHT_SAMPLES:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"GPU preflight failed: unable to query nvidia-smi: {exc}"
            ) from exc

        rows = [line.strip() for line in out.strip().splitlines() if line.strip()]
        if not rows:
            raise RuntimeError(
                f"GPU preflight failed: no GPUs visible for {target_label}"
            )

        selected = rows[0]
        for row in rows:
            parts = [p.strip() for p in row.split(",")]
            if target_index >= 0 and parts[0].isdigit() and int(parts[0]) == target_index:
                selected = row
                break
            if target_uuid.startswith("GPU-") and len(parts) > 2 and parts[2] == target_uuid:
                selected = row
                break

        parts = [p.strip() for p in selected.split(",")]
        if len(parts) != 8:
            raise RuntimeError(
                f"GPU preflight failed: unexpected nvidia-smi format: {selected!r}"
            )

        index, pci_bus_id, uuid, name, memory_total, memory_free, memory_used, util = parts
        last_info = {
            "index": index,
            "pci_bus_id": pci_bus_id,
            "uuid": uuid,
            "name": name,
            "memory_total_mb": memory_total,
            "memory_free_mb": memory_free,
            "memory_used_mb": memory_used,
            "util_pct": util,
        }

        try:
            free_mb = float(memory_free)
            util_pct = float(util)
        except ValueError as exc:
            raise RuntimeError(
                f"GPU preflight failed: non-numeric nvidia-smi field: {parts}"
            ) from exc

        print(
            f"  sample {idle_count + 1}/{PREFLIGHT_SAMPLES}: {name} "
            f"pci={pci_bus_id} uuid={uuid} free={free_mb:.0f}MiB util={util_pct:.0f}%"
        )

        if util_pct <= PREFLIGHT_MAX_UTIL_PCT and free_mb >= PREFLIGHT_MIN_FREE_MB:
            idle_count += 1
        else:
            print(
                f"  GPU not ready (util={util_pct:.0f}%, free={free_mb:.0f}MiB); "
                f"resetting idle counter."
            )
            idle_count = 0

        if idle_count < PREFLIGHT_SAMPLES:
            time.sleep(PREFLIGHT_INTERVAL_S)

    print(f"Preflight passed: {target_label} info={last_info}")


# Run the idle gate before importing torch/CUDA.
_gpu_preflight()


class _BaseMacaTest(ModelTest):
    """Shared Llama-3.2-1B configuration for the MaCa A/B variants."""

    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_SIZE = 512
    DATASET_CONCAT_SIZE = 2048
    DELETE_QUANTIZED_MODEL = True
    SAVE_PATH = None
    LOAD_BACKEND = BACKEND.MARLIN
    QUANT_BACKEND = BACKEND.AUTO
    HESSIAN_CHUNK_SIZE = None
    HESSIAN_LENGTH_AWARE: Union[bool, LengthAwareConfig] = False
    HESSIAN_TARGET_BUCKET_COUNT: Optional[int] = None
    HESSIAN_BUCKET_WEIGHT_EXPONENT: float = 1.0
    # Use ModelTest defaults (OFFLOAD_TO_DISK=True, DELETE_QUANTIZED_MODEL=True)

    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "paged|flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {
                "value": 0.47229114971050457,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3242320819112628,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.3515358361774744,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "evalution_batch_size": 16,
        },
    }
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    def test_maca(self):
        """Dummy test method so this unittest.TestCase subclass can be instantiated."""
        pass

    def _build_quantize_config(self):
        qcfg = super()._build_quantize_config()
        if hasattr(qcfg, "hessian"):
            if getattr(self, "HESSIAN_TARGET_BUCKET_COUNT", None) is not None:
                length_aware = LengthAwareConfig.from_lengths(
                    _CALIBRATION_TOKEN_LENGTHS,
                    mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
                    bucket_weight_exponent=self.HESSIAN_BUCKET_WEIGHT_EXPONENT,
                    target_bucket_count=self.HESSIAN_TARGET_BUCKET_COUNT,
                )
            else:
                length_aware = self.HESSIAN_LENGTH_AWARE
            qcfg.hessian = HessianConfig(
                chunk_size=self.HESSIAN_CHUNK_SIZE,
                length_aware=length_aware,
            )
        return qcfg


class _TestA(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 2048
    HESSIAN_LENGTH_AWARE = False


class _TestB(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = False


class _TestC(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = True


def _compute_calibration_token_lengths(tokenizer, rows: int = 512) -> list[int]:
    """Tokenize the calibration dataset and return per-example token counts."""
    raw_dataset = _BaseMacaTest.load_dataset(tokenizer=tokenizer, rows=rows)
    dummy_qmodel = SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=False,
        quantize_config=None,
    )
    prepared = prepare_calibration_dataset(
        dummy_qmodel,
        calibration_dataset=raw_dataset,
        calibration_dataset_concat_size=None,
        batch_size=1,
        calibration_data_min_length=10,
    )
    lengths = []
    for batch in prepared:
        if isinstance(batch, dict) and "attention_mask" in batch:
            lengths.append(int(batch["attention_mask"].sum().item()))
        else:
            lengths.append(int(batch["input_ids"].shape[-1]))
    return lengths


_TOKENIZER = _BaseMacaTest.load_tokenizer(_BaseMacaTest.NATIVE_MODEL_ID)
_CALIBRATION_TOKEN_LENGTHS = _compute_calibration_token_lengths(
    _TOKENIZER, rows=_BaseMacaTest.DATASET_SIZE
)


class _TestD(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
    )


class _TestE(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig(mode=LengthAwareMode.SINGLE)


class _TestF(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.4,
    )


class _TestG(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.2,
    )


class _TestH(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.6,
    )


class _TestI(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.3,
    )


class _TestJ(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.35,
    )


class _TestK(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.45,
    )


class _TestL(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.5,
    )


class _TestM(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.55,
    )


class _TestN(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.65,
    )


class _TestO(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.625,
    )


class _TestP(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.675,
    )


class _TestQ(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_LENGTH_AWARE = LengthAwareConfig.from_lengths(
        _CALIBRATION_TOKEN_LENGTHS,
        mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
        bucket_weight_exponent=0.7,
    )


class _TestR(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 10
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.2


class _TestS(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 12
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.2


class _TestT(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 14
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.2


class _TestU(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 18
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.2


class _TestV(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 20
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.2


class _TestW(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 22
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.2


class _TestX(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 24
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.2


class _TestY(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 6
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.15


class _TestZ(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 6
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.25


class _TestAA(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 14
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.15


class _TestAB(_BaseMacaTest):
    DATASET_CONCAT_SIZE = 0
    HESSIAN_TARGET_BUCKET_COUNT = 14
    HESSIAN_BUCKET_WEIGHT_EXPONENT = 0.25


def _length_aware_label(cfg: Union[bool, LengthAwareConfig]) -> str:
    """Short label for the length-aware setting used in the results table."""
    if isinstance(cfg, bool):
        return str(cfg)
    if cfg.mode is LengthAwareMode.DISABLED:
        return "False"
    if cfg.mode is LengthAwareMode.SINGLE:
        return "True"
    buckets = len(cfg.bucket_boundaries) - 1 if cfg.bucket_boundaries else 0
    exponent = getattr(cfg, "bucket_weight_exponent", 1.0)
    return f"{cfg.mode.value}({buckets}b,p={exponent})"


def _extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Flatten task-level metric dicts to ``task.metric`` -> float."""
    flat: Dict[str, float] = {}
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if metric_name == "alias" or "stderr" in metric_name:
                continue
            try:
                flat[f"{task_name}.{metric_name}"] = float(value)
            except (TypeError, ValueError):
                pass
    return flat


def _run_extra_mmlu(q_model, backend: BACKEND) -> Dict[str, Dict[str, float]]:
    """Evaluate mmlu_history and mmlu_chemistry on the loaded quantized model."""
    extra: Dict[str, Dict[str, float]] = {}
    for label, subsets in (
        ("mmlu_history", MMLU_HISTORY_SUBSETS),
        ("mmlu_chemistry", MMLU_CHEMISTRY_SUBSETS),
    ):
        print(f"\n[extra eval] {label}: subsets={subsets}")
        result = eval_run(
            model_or_id_or_path=q_model,
            tasks=["mmlu"],
            backend=backend,
            batch_size=16,
            apply_chat_template=False,
            suite_kwargs={"subsets": subsets},
            trust_remote_code=False,
        )
        task_results = get_eval_task_results(result)
        if task_results:
            metrics = next(iter(task_results.values()))
            extra[label] = metrics
            print(f"[extra eval] {label} metrics: {metrics}")
        else:
            print(f"[extra eval] {label}: no metrics returned")
    return extra


def run_variant(test_cls, variant_label: str) -> Dict[str, Any]:
    test_cls.setUpClass()
    test = test_cls(methodName="test_maca")
    length_aware_cfg = test._build_quantize_config().hessian.length_aware
    print(
        f"\n=== Variant {variant_label}: "
        f"concat_size={test_cls.DATASET_CONCAT_SIZE}, "
        f"length_aware={_length_aware_label(length_aware_cfg)} ==="
    )
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    q_model = None
    try:
        q_model, _, _ = test.quantModel(
            test.NATIVE_MODEL_ID,
            batch_size=test.QUANT_BATCH_SIZE,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Variant {variant_label} failed during quantize/eval: {exc}")
        traceback.print_exc()
        return {
            "variant": variant_label,
            "concat_size": test_cls.DATASET_CONCAT_SIZE,
            "length_aware": _length_aware_label(test_cls.HESSIAN_LENGTH_AWARE),
            "total_time": time.perf_counter() - t0,
            "error": str(exc),
            "results": {},
        }
    total_time = time.perf_counter() - t0

    backend = test._current_load_backend()
    records = getattr(test, "_post_quant_eval_records", {}) or {}
    results = dict(records.get(backend, {}))
    print(
        f"Variant {variant_label} backend={backend.name} results: "
        f"{json.dumps(results, indent=2)}"
    )

    # Run additional MMLU subsets on the already-loaded quantized model.
    if q_model is not None:
        try:
            extra_mmlu = _run_extra_mmlu(q_model, backend)
            results.update(extra_mmlu)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Variant {variant_label} extra MMLU eval failed: {exc}")
            traceback.print_exc()

        # Clean up the quantized model and its temp save directory.
        try:
            test._cleanup_quantized_model(q_model, enabled=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Cleanup warning for {variant_label}: {exc}")
        del q_model

    torch_empty_cache()
    torch.cuda.empty_cache()

    return {
        "variant": variant_label,
        "concat_size": test_cls.DATASET_CONCAT_SIZE,
        "length_aware": _length_aware_label(length_aware_cfg),
        "total_time": total_time,
        "results": _extract_metrics(results),
    }


VARIANT_MAP = {
    "A": _TestA,
    "B": _TestB,
    "C": _TestC,
    "D": _TestD,
    "E": _TestE,
    "F": _TestF,
    "G": _TestG,
    "H": _TestH,
    "I": _TestI,
    "J": _TestJ,
    "K": _TestK,
    "L": _TestL,
    "M": _TestM,
    "N": _TestN,
    "O": _TestO,
    "P": _TestP,
    "Q": _TestQ,
    "R": _TestR,
    "S": _TestS,
    "T": _TestT,
    "U": _TestU,
    "V": _TestV,
    "W": _TestW,
    "X": _TestX,
    "Y": _TestY,
    "Z": _TestZ,
    "AA": _TestAA,
    "AB": _TestAB,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Full all-layer Llama-3.2-1B MaCa A/B test on GPU 3."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Comma-separated list of variants to run "
             "(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB). "
             "If omitted, all variants are run.",
    )
    args = parser.parse_args(argv)

    print(f"PyTorch: {torch.__version__}")
    print(
        f"CUDA visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}"
    )

    selected = list(VARIANT_MAP.items())
    if args.variant:
        labels = [v.strip() for v in args.variant.split(",") if v.strip()]
        selected = [(label, VARIANT_MAP[label]) for label in labels if label in VARIANT_MAP]

    all_results = []
    for label, cls in selected:
        all_results.append(run_variant(cls, label))

    print("\n=== A/B Summary ===")
    header = (
        "| variant | concat | length_aware           | total(s) | "
        "arc acc | arc acc_norm | gsm8k plat acc | "
        "mmlu_stem acc,ll | mmlu_stem acc,ll_avg | "
        "mmlu_history acc,ll | mmlu_chemistry acc,ll |"
    )
    print(header)
    print(
        "|"
        + "|".join(
            ["-" * (len(h) + 2) for h in header.split("|") if h]
        )
        + "|"
    )
    for r in all_results:
        res = r["results"]
        arc_acc = res.get("arc_challenge.accuracy,loglikelihood", float("nan"))
        arc_acc_norm = res.get(
            "arc_challenge.accuracy,loglikelihood_norm", float("nan")
        )
        gsm_acc = res.get("gsm8k_platinum_cot.acc,num", float("nan"))
        mmlu_stem_acc = res.get("mmlu_stem.acc,ll", float("nan"))
        mmlu_stem_acc_norm = res.get("mmlu_stem.acc,ll_avg", float("nan"))
        mmlu_hist_acc = res.get("mmlu_history.acc,ll", float("nan"))
        mmlu_chem_acc = res.get("mmlu_chemistry.acc,ll", float("nan"))
        print(
            f"| {r['variant']:7} | {str(r['concat_size']):6} | "
            f"{r['length_aware']:20} | {r['total_time']:8.1f} | "
            f"{arc_acc:.4f} | {arc_acc_norm:.4f} | {gsm_acc:.4f} | "
            f"{mmlu_stem_acc:.4f} | {mmlu_stem_acc_norm:.4f} | "
            f"{mmlu_hist_acc:.4f} | {mmlu_chem_acc:.4f} |"
        )

    if args.variant:
        suffix = args.variant.replace(",", "_")
        out_name = f"maca_ab_full_results_{suffix}.json"
    else:
        out_name = "maca_ab_full_results.json"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results written to {out_path}")


if __name__ == "__main__":
    main()
