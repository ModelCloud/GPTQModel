# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired, full-row W4A16 versus W4A float activation quality checks.

The W4A16 reference is the same GPTQ checkpoint with its activation policy
removed. Its large tensor file is symlinked, so packed INT4 weights and GPTQ
scales are identical in both lanes. Full GSM8K Platinum is the acceptance
gate; ARC-Challenge is a sensitivity diagnostic and never controls the exit
status of a comparison run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


TASKS = {
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "accuracy,loglikelihood"),
    "gsm8k_platinum_cot": ("madrylab/gsm8k-platinum", "main", "acc,num"),
}
EXPECTED_TEST_ROWS = {"arc_challenge": 1172, "gsm8k_platinum_cot": 1209}


def prepare_stream_view(checkpoint: Path, view: Path) -> Path:
    """Explicitly migrate an earlier policy to the consumer-driven stream.

    The packed GPTQ weights and all tokenizer files are linked unchanged.
    This only changes the activation execution contract; callers must repeat
    their quality evaluation before treating the view as a validated model.
    """
    source = checkpoint.resolve()
    if source == view.resolve():
        raise ValueError("The stream view must be separate from its source checkpoint.")
    config = json.loads((source / "quantize_config.json").read_text())
    activation = config.get("activation")
    if (not isinstance(activation, dict) or activation.get("version") not in {1, 2}
            or activation.get("mode") not in {"w4afp8", "w4a_nvfp4"}):
        raise ValueError("Expected a version 1 or version 2 W4A checkpoint.")
    if config.get("bits") != 4 or config.get("pack_dtype") != "int32":
        raise ValueError("The source must retain INT32-packed GPTQ INT4 weights.")
    migrated = {"version": 3, "mode": activation["mode"]}
    if activation["mode"] == "w4a_nvfp4":
        from gptqmodel.quantization.activation_floatx import normalize_nvfp4_recipe

        migrated["recipe"] = normalize_nvfp4_recipe(activation.get("recipe", "four_six"))
    config["activation"] = migrated
    view.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file() or item.name == "quantize_config.json":
            continue
        target = view / item.name
        if target.is_symlink() and target.resolve() == item.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Stream view artifact differs: {target}")
        target.symlink_to(item.resolve())
    target = view / "quantize_config.json"
    payload = json.dumps(config, indent=2) + "\n"
    if target.exists() and target.read_text() != payload:
        raise ValueError(f"Stream config differs: {target}")
    target.write_text(payload)
    if not (view / "model.safetensors").samefile(source / "model.safetensors"):
        raise AssertionError("Stream migration must preserve the exact packed-weight tensor file.")
    print(json.dumps({
        "source": str(source), "stream_view": str(view), "mode": activation["mode"],
        "source_version": activation["version"], "target_version": 3,
    }))
    return view


def prepare_dated_view(checkpoint: Path, view: Path, baseline_result: Path) -> Path:
    """Reuse a checkpoint with the chat-template date captured by a baseline run."""
    source = checkpoint.resolve()
    if source == view.resolve():
        raise ValueError("The dated view must be separate from the checkpoint.")
    baseline = json.loads(baseline_result.read_text())
    sample = baseline["tests"][0]["samples"][0]
    match = re.search(r"Today Date: ([^\n]+)", sample["prompt"])
    if match is None or not re.fullmatch(r"\d{1,2} [A-Z][a-z]{2} \d{4}", match.group(1)):
        raise ValueError("The baseline prompt has no supported Llama date stamp.")
    date = match.group(1)
    template = (source / "chat_template.jinja").read_text()
    expression = 'strftime_now("%d %b %Y")'
    if template.count(expression) != 1:
        raise ValueError("Expected one dynamic date expression in the chat template.")
    frozen = template.replace(expression, json.dumps(date))
    view.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file() or item.name == "chat_template.jinja":
            continue
        link = view / item.name
        if link.is_symlink() and link.resolve() == item.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Dated view artifact differs: {link}")
        link.symlink_to(item.resolve())
    destination = view / "chat_template.jinja"
    if destination.exists() and destination.read_text() != frozen:
        raise ValueError(f"Dated chat template differs: {destination}")
    destination.write_text(frozen)
    if (view / "model.safetensors").resolve() != (source / "model.safetensors").resolve():
        raise AssertionError("Dated view must use the identical packed weight tensor file.")
    print(json.dumps({"view": str(view), "date": date, "baseline": str(baseline_result)}))
    return view


def prepare_reference(checkpoint: Path, reference: Path) -> Path:
    """Create a metadata-only W4A16 view of one W4A checkpoint."""
    source = checkpoint.resolve()
    config_path = source / "quantize_config.json"
    config = json.loads(config_path.read_text())
    if config.get("bits") != 4 or config.get("pack_dtype") != "int32":
        raise ValueError("The source must contain native INT32-packed GPTQ INT4 weights.")
    if config.get("activation", {}).get("mode") not in {"w4afp8", "w4a_nvfp4"}:
        raise ValueError("The source must be a W4A float activation checkpoint.")
    config.pop("activation")
    reference.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file() or item.name == "quantize_config.json":
            continue
        link = reference / item.name
        if link.is_symlink() and link.resolve() == item.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Reference artifact differs: {link}")
        link.symlink_to(item.resolve())
    target = reference / "quantize_config.json"
    payload = json.dumps(config, indent=2) + "\n"
    if target.exists() and target.read_text() != payload:
        raise ValueError(f"Reference config differs: {target}")
    target.write_text(payload)
    if (reference / "model.safetensors").resolve() != (source / "model.safetensors").resolve():
        raise AssertionError("W4A16 must use the identical packed weight tensor file.")
    return reference


def prepare_attention_split_view(checkpoint: Path, view: Path,
                                 attention_mode: str = "w4afp8") -> Path:
    """Create a metadata-only mixed FP8-attention / NVFP4-MLP view.

    The packed GPTQ INT4 weights and every tokenizer artifact are linked
    unchanged. Only the activation policy gains an ``attention`` sub-policy, so
    the saved weight format is untouched and the same-weight paired gate stays
    valid.
    """
    if attention_mode not in {"w4afp8", "w4a_nvfp4"}:
        raise ValueError("Attention mode must be `w4afp8` or `w4a_nvfp4`.")
    source = checkpoint.resolve()
    if source == view.resolve():
        raise ValueError("The split view must be separate from its source checkpoint.")
    config = json.loads((source / "quantize_config.json").read_text())
    activation = config.get("activation")
    if not isinstance(activation, dict) or activation.get("mode") != "w4a_nvfp4":
        raise ValueError("The attention split requires an NVFP4 W4A checkpoint.")
    if activation.get("version") not in {3, 4}:
        raise ValueError("The attention split requires activation version 3 or 4.")
    if config.get("bits") != 4 or config.get("pack_dtype") != "int32":
        raise ValueError("The source must retain INT32-packed GPTQ INT4 weights.")
    attention = {"mode": attention_mode}
    if attention_mode == "w4a_nvfp4":
        attention["recipe"] = activation.get("recipe", "least_squares")
    split = dict(activation)
    split["attention"] = attention
    config["activation"] = split
    view.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file() or item.name == "quantize_config.json":
            continue
        target = view / item.name
        if target.is_symlink() and target.resolve() == item.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Split view artifact differs: {target}")
        target.symlink_to(item.resolve())
    target = view / "quantize_config.json"
    payload = json.dumps(config, indent=2) + "\n"
    if target.exists() and target.read_text() != payload:
        raise ValueError(f"Split config differs: {target}")
    target.write_text(payload)
    if not (view / "model.safetensors").samefile(source / "model.safetensors"):
        raise AssertionError("The split view must preserve the exact packed-weight tensor file.")
    print(json.dumps({"source": str(source), "view": str(view), "attention": attention}))
    return view


def prepare_mlp_override_view(checkpoint: Path, view: Path,
                              layers: Sequence[int]) -> Path:
    """Create a metadata-only per-layer FP8-MLP view.

    Selected decoder layers carry FP8 on their MLP boundaries while every other
    layer keeps the NVFP4 stream default. The packed GPTQ INT4 weights and all
    tokenizer artifacts are linked unchanged, so a paired same-weight gate
    stays valid.
    """
    layers = tuple(sorted(layers))
    if not layers or any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                         for index in layers):
        raise ValueError("The MLP override requires unique non-negative decoder layer indices.")
    if len(set(layers)) != len(layers):
        raise ValueError("The MLP override requires unique decoder layer indices.")
    source = checkpoint.resolve()
    if source == view.resolve():
        raise ValueError("The MLP view must be separate from its source checkpoint.")
    config = json.loads((source / "quantize_config.json").read_text())
    activation = config.get("activation")
    if not isinstance(activation, dict) or activation.get("mode") != "w4a_nvfp4":
        raise ValueError("The MLP override requires an NVFP4 W4A checkpoint.")
    if activation.get("version") not in {3, 4}:
        raise ValueError("The MLP override requires activation version 3 or 4.")
    if config.get("bits") != 4 or config.get("pack_dtype") != "int32":
        raise ValueError("The source must retain INT32-packed GPTQ INT4 weights.")
    if "mlp" in activation:
        raise ValueError("The source already carries an MLP override; start from the stream default.")
    override = dict(activation)
    override["mlp"] = {"mode": "w4afp8", "layers": list(layers)}
    config["activation"] = override
    view.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file() or item.name == "quantize_config.json":
            continue
        target = view / item.name
        if target.is_symlink() and target.resolve() == item.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"MLP view artifact differs: {target}")
        target.symlink_to(item.resolve())
    target = view / "quantize_config.json"
    payload = json.dumps(config, indent=2) + "\n"
    if target.exists() and target.read_text() != payload:
        raise ValueError(f"MLP view config differs: {target}")
    target.write_text(payload)
    if not (view / "model.safetensors").samefile(source / "model.safetensors"):
        raise AssertionError("The MLP view must preserve the exact packed-weight tensor file.")
    print(json.dumps({"source": str(source), "view": str(view), "mlp": override["mlp"]}))
    return view


def verify_tokenizer(native: Path, checkpoint: Path, reference: Path) -> None:
    """Check the dense and two quantized lanes render identical input IDs."""
    from transformers import AutoTokenizer
    from tokenicer import Tokenicer

    paths = (native, checkpoint, reference)
    prompts = (
        "Which city is the capital of France?",
        "If there are 3 cars and 2 arrive, how many cars are there?",
    )
    tokenizers = [AutoTokenizer.from_pretrained(path, local_files_only=True) for path in paths]
    wrapped = [Tokenicer.load(str(path)) for path in paths]

    def token_ids(value):
        if isinstance(value, Mapping):
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return list(value)

    report = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        families = []
        for family in (tokenizers, wrapped):
            rendered = [tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for tok in family]
            ids = [token_ids(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                   for tok in family]
            if not (rendered[0] == rendered[1] == rendered[2] and ids[0] == ids[1] == ids[2]):
                raise AssertionError(f"Dense/W4A16/W4A prompt or input IDs differ for {prompt!r}")
            masks = [list(tok(rendered[0], add_special_tokens=False)["attention_mask"]) for tok in family]
            if not (masks[0] == masks[1] == masks[2]):
                raise AssertionError(f"Dense/W4A16/W4A attention masks differ for {prompt!r}")
            families.append((rendered[0], ids[0], masks[0]))
        if families[0] != families[1]:
            raise AssertionError(f"AutoTokenizer and Tokenicer disagree for {prompt!r}")
        report.append({"prompt": prompt, "rendered": families[0][0],
                       "input_ids": families[0][1], "attention_mask": families[0][2]})
    print(json.dumps(report, indent=2))


def paired_eval_batch_size(checkpoint: Path, baseline_result: Path, task: str,
                           requested: int | None = None) -> int:
    """Reject an incompatible paired run before model loading or generation."""
    baseline = json.loads(baseline_result.read_text())
    if len(baseline["tests"]) != 1 or baseline["tests"][0]["name"] != task:
        raise ValueError("The frozen baseline must contain the requested task")
    test = baseline["tests"][0]
    if len(test["samples"]) != EXPECTED_TEST_ROWS[task]:
        raise ValueError("The frozen baseline must contain every test row")
    engine = baseline["engine"]
    fixed = {"seed": 42, "dtype": "bfloat16", "padding_side": "left"}
    if task == "gsm8k_platinum_cot":
        fixed["max_new_tokens"] = 256
    for key, value in fixed.items():
        if engine.get(key) != value:
            raise ValueError(f"Frozen baseline uses unsupported engine setting {key}: {engine.get(key)}")
    batch = engine.get("batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("Frozen baseline batch size must be a positive integer")
    if requested is not None and requested != batch:
        raise ValueError(f"Requested batch size {requested} differs from frozen baseline {batch}")
    reference = Path(baseline["model"]["path"])
    if not (checkpoint / "model.safetensors").samefile(reference / "model.safetensors"):
        raise ValueError("Paired evaluation requires the exact frozen baseline weight file")
    for name in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        if (checkpoint / name).read_bytes() != (reference / name).read_bytes():
            raise ValueError(f"Paired evaluation tokenizer artifact differs: {name}")
    return batch


def evaluate_full_rows(checkpoint: Path, variant: str, task: str, output: Path,
                       reference: Path | None = None, batch_size: int | None = None,
                       baseline_result: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    if baseline_result is not None:
        batch_size = paired_eval_batch_size(checkpoint, baseline_result, task, batch_size)
        print(json.dumps({"paired_baseline": str(baseline_result), "batch_size": batch_size,
                          "preflight": "matched_weights_tokenizer_and_engine"}), flush=True)
    elif batch_size is None:
        batch_size = 8
    from datasets import load_dataset
    from gptqmodel import BACKEND
    from tests.eval import evaluate

    if variant == "w4a16":
        if reference is None:
            raise ValueError("W4A16 requires --reference pointing to a prepared checkpoint view.")
        model_path = prepare_reference(checkpoint, reference)
        # This portable baseline does not require Marlin's optional CUDA JIT extension.
        backend = BACKEND.GPTQ_TRITON
    else:
        model_path = checkpoint
        backend = {
            "w4afp8": BACKEND.GPTQ_W4AFP8,
            "w4a_nvfp4": BACKEND.GPTQ_W4A_NVFP4,
        }[variant]

    dataset_path, dataset_name, metric = TASKS[task]
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive.")
    suite_kwargs = {"stream": True}
    if task == "gsm8k_platinum_cot":
        suite_kwargs.update(batch_size=batch_size, max_new_tokens=256)
    result = evaluate(
        model_or_id_or_path=str(model_path),
        backend=backend,
        tasks=[task],
        batch_size=batch_size,
        apply_chat_template=True,
        model_args={"dtype": "bfloat16", "device": "cuda:0", "seed": 42},
        gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
        suite_kwargs=suite_kwargs,
        output_path=str(output),
    )
    test = result["tests"][0]
    expected_rows = len(load_dataset(dataset_path, dataset_name, split="test"))
    if len(test["samples"]) != expected_rows:
        raise AssertionError(f"{task} evaluated {len(test['samples'])}/{expected_rows} rows.")
    print(json.dumps({"variant": variant, "task": task, "rows": expected_rows,
                      "metric": metric, "score": test["metrics"][metric], "output": str(output)}))


def compare(reference_result: Path, quantized_result: Path, task: str,
            allowed_drop_pp: float, metric_name: str | None = None) -> dict:
    base_run = json.loads(reference_result.read_text())
    quant_run = json.loads(quantized_result.read_text())
    summary = _compare_runs(base_run, quant_run, task, allowed_drop_pp, metric_name)
    print(json.dumps(summary, indent=2))
    return summary


def _compare_runs(base_run, quant_run, task, allowed_drop_pp, metric_name=None,
                  *, same_weights=True):
    if not math.isfinite(allowed_drop_pp) or not 0 <= allowed_drop_pp <= 100:
        raise ValueError("Allowed drop must be finite and between 0 and 100 percentage points")
    if len(base_run["tests"]) != 1 or len(quant_run["tests"]) != 1:
        raise ValueError("Paired results must contain exactly one test")
    base = base_run["tests"][0]
    quant = quant_run["tests"][0]
    metric = metric_name or TASKS[task][2]
    if base["name"] != task or quant["name"] != task:
        raise ValueError("Result task names do not match the requested task.")
    if base["metadata"] != quant["metadata"]:
        raise ValueError("The paired evaluations used different task settings.")
    for key in ("seed", "batch_size", "dtype", "max_new_tokens", "padding_side"):
        if base_run["engine"].get(key) != quant_run["engine"].get(key):
            raise ValueError(f"The paired evaluations differ in engine setting {key}.")
    left_model = Path(base_run["model"]["path"]) / "model.safetensors"
    right_model = Path(quant_run["model"]["path"]) / "model.safetensors"
    if same_weights and not left_model.samefile(right_model):
        raise ValueError("The paired results must use the same packed-weight tensor file.")
    left, right = base["samples"], quant["samples"]
    if len(left) != len(right) or not left:
        raise ValueError("The paired results have different or empty row sets.")
    if len(left) != EXPECTED_TEST_ROWS[task]:
        raise ValueError(f"The paired results do not contain every {task} test row.")
    for test in (base, quant):
        if {row["index"] for row in test["samples"]} != set(range(EXPECTED_TEST_ROWS[task])):
            raise ValueError("Result indices must cover every test row exactly once")
        scores = [float(row["scores"][metric]) for row in test["samples"]]
        if any(value not in (0., 1.) for value in scores):
            raise ValueError("Classification scores must be finite binary values")
        if not math.isclose(float(test["metrics"][metric]), sum(scores) / len(scores),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("Aggregate score differs from the saved per-row scores")
    differences = []
    losses = gains = 0
    extracted_changes = 0
    answer_changes = 0
    for old, new in zip(left, right, strict=True):
        if (old["index"], old["prompt"], old["target"]) != (
                new["index"], new["prompt"], new["target"]):
            raise AssertionError("Paired evaluation rows or rendered prompts differ.")
        old_score, new_score = float(old["scores"][metric]), float(new["scores"][metric])
        differences.append(new_score - old_score)
        losses += old_score > new_score
        gains += new_score > old_score
        extracted_changes += old["extracted"] != new["extracted"]
        if task == "arc_challenge":
            answer_changes += old["prediction"] != new["prediction"]
        else:
            answer_changes += (old["extracted"].get("numeric-extract") !=
                               new["extracted"].get("numeric-extract"))
    count = len(differences)
    delta = sum(differences) / count
    variance = sum((value - delta) ** 2 for value in differences) / max(1, count - 1)
    half_width = 1.96 * math.sqrt(variance / count)
    discordant = losses + gains
    mcnemar_p = (min(1.0, 2 * sum(math.comb(discordant, k)
                               for k in range(min(losses, gains) + 1)) / 2 ** discordant)
                 if discordant else 1.0)
    if 100 * (delta + half_width) < -allowed_drop_pp:
        statistical_verdict = "confirmed_regression"
    elif 100 * (delta - half_width) >= -allowed_drop_pp:
        statistical_verdict = "within_budget"
    else:
        statistical_verdict = "inconclusive"
    acceptance_task = task == "gsm8k_platinum_cot"
    verdict = statistical_verdict if acceptance_task else "diagnostic_only"
    summary = {
        "task": task, "metric": metric, "rows": count,
        "w4a16": base["metrics"][metric], "w4a_float": quant["metrics"][metric],
        "delta_pp": 100 * delta, "paired_95pct_ci_pp": [100 * (delta - half_width), 100 * (delta + half_width)],
        "base_only_correct": losses, "float_only_correct": gains,
        "correctness_flips": discordant, "answer_changes": answer_changes,
        "extracted_field_changes": extracted_changes,
        "mcnemar_exact_p": mcnemar_p,
        "allowed_drop_pp": allowed_drop_pp,
        "acceptance_task": acceptance_task,
        "verdict": verdict,
        "statistical_verdict": statistical_verdict,
        "statistically_detectable_drop": delta + half_width < 0,
        "material_regression": acceptance_task and statistical_verdict == "confirmed_regression",
    }
    return summary


def _engine_protocol(run):
    engine = copy.deepcopy(run["engine"])
    engine.pop("backend", None)
    execution = engine.get("execution", {})
    execution.pop("quantized_backend", None)
    execution.pop("runtime_format", None)
    return engine


def compare_adapted(frozen_reference: Path, candidate_reference: Path,
                    candidate_float: Path, allowed_drop_pp: float = 2.0) -> dict:
    """Keep a degraded weight-only control from hiding activation regression.

    The same-weight activation comparison retains its original gate. An
    additional paired comparison uses the frozen original reference, and the
    candidate A16 point score must not fall below that reference. The native
    confidence interval is reported separately; a point floor is not a claim
    of statistically proven native noninferiority.
    """
    paths = (frozen_reference, candidate_reference, candidate_float)
    runs = [json.loads(path.read_text()) for path in paths]
    frozen, native, quant = runs
    for run in runs:
        model_path = Path(run["model"]["path"])
        if not (model_path / "model.safetensors").is_file():
            raise ValueError("Evaluation checkpoint weights are missing")
        cfg = json.loads((model_path / "quantize_config.json").read_text())
        if cfg.get("bits") != 4 or cfg.get("pack_dtype") != "int32":
            raise ValueError("Adapted acceptance requires native INT32-packed GPTQ INT4")
        activation = cfg.get("activation", {}).get("mode")
        if run is quant:
            if activation not in {"w4afp8", "w4a_nvfp4"}:
                raise ValueError("Candidate activation result must use a W4A float policy")
            expected_backend = {"w4afp8": "gptq_w4afp8", "w4a_nvfp4": "gptq_w4a_nvfp4"}[activation]
            if (run["engine"].get("backend") != expected_backend
                    or run["engine"].get("execution", {}).get("quantized_backend") != expected_backend):
                raise ValueError("Candidate execution backend differs from its activation policy")
        elif activation is not None:
            raise ValueError("Native reference results must have activation quantization disabled")
        if _engine_protocol(run) != _engine_protocol(frozen):
            raise ValueError("Adapted comparisons require identical evaluation engine settings")
        if run.get("versions", {}) != frozen.get("versions", {}):
            raise ValueError("Evaluation software versions differ")
    if native["engine"].get("backend") != frozen["engine"].get("backend"):
        raise ValueError("Native preservation requires the same evaluation backend")
    task = "gsm8k_platinum_cot"
    paired = _compare_runs(native, quant, task, allowed_drop_pp)
    original = _compare_runs(frozen, quant, task, allowed_drop_pp, same_weights=False)
    preservation = _compare_runs(frozen, native, task, 0., same_weights=False)
    metric = TASKS[task][2]
    counts = [sum(int(row["scores"][metric]) for row in run["tests"][0]["samples"])
              for run in runs]
    native_floor = counts[1] >= counts[0]
    accepted = (native_floor and paired["verdict"] == "within_budget"
                and original["verdict"] == "within_budget")
    return {
        "task": task, "rows": EXPECTED_TEST_ROWS[task], "allowed_drop_pp": allowed_drop_pp,
        "frozen_native_correct": counts[0], "candidate_native_correct": counts[1],
        "candidate_float_correct": counts[2], "native_point_floor_met": native_floor,
        "same_weight_activation": paired, "frozen_reference_activation": original,
        "native_preservation": preservation, "accepted": accepted,
        "verdict": "accepted" if accepted else "not_accepted",
        "result_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--reference", type=Path, required=True)
    prepare.add_argument("--native", type=Path)
    dated = sub.add_parser("freeze-date")
    dated.add_argument("--checkpoint", type=Path, required=True)
    dated.add_argument("--view", type=Path, required=True)
    dated.add_argument("--baseline", type=Path, required=True)
    stream = sub.add_parser("stream-view")
    stream.add_argument("--checkpoint", type=Path, required=True)
    stream.add_argument("--view", type=Path, required=True)
    split = sub.add_parser("split-view")
    split.add_argument("--checkpoint", type=Path, required=True)
    split.add_argument("--view", type=Path, required=True)
    split.add_argument("--attention-mode", choices=("w4afp8", "w4a_nvfp4"), default="w4afp8")
    mlp = sub.add_parser("mlp-view")
    mlp.add_argument("--checkpoint", type=Path, required=True)
    mlp.add_argument("--view", type=Path, required=True)
    mlp.add_argument("--layers", type=int, nargs="+", required=True)
    run = sub.add_parser("eval")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--reference", type=Path)
    run.add_argument("--variant", choices=("w4a16", "w4afp8", "w4a_nvfp4"), required=True)
    run.add_argument("--task", choices=tuple(TASKS), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--batch-size", type=int,
                     help="Default: inherit --baseline-result batch size, otherwise 8.")
    run.add_argument("--baseline-result", type=Path,
                     help="Preflight a frozen same-weight baseline and inherit its batch size.")
    analysis = sub.add_parser("compare")
    analysis.add_argument("--w4a16", type=Path, required=True)
    analysis.add_argument("--w4a-float", type=Path, required=True)
    analysis.add_argument("--task", choices=tuple(TASKS), required=True)
    analysis.add_argument("--metric")
    analysis.add_argument("--allowed-drop-pp", type=float, default=2.0)
    analysis.add_argument("--output", type=Path)
    adapted = sub.add_parser("accept-adapted")
    adapted.add_argument("--frozen-w4a16", type=Path, required=True)
    adapted.add_argument("--w4a16", type=Path, required=True)
    adapted.add_argument("--w4a-float", type=Path, required=True)
    adapted.add_argument("--allowed-drop-pp", type=float, default=2.0)
    adapted.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        reference = prepare_reference(args.checkpoint, args.reference)
        if args.native:
            verify_tokenizer(args.native, args.checkpoint, reference)
        print(f"W4A16 reference: {reference}")
    elif args.command == "freeze-date":
        prepare_dated_view(args.checkpoint, args.view, args.baseline)
    elif args.command == "stream-view":
        prepare_stream_view(args.checkpoint, args.view)
    elif args.command == "split-view":
        prepare_attention_split_view(args.checkpoint, args.view, args.attention_mode)
    elif args.command == "mlp-view":
        prepare_mlp_override_view(args.checkpoint, args.view, args.layers)
    elif args.command == "eval":
        evaluate_full_rows(
            args.checkpoint,
            args.variant,
            args.task,
            args.output,
            args.reference,
            args.batch_size,
            args.baseline_result,
        )
    elif args.command == "accept-adapted":
        if args.output.exists():
            raise FileExistsError(args.output)
        summary = compare_adapted(args.frozen_w4a16, args.w4a16, args.w4a_float, args.allowed_drop_pp)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if not summary["accepted"]:
            raise SystemExit(1)
    else:
        summary = compare(args.w4a16, args.w4a_float, args.task, args.allowed_drop_pp, args.metric)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n")
        if summary["acceptance_task"] and summary["verdict"] != "within_budget":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
