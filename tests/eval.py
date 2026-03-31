# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from tabulate import tabulate

from gptqmodel.utils.backend import BACKEND


_MMLU_LOCAL_DATASET = Path("/monster/data/model/dataset/hails-mmlu_no_train")
_GSM8K_LOCAL_DATASET = Path("/monster/data/model/dataset/gsm8k")
_ENGINE_OPTION_KEYS = {
    "attn_implementation",
    "attention_backend",
    "base_url",
    "context_length",
    "dp_size",
    "device",
    "device_map",
    "dtype",
    "enforce_eager",
    "gpu_memory_utilization",
    "llm_kwargs",
    "load_format",
    "max_model_len",
    "max_running_requests",
    "max_total_tokens",
    "mem_fraction_static",
    "padding_side",
    "pp_size",
    "quantization",
    "sampling_backend",
    "sampling_params",
    "seed",
    "skip_tokenizer_init",
    "tensor_parallel_size",
    "tokenizer_worker_num",
    "tokenizer_mode",
    "tokenizer_revision",
    "tp_size",
    "trust_remote_code",
    "vllm_path",
}
_DROPPED_MODEL_ARG_KEYS = {
    "backend",
    "gptqmodel",
    "model_id_or_path",
    "pretrained",
    "tokenizer",
}

DEFAULT_TASKS: tuple[str, ...] = ("arc_challenge",)
SUPPORTED_TASKS: tuple[str, ...] = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "gsm8k_cot",
    "gsm8k_platinum_cot",
    "gpqa",
    "hellaswag",
    "mmlu",
    "mmlu_pro",
    "mmlu_pro:math",
    "mmlu_stem",
    "openbookqa",
)


def import_evalution():
    try:
        return importlib.import_module("evalution")
    except ModuleNotFoundError:
        raise ValueError(
            "Evalution is required for evaluation. "
            "Install the `Evalution` package before running evaluation."
        ) from None


def list_supported_tasks() -> tuple[str, ...]:
    return SUPPORTED_TASKS


def normalize_eval_task_name(task: Any) -> str:
    if task is None:
        raise ValueError("Evaluation task identifier cannot be None")
    if isinstance(task, str):
        normalized = task.strip()
    else:
        normalized = str(task).strip()
    if not normalized:
        raise ValueError("Evaluation task identifier cannot be empty")
    return normalized


def format_eval_result_table(result: Mapping[str, Any]) -> str:
    rows = []
    for test in _result_tests(result):
        metrics = test.get("metrics", {})
        if not metrics:
            rows.append([test.get("name", ""), "-", "-"])
            continue
        for metric_name, value in metrics.items():
            rows.append([test.get("name", ""), metric_name, f"{float(value):.4f}"])

    if not rows:
        rows.append(["-", "-", "-"])
    return tabulate(rows, headers=["Task", "Metric", "Value"], tablefmt="github")


def get_eval_task_results(result: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(test.get("name", "")): {
            str(metric_name): float(metric_value)
            for metric_name, metric_value in (test.get("metrics", {}) or {}).items()
        }
        for test in _result_tests(result)
    }


def get_eval_task_metrics(result: Mapping[str, Any], task: Any) -> dict[str, float]:
    return get_eval_task_results(result).get(normalize_eval_task_name(task), {})


def resolve_eval_metric_alias(metric_name: str, metrics: Mapping[str, Any]) -> str | None:
    if metric_name in metrics:
        return metric_name

    aliases = {
        "acc": "accuracy,loglikelihood",
        "acc_norm": "accuracy,loglikelihood_norm",
        "acc,none": "accuracy,loglikelihood",
        "acc_norm,none": "accuracy,loglikelihood_norm",
    }
    alias = aliases.get(metric_name)
    if alias and alias in metrics:
        return alias
    return None


def evaluate(
    model_or_id_or_path: Any = None,
    tokenizer: Any = None,
    tasks: Any = None,
    batch_size: int | str = 1,
    trust_remote_code: bool = False,
    output_path: Optional[str] = None,
    llm_backend: str = "gptqmodel",
    backend: BACKEND | str | None = BACKEND.AUTO,
    model_args: Optional[Dict[str, Any]] = None,
    **args,
):
    normalized_llm_backend = str(llm_backend).strip().lower()
    if normalized_llm_backend not in {"gptqmodel", "vllm", "sglang"}:
        raise ValueError(
            "Evalution-backed evaluation only supports llm_backend='gptqmodel', 'vllm', or 'sglang'."
        )

    if tasks is None:
        task_list = list(DEFAULT_TASKS)
    elif isinstance(tasks, (list, tuple)):
        task_list = [normalize_eval_task_name(task) for task in tasks]
    else:
        task_list = [normalize_eval_task_name(tasks)]

    model_args = dict(model_args or {})
    gen_kwargs = args.pop("gen_kwargs", None)
    apply_chat_template = bool(args.pop("apply_chat_template", False))
    suite_kwargs = dict(args.pop("suite_kwargs", {}) or {})

    if args:
        unexpected = ", ".join(sorted(args.keys()))
        raise TypeError(f"Unsupported evaluation keyword arguments: {unexpected}")

    return run_evalution(
        model_or_id_or_path=model_or_id_or_path,
        tokenizer=tokenizer,
        tasks=task_list,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        output_path=output_path,
        llm_backend=normalized_llm_backend,
        backend=backend,
        model_args=model_args,
        apply_chat_template=apply_chat_template,
        gen_kwargs=gen_kwargs,
        suite_kwargs=suite_kwargs,
    )


def run_evalution(
    *,
    model_or_id_or_path: Any,
    tokenizer: Any,
    tasks: list[str],
    batch_size: int | str,
    trust_remote_code: bool,
    output_path: Optional[str],
        llm_backend: str,
    backend: BACKEND | str | None,
    model_args: Dict[str, Any],
    apply_chat_template: bool,
    gen_kwargs: Any,
    suite_kwargs: Dict[str, Any],
) -> dict[str, Any]:
    evalution = import_evalution()
    engine_config, model_config, session = _build_evalution_runtime(
        evalution=evalution,
        model_or_id_or_path=model_or_id_or_path,
        llm_backend=llm_backend,
        backend=backend,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        model_args=model_args,
        tokenizer=tokenizer,
    )
    suite_batch_size = _coerce_suite_batch_size(batch_size)
    generation_settings = _parse_generation_settings(gen_kwargs)

    try:
        test_results = []
        for index, task_name in enumerate(tasks):
            if index:
                session.gc()
            suite = _build_evalution_suite(
                evalution=evalution,
                task_name=task_name,
                apply_chat_template=apply_chat_template,
                batch_size=suite_batch_size,
                generation_settings=generation_settings,
                suite_kwargs=suite_kwargs,
            )
            test_results.append(suite.evaluate(session))

        engine_payload = {}
        if hasattr(engine_config, "to_dict"):
            engine_payload = engine_config.to_dict()
        try:
            engine_payload["execution"] = session.describe_execution()
        except Exception:
            # Best-effort metadata only; evaluation should continue if unavailable.
            pass

        result = evalution.RunResult(
            model=model_config.to_dict(),
            engine=engine_payload,
            tests=test_results,
        ).to_dict()
    finally:
        session.close()

    _maybe_write_evalution_output(output_path, result)
    return result


@dataclass(slots=True)
class _ArcChallengeLoglikelihoodSuite:
    apply_chat_template: bool = False
    batch_size: int | None = None
    dataset_path: str = "allenai/ai2_arc"
    dataset_name: str | None = "ARC-Challenge"
    split: str = "test"
    max_rows: int | None = None
    cache_dir: str | None = None
    stream: bool = True

    def dataset_loader(self) -> Any:
        from datasets import load_dataset

        def _loader(path: str, *args, stream: bool = True, **kwargs):
            # Evalution forwards `stream`; Hugging Face expects `streaming`.
            # Enforce the new API by rejecting legacy `streaming`.
            if "streaming" in kwargs:
                raise TypeError("use `stream=` (Evalution) not `streaming=`")
            return load_dataset(path, *args, streaming=stream, **kwargs)

        return _loader

    def task_name(self) -> str:
        return "arc_challenge"

    def continuation_for_choice(self, choice: str) -> str:
        return choice if choice[:1].isspace() else f" {choice}"

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "stream": self.stream,
            "apply_chat_template": self.apply_chat_template,
            "scoring_mode": "multiple_choice_loglikelihood",
        }

    def build_sample(self, doc: dict[str, Any], *, index: int) -> Any:
        from evalution.benchmarks.multiple_choice import MultipleChoiceSample
        from evalution.benchmarks.multiple_choice_utils import choice_index_from_labels, question_answer_prompt

        labels = list(doc["choices"]["label"])
        texts = list(doc["choices"]["text"])
        return MultipleChoiceSample(
            index=index,
            prompt=question_answer_prompt(doc["question"]),
            choices=texts,
            gold_index=choice_index_from_labels(labels, doc["answerKey"]),
            metadata={"id": doc["id"], "choice_labels": labels},
        )

    def evaluate(self, session: Any) -> Any:
        from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
        from evalution.engines.base import LoglikelihoodRequest
        from evalution.logbar import get_logger
        from evalution.results import SampleResult, TestResult

        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            stream=self.stream,
        )

        docs = limit_docs(loaded_docs, self.max_rows)
        if not isinstance(docs, list):
            docs = list(docs)

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        samples = [self.build_sample(doc, index=index) for index, doc in enumerate(docs)]
        rendered_prompts = [
            _render_evalution_prompt(session, sample.prompt, apply_chat_template=self.apply_chat_template)
            for sample in samples
        ]

        requests = []
        request_to_choice = []
        for sample, prompt in zip(samples, rendered_prompts, strict=True):
            for choice_index, choice in enumerate(sample.choices):
                requests.append(
                    LoglikelihoodRequest(
                        context=prompt,
                        continuation=self.continuation_for_choice(choice),
                    )
                )
                request_to_choice.append((sample.index, choice_index))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(samples), total)

        sample_choice_scores: dict[int, list[tuple[float, float, int]]] = {}
        for (sample_index, choice_index), output in zip(request_to_choice, outputs, strict=True):
            sample_choice_scores.setdefault(sample_index, []).append(
                (
                    output.logprob,
                    output.logprob / max(output.token_count, 1),
                    choice_index,
                )
            )

        sample_results = []
        raw_total = 0.0
        norm_total = 0.0
        for sample, prompt in zip(samples, rendered_prompts, strict=True):
            choice_scores = sorted(sample_choice_scores[sample.index], key=lambda item: item[2])
            raw_best = max(choice_scores, key=lambda item: item[0])[2]
            norm_best = max(choice_scores, key=lambda item: item[1])[2]
            raw_score = 1.0 if raw_best == sample.gold_index else 0.0
            norm_score = 1.0 if norm_best == sample.gold_index else 0.0
            raw_total += raw_score
            norm_total += norm_score
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt=prompt,
                    target=sample.choices[sample.gold_index],
                    prediction=sample.choices[norm_best],
                    extracted={
                        "gold_index": str(sample.gold_index),
                        "predicted_index": str(raw_best),
                        "predicted_index_norm": str(norm_best),
                    },
                    scores={
                        "accuracy,loglikelihood": raw_score,
                        "accuracy,loglikelihood_norm": norm_score,
                    },
                    metadata={
                        **sample.metadata,
                        "choice_logprobs": [score for score, _norm, _index in choice_scores],
                        "choice_logprobs_norm": [norm for _score, norm, _index in choice_scores],
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {
            "accuracy,loglikelihood": raw_total / denominator,
            "accuracy,loglikelihood_norm": norm_total / denominator,
        }
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )


def _result_tests(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    tests = result.get("tests")
    return list(tests) if isinstance(tests, list) else []


def _build_evalution_runtime(
    *,
    evalution: Any,
    model_or_id_or_path: Any,
        llm_backend: str,
    backend: BACKEND | str | None,
    batch_size: int | str,
    trust_remote_code: bool,
    model_args: Dict[str, Any],
    tokenizer: Any,
):
    from transformers import PreTrainedModel

    try:
        from peft import PeftModel
    except Exception:  # pragma: no cover - optional dependency
        PeftModel = ()

    engine_options, load_kwargs = _split_evalution_model_args(model_args)
    engine_dtype = _normalize_dtype_name(engine_options.get("dtype"))
    engine_device = engine_options.get("device")
    engine_device_map = engine_options.get("device_map")
    engine_attn = engine_options.get("attn_implementation")
    engine_padding_side = engine_options.get("padding_side", "left")

    tokenizer_path = _resolve_tokenizer_path(tokenizer)

    if llm_backend in {"vllm", "sglang"}:
        model_path = (
            model_or_id_or_path
            if isinstance(model_or_id_or_path, str)
            else _resolve_model_path(model_or_id_or_path)
        )
        if model_path is None:
            raise ValueError("Evalution vLLM evaluation requires a model path.")

        if llm_backend == "vllm":
            max_model_len = engine_options.get("max_model_len")
            tensor_parallel_size = engine_options.get("tensor_parallel_size", 1)
            gpu_memory_utilization = engine_options.get("gpu_memory_utilization", 0.9)
            llm_kwargs = dict(engine_options.get("llm_kwargs", {}) or {})

            engine = evalution.VLLM(
                dtype=engine_dtype,
                batch_size=batch_size,
                trust_remote_code=trust_remote_code,
                padding_side=engine_padding_side,
                seed=engine_options.get("seed"),
                tokenizer_mode=engine_options.get("tokenizer_mode", "auto"),
                tensor_parallel_size=int(tensor_parallel_size),
                gpu_memory_utilization=float(gpu_memory_utilization),
                quantization=engine_options.get("quantization"),
                max_model_len=int(max_model_len) if max_model_len is not None else None,
                enforce_eager=bool(engine_options.get("enforce_eager", False)),
                tokenizer_revision=engine_options.get("tokenizer_revision"),
                vllm_path=engine_options.get("vllm_path"),
                llm_kwargs=llm_kwargs,
            )
        else:
            sglang_config = _build_sglang_engine_kwargs(
                engine_options=engine_options,
                batch_size=batch_size,
                trust_remote_code=trust_remote_code,
                padding_side=engine_padding_side,
                engine_dtype=engine_dtype,
            )
            engine = evalution.SGLang(**sglang_config)

        model_config = evalution.Model(
            path=model_path,
            tokenizer_path=tokenizer_path,
            trust_remote_code=trust_remote_code,
            model_kwargs=load_kwargs,
        )
        session = engine.build(model_config)
        return engine, model_config, session

    if isinstance(model_or_id_or_path, str):
        engine = evalution.GPTQModel(
            dtype=engine_dtype,
            attn_implementation=engine_attn,
            device=engine_device,
            device_map=engine_device_map,
            seed=engine_options.get("seed"),
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            padding_side=engine_padding_side,
            backend=_normalize_backend_name(backend),
            gptqmodel_path=str(Path(__file__).resolve().parents[2]),
        )
        model_config = evalution.Model(
            path=model_or_id_or_path,
            tokenizer_path=tokenizer_path,
            trust_remote_code=trust_remote_code,
            model_kwargs=load_kwargs,
        )
        session = engine.build(model_config)
        return engine, model_config, session

    if isinstance(model_or_id_or_path, (PreTrainedModel, PeftModel)) or hasattr(model_or_id_or_path, "model"):
        model_path = _resolve_model_path(model_or_id_or_path)
        if model_path is None:
            raise ValueError("Evalution requires a model path when evaluating a live model instance.")

        engine = evalution.TransformersCompat(
            dtype=engine_dtype or _normalize_dtype_name(getattr(model_or_id_or_path, "dtype", None)),
            attn_implementation=engine_attn,
            device=engine_device,
            device_map=engine_device_map,
            seed=engine_options.get("seed"),
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            padding_side=engine_padding_side,
        )
        engine.resolved_engine = "TransformersCompat"

        model_config = evalution.Model(
            path=model_path,
            tokenizer_path=tokenizer_path,
            trust_remote_code=trust_remote_code,
            model_kwargs=load_kwargs,
        )
        session = _build_evalution_session_from_model(
            engine=engine,
            model_config=model_config,
            model=model_or_id_or_path,
        )
        return engine, model_config, session

    raise ValueError(
        f"`model_or_id_or_path` is invalid. expected: `model instance or str` actual: `{model_or_id_or_path}`"
    )


def _build_evalution_session_from_model(*, engine: Any, model_config: Any, model: Any):
    from evalution.engines.transformers_common import _clone_prepare_tokenizer, _resolve_input_device
    from evalution.engines.transformers_compat import TransformersCompatSession

    inner_model = getattr(model, "model", model)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Tokenizer must be attached to the loaded model instance for Evalution-backed evaluation.")

    trust_remote_code = (
        engine.trust_remote_code
        if engine.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    requested_attn = (
        getattr(getattr(inner_model, "config", None), "_attn_implementation", None)
        or getattr(getattr(inner_model, "config", None), "attn_implementation", None)
        or engine.attn_implementation
    )
    prepare_tokenizer = _clone_prepare_tokenizer(
        tokenizer=tokenizer,
        model_config=model_config,
        trust_remote_code=trust_remote_code,
    )
    requested_padding_side = getattr(engine, "padding_side", None)
    if requested_padding_side:
        for tok in (tokenizer, prepare_tokenizer):
            if tok is None:
                continue
            if getattr(tok, "padding_side", None) != requested_padding_side:
                tok.padding_side = requested_padding_side
            if getattr(tok, "pad_token_id", None) is None:
                eos_token_id = getattr(tok, "eos_token_id", None)
                if eos_token_id is not None:
                    tok.pad_token_id = eos_token_id

    session = TransformersCompatSession(
        config=engine,
        model_config=model_config,
        model=inner_model,
        tokenizer=tokenizer,
        prepare_tokenizer=prepare_tokenizer,
        input_device=_resolve_input_device(inner_model, prefer=engine.device),
        requested_attn_implementation=requested_attn,
        effective_attn_implementation=requested_attn,
        paged_attention_enabled=False,
        generation_backend="generate_compat",
    )
    session._gptqmodel_wrapper = model
    return session


def _build_evalution_suite(
    *,
    evalution: Any,
    task_name: str,
    apply_chat_template: bool,
    batch_size: int | None,
    generation_settings: Dict[str, Any],
    suite_kwargs: Dict[str, Any],
):
    benchmarks = evalution.benchmarks
    normalized_task = normalize_eval_task_name(task_name)
    max_new_tokens = int(generation_settings.get("max_new_tokens", 256))
    do_sample = bool(generation_settings.get("do_sample", False))
    temperature = float(generation_settings.get("temperature", 0.0))
    kwargs = dict(suite_kwargs or {})
    if "stream" not in kwargs and "streaming" in kwargs:
        kwargs["stream"] = bool(kwargs.pop("streaming"))
    elif "streaming" in kwargs:
        kwargs.pop("streaming")

    if normalized_task == "arc_challenge":
        kwargs.setdefault("apply_chat_template", apply_chat_template)
        kwargs.setdefault("batch_size", batch_size)
        kwargs.pop("num_fewshot", None)
        return _ArcChallengeLoglikelihoodSuite(**kwargs)
    if normalized_task == "mmlu_stem":
        kwargs.setdefault("subsets", "stem")
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("stream", True)
        if _MMLU_LOCAL_DATASET.exists():
            kwargs.setdefault("dataset_path", str(_MMLU_LOCAL_DATASET))
        return benchmarks.mmlu(**kwargs)
    if normalized_task == "gsm8k_cot":
        kwargs.setdefault("variant", "cot")
        kwargs.setdefault("apply_chat_template", apply_chat_template)
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("do_sample", do_sample)
        kwargs.setdefault("temperature", temperature)
        if _GSM8K_LOCAL_DATASET.exists():
            kwargs.setdefault("dataset_path", str(_GSM8K_LOCAL_DATASET))
            kwargs.setdefault("dataset_name", "main")
        return benchmarks.gsm8k(**kwargs)
    if normalized_task == "gsm8k_platinum_cot":
        kwargs.setdefault("variant", "cot")
        kwargs.setdefault("apply_chat_template", apply_chat_template)
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("do_sample", do_sample)
        kwargs.setdefault("temperature", temperature)
        return benchmarks.gsm8k_platinum(**kwargs)
    if normalized_task == "mmlu":
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("stream", True)
        if _MMLU_LOCAL_DATASET.exists():
            kwargs.setdefault("dataset_path", str(_MMLU_LOCAL_DATASET))
        return benchmarks.mmlu(**kwargs)
    if normalized_task == "mmlu_pro":
        kwargs.setdefault("apply_chat_template", apply_chat_template)
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("do_sample", do_sample)
        kwargs.setdefault("temperature", temperature)
        return benchmarks.mmlu_pro(**kwargs)
    if normalized_task.startswith("mmlu_pro:"):
        subset = normalized_task.split(":", 1)[1].strip()
        if not subset:
            raise ValueError(f"Invalid Evalution task: `{normalized_task}`")
        kwargs.setdefault("subsets", subset)
        kwargs.setdefault("apply_chat_template", apply_chat_template)
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("do_sample", do_sample)
        kwargs.setdefault("temperature", temperature)
        return benchmarks.mmlu_pro(**kwargs)

    generic_factory = getattr(benchmarks, normalized_task, None)
    if callable(generic_factory):
        kwargs.setdefault("batch_size", batch_size)
        return generic_factory(**kwargs)

    raise ValueError(f"Unsupported Evalution task: `{normalized_task}`")


def _split_evalution_model_args(model_args: Dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    engine_options = {}
    load_kwargs = {}
    for key, value in model_args.items():
        if key in _DROPPED_MODEL_ARG_KEYS:
            continue
        if key in _ENGINE_OPTION_KEYS:
            engine_options[key] = value
        else:
            load_kwargs[key] = value
    return engine_options, load_kwargs

def _build_sglang_engine_kwargs(
    *,
    engine_options: Dict[str, Any],
    batch_size: int | str,
    trust_remote_code: bool,
    padding_side: str,
    engine_dtype: str | None,
) -> Dict[str, Any]:
    context_length = engine_options.get("context_length", engine_options.get("max_model_len"))
    tp_size = engine_options.get("tp_size", engine_options.get("tensor_parallel_size", 1))
    mem_fraction_static = engine_options.get(
        "mem_fraction_static",
        engine_options.get("gpu_memory_utilization"),
    )

    return {
        "dtype": engine_dtype,
        "device": engine_options.get("device"),
        "seed": engine_options.get("seed"),
        "trust_remote_code": trust_remote_code,
        "padding_side": padding_side,
        "base_url": engine_options.get("base_url"),
        "batch_size": batch_size,
        "tokenizer_mode": engine_options.get("tokenizer_mode", "auto"),
        "tokenizer_worker_num": int(engine_options.get("tokenizer_worker_num", 1)),
        "skip_tokenizer_init": bool(engine_options.get("skip_tokenizer_init", False)),
        "load_format": engine_options.get("load_format", "auto"),
        "context_length": int(context_length) if context_length is not None else None,
        "quantization": engine_options.get("quantization"),
        "mem_fraction_static": float(mem_fraction_static) if mem_fraction_static is not None else None,
        "tp_size": int(tp_size),
        "dp_size": int(engine_options.get("dp_size", 1)),
        "pp_size": int(engine_options.get("pp_size", 1)),
        "attention_backend": engine_options.get("attention_backend"),
        "sampling_backend": engine_options.get("sampling_backend"),
        "max_running_requests": (
            int(engine_options["max_running_requests"])
            if engine_options.get("max_running_requests") is not None
            else None
        ),
        "max_total_tokens": (
            int(engine_options["max_total_tokens"])
            if engine_options.get("max_total_tokens") is not None
            else None
        ),
        "sampling_params": dict(engine_options.get("sampling_params", {}) or {}),
    }


def _parse_generation_settings(gen_kwargs: Any) -> Dict[str, Any]:
    if not gen_kwargs:
        return {}
    if isinstance(gen_kwargs, Mapping):
        return dict(gen_kwargs)

    settings = {}
    for item in str(gen_kwargs).split(","):
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        settings[key.strip()] = _coerce_scalar(raw_value.strip())
    return settings


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _coerce_suite_batch_size(batch_size: int | str) -> int | None:
    if isinstance(batch_size, str):
        normalized = batch_size.strip().lower()
        if normalized == "auto":
            return None
        return int(normalized)
    return int(batch_size)


def _normalize_backend_name(backend: BACKEND | str | None) -> str:
    if backend is None:
        return BACKEND.AUTO.value
    if isinstance(backend, BACKEND):
        return backend.value
    return str(backend)


def _normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return dtype
    try:
        import torch

        mapping = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
            torch.float64: "float64",
        }
        if dtype in mapping:
            return mapping[dtype]
    except ImportError:
        # torch is optional here; fall back to returning str(dtype) below.
        pass
    return str(dtype)


def _resolve_tokenizer_path(tokenizer: Any) -> str | None:
    if tokenizer is None:
        return None
    if isinstance(tokenizer, str):
        return tokenizer
    return getattr(tokenizer, "name_or_path", None)


def _resolve_model_path(model: Any) -> str | None:
    model_path = getattr(model, "model_local_path", None)
    if isinstance(model_path, str) and model_path.strip():
        return model_path
    config = getattr(model, "config", None)
    name_or_path = getattr(config, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path.strip():
        return name_or_path
    return None


def _render_evalution_prompt(session: Any, prompt: str, *, apply_chat_template: bool) -> str:
    if not apply_chat_template:
        return prompt

    tokenizer = getattr(session, "prepare_tokenizer", None) or getattr(session, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt
    return rendered if isinstance(rendered, str) and rendered.strip() else prompt


def _maybe_write_evalution_output(output_path: Optional[str], result: Mapping[str, Any]) -> None:
    if not output_path:
        return

    path = Path(output_path)
    if path.suffix.lower() != ".json":
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_TASKS",
    "SUPPORTED_TASKS",
    "evaluate",
    "format_eval_result_table",
    "get_eval_task_metrics",
    "get_eval_task_results",
    "import_evalution",
    "list_supported_tasks",
    "normalize_eval_task_name",
    "resolve_eval_metric_alias",
]
