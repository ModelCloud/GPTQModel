# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, Union

from tabulate import tabulate

from .backend import BACKEND
from .evalplus import patch_evalplus


try:
    from enum import EnumType
except ImportError:
    EnumType = type(Enum)


_EVALUTION_ROOT = Path(os.environ.get("GPTQMODEL_EVALUTION_PATH", "/root/Evalution")).expanduser()
_MMLU_LOCAL_DATASET = Path("/monster/data/model/dataset/hails-mmlu_no_train")
_GSM8K_LOCAL_DATASET = Path("/monster/data/model/dataset/gsm8k")
_ENGINE_OPTION_KEYS = {
    "attn_implementation",
    "device",
    "device_map",
    "dtype",
    "padding_side",
    "trust_remote_code",
}
_DROPPED_MODEL_ARG_KEYS = {
    "backend",
    "gptqmodel",
    "model_id_or_path",
    "pretrained",
    "tokenizer",
}


class EVAL:
    class LM_EVAL(str, Enum):
        ARC_CHALLENGE = "arc_challenge"
        GSM8K_COT = "gsm8k_cot"
        GSM8K_PLATINUM_COT = "gsm8k_platinum_cot"
        HELLASWAG = "hellaswag"
        MMLU = "mmlu"
        MMLU_STEM = "mmlu_stem"
        GPQA = "gpqa"
        ARC_EASY = "arc_easy"
        BOOLQ = "boolq"
        OPENBOOKQA = "openbookqa"

    class EVALPLUS(str, Enum):
        HUMAN = "humaneval"
        MBPP = "mbpp"

    class MMLU_PRO(str, Enum):
        BIOLOGY = "biology"
        BUSINESS = "business"
        CHEMISTRY = "chemistry"
        COMPUTER_SCIENCE = "computer science"
        ECONOMICS = "economics"
        ENGINEERING = "engineering"
        HEALTH = "health"
        HISTORY = "history"
        LAW = "law"
        MATH = "math"
        OTHER = "other"
        PHILOSOPHY = "philosophy"
        PHYSICS = "physics"
        PSYCHOLOGY = "psychology"

    @classmethod
    def get_tasks_for_framework(cls, framework: Union[str, Type[Enum]]) -> list:
        if isinstance(framework, EnumType):
            framework = framework.__name__

        if not hasattr(cls, framework):
            raise ValueError(f"No such EVAL framework: `{framework}`")

        enum_class = getattr(cls, framework)
        return list(enum_class)

    @classmethod
    def get_task_enums(cls):
        task_lists = []
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, type) and issubclass(attr, Enum):
                task_lists.extend(list(attr))
        return task_lists

    @classmethod
    def get_full_name(cls, member):
        return f"{cls.__name__}.{member.__class__.__name__}.{member.name}"

    @classmethod
    def get_all_tasks_string(cls):
        full_names = []
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, type) and issubclass(attr, Enum):
                full_names.extend(cls.get_full_name(member) for member in attr)
        return ', '.join(full_names)

    @classmethod
    def get_task_groups_from_tasks(cls, tasks: Union[str, List[str]]) -> Dict[Type[Enum], List[str]]:
        """Group tasks by their evaluation framework.

        Args:
            tasks: Either a single task name or list of task names

        Returns:
            Dictionary mapping framework enum classes to lists of tasks
            Example: {EVAL.LM_EVAL: ["arc_challenge", "hellaswag"], EVAL.EVALPLUS: ["humaneval"]}

        Raises:
            ValueError: If any task doesn't match a known framework
        """
        if isinstance(tasks, str):
            tasks = [tasks]

        # Create a mapping of task values to their enum classes
        task_to_framework = {}

        # Populate the mapping for all frameworks
        for framework in [cls.LM_EVAL, cls.EVALPLUS, cls.MMLU_PRO]:
            for task in framework:
                task_to_framework[task.value] = framework

        # Group tasks by their framework
        task_groups = {}
        unknown_tasks = []

        for task in tasks:
            if task in task_to_framework:
                framework = task_to_framework[task]
                if framework not in task_groups:
                    task_groups[framework] = []
                task_groups[framework].append(task)
            else:
                unknown_tasks.append(task)

        if unknown_tasks:
            raise ValueError(f"Unknown tasks: {unknown_tasks}")

        return task_groups


def import_evalution():
    try:
        return importlib.import_module("evalution")
    except ModuleNotFoundError:
        if _EVALUTION_ROOT.exists():
            root = str(_EVALUTION_ROOT)
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            return importlib.import_module("evalution")
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Evalution is required for framework=EVAL.LM_EVAL. "
                f"Expected a local checkout at `{_EVALUTION_ROOT}` or an installed `evalution` package."
            ) from exc


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
    return get_eval_task_results(result).get(_task_name(task), {})


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


def run_evalution_lm_eval(
    *,
    model_or_id_or_path: Any,
    tasks: list[Any],
    batch_size: int | str,
    trust_remote_code: bool,
    output_path: Optional[str],
    llm_backend: str,
    backend: BACKEND | str | None,
    model_args: Optional[Dict[str, Any]],
    tokenizer: Any,
    apply_chat_template: bool,
    gen_kwargs: Any,
    suite_kwargs: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    if llm_backend != "gptqmodel":
        raise ValueError("Evalution-backed framework=EVAL.LM_EVAL only supports llm_backend='gptqmodel'.")

    evalution = import_evalution()
    engine_config, model_config, session = _build_evalution_runtime(
        evalution=evalution,
        model_or_id_or_path=model_or_id_or_path,
        backend=backend,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        model_args=model_args or {},
        tokenizer=tokenizer,
    )
    suite_batch_size = _coerce_suite_batch_size(batch_size)
    generation_settings = _parse_generation_settings(gen_kwargs)

    try:
        test_results = []
        for index, task in enumerate(tasks):
            if index:
                session.gc()
            suite = _build_evalution_suite(
                evalution=evalution,
                task=task,
                apply_chat_template=apply_chat_template,
                batch_size=suite_batch_size,
                generation_settings=generation_settings,
                suite_kwargs=suite_kwargs or {},
            )
            test_results.append(suite.evaluate(session))

        engine_payload = {}
        if hasattr(engine_config, "to_dict"):
            engine_payload = engine_config.to_dict()
        try:
            engine_payload["execution"] = session.describe_execution()
        except Exception:
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


def evaluate(
    model_or_id_or_path: Any = None,
    tokenizer: Any = None,
    tasks: Any = None,
    framework: Any = EVAL.LM_EVAL,
    batch_size: int | str = 1,
    trust_remote_code: bool = False,
    output_path: Optional[str] = None,
    llm_backend: str = "gptqmodel",
    backend: BACKEND | str = BACKEND.AUTO,
    random_seed: int = 1234,
    model_args: Optional[Dict[str, Any]] = None,
    ntrain: int = 1,
    **args,
):
    import torch
    from tokenicer import Tokenicer
    from transformers import PreTrainedModel

    from ..models.base import BaseQModel
    from .hf import resolve_trust_remote_code

    try:
        from peft import PeftModel
    except Exception:  # pragma: no cover - optional dependency
        PeftModel = ()

    if isinstance(model_or_id_or_path, str):
        trust_remote_code = resolve_trust_remote_code(
            model_or_id_or_path,
            trust_remote_code=trust_remote_code,
        )

    if model_args is None:
        model_args = {}
    else:
        model_args = dict(model_args)

    if tasks is None:
        if framework == EVAL.LM_EVAL:
            tasks = [EVAL.LM_EVAL.ARC_CHALLENGE]
        elif framework == EVAL.MMLU_PRO:
            tasks = [EVAL.MMLU_PRO.MATH]
        else:
            tasks = [EVAL.EVALPLUS.HUMAN]
    elif not isinstance(tasks, list):
        tasks = [tasks]

    if framework is None:
        raise ValueError("Eval parameter: `framework` cannot be set to None")

    if not isinstance(tasks, list):
        raise ValueError("Eval parameter: `tasks` must be of List type")

    if llm_backend not in ["gptqmodel", "vllm"]:
        raise ValueError("Eval framework support llm_backend: [gptqmodel, vllm]")

    if llm_backend == "vllm":
        if "tensor_parallel_size" not in model_args:
            try:
                cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except Exception:
                cuda_devices = 0
            if cuda_devices:
                model_args["tensor_parallel_size"] = cuda_devices
        if "gpu_memory_utilization" not in model_args:
            model_args["gpu_memory_utilization"] = 0.90

    if framework == EVAL.LM_EVAL:
        for task in tasks:
            if task not in EVAL.get_task_enums():
                raise ValueError(
                    f"Eval.lm_eval supported `tasks`: `{EVAL.get_all_tasks_string()}`, actual = `{task}`"
                )

        gen_kwargs = args.pop("gen_kwargs", None)
        if gen_kwargs is None:
            gen_kwargs = "temperature=0.0,top_k=50"

        apply_chat_template = args.pop("apply_chat_template", False)
        suite_kwargs = args.pop("suite_kwargs", None)

        return run_evalution_lm_eval(
            model_or_id_or_path=model_or_id_or_path,
            tasks=tasks,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            output_path=output_path,
            llm_backend=llm_backend,
            backend=backend,
            model_args=model_args,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            gen_kwargs=gen_kwargs,
            suite_kwargs=suite_kwargs,
        )

    from ..models.auto import GPTQModel

    if isinstance(model_or_id_or_path, str):
        load_backend = backend
        disallowed_keys = {"pretrained", "tokenizer", "gptqmodel", "trust_remote_code", "backend", "model_id_or_path"}
        load_kwargs = {k: v for k, v in model_args.items() if k not in disallowed_keys}
        model = GPTQModel.load(
            model_id_or_path=model_or_id_or_path,
            backend=load_backend,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
        model_id_or_path = model_or_id_or_path
    elif isinstance(model_or_id_or_path, BaseQModel) or isinstance(model_or_id_or_path, (PreTrainedModel, PeftModel)):
        model = model_or_id_or_path
        model_id_or_path = model.config.name_or_path
    else:
        raise ValueError(
            f"`model_or_id_or_path` is invalid. expected: `model instance or str` actual: `{model_or_id_or_path}`"
        )

    if tokenizer is None:
        if isinstance(model, BaseQModel):
            tokenizer = model.tokenizer
        elif isinstance(model, PreTrainedModel) or (isinstance(model_id_or_path, str) and model_id_or_path.strip()):
            tokenizer = Tokenicer.load(model_id_or_path.strip())

    if tokenizer is None:
        raise ValueError(
            "Tokenizer: Auto-loading of tokenizer failed with `model_or_id_or_path`. Please pass in `tokenizer` as argument."
        )

    if llm_backend == "gptqmodel":
        model_args["tokenizer"] = tokenizer

    if framework == EVAL.EVALPLUS:
        for task in tasks:
            if task not in EVAL.get_task_enums():
                raise ValueError(f"evalplus support tasks: {EVAL.get_all_tasks_string()}")

        results = {}
        for task in tasks:
            base_formatted, plus_formatted, result_path = evalplus(
                model=model_id_or_path,
                dataset=task.value,
                batch=batch_size,
                trust_remote_code=trust_remote_code,
                output_file=output_path,
                backend=llm_backend,
            )
            results[task.value] = {
                "base tests": base_formatted,
                "base + extra tests": plus_formatted,
                "results_path": result_path,
            }
        return results

    if framework == EVAL.MMLU_PRO:
        for task in tasks:
            if task not in EVAL.get_task_enums():
                raise ValueError(f"eval support tasks: {EVAL.get_all_tasks_string()}")

        from .mmlupro import mmlupro

        selected_subjects = ",".join(_task_name(task) for task in tasks)
        return mmlupro(
            model,
            tokenizer,
            save_dir=output_path,
            seed=random_seed,
            selected_subjects=selected_subjects,
            ntrain=ntrain,
            batch_size=batch_size,
            max_samples=args.pop("max_samples", None),
        )

    raise ValueError("Eval framework support: EVAL.LM_EVAL, EVAL.EVALPLUS, EVAL.MMLUPRO")


@dataclass(slots=True)
class _ArcChallengeLoglikelihoodSuite:
    apply_chat_template: bool = False
    batch_size: int | None = None
    dataset_path: str = "allenai/ai2_arc"
    dataset_name: str | None = "ARC-Challenge"
    split: str = "test"
    max_rows: int | None = None
    cache_dir: str | None = None
    streaming: bool = False

    def dataset_loader(self) -> Any:
        from datasets import load_dataset

        return load_dataset

    def task_name(self) -> str:
        return "arc_challenge"

    def continuation_for_choice(self, choice: str) -> str:
        return choice if choice[:1].isspace() else f" {choice}"

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
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
        from evalution.engines.base import LoglikelihoodRequest
        from evalution.logbar import get_logger
        from evalution.results import SampleResult, TestResult
        from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset

        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
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


def _task_name(task: Any) -> str:
    return getattr(task, "value", task)


def _build_evalution_runtime(
    *,
    evalution: Any,
    model_or_id_or_path: Any,
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

    if isinstance(model_or_id_or_path, str):
        engine = evalution.GPTQModel(
            dtype=engine_dtype,
            attn_implementation=engine_attn,
            device=engine_device,
            device_map=engine_device_map,
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
            raise ValueError(
                "Evalution-backed framework=EVAL.LM_EVAL requires a model path when evaluating a live model instance."
            )

        engine = evalution.TransformersCompat(
            dtype=engine_dtype or _normalize_dtype_name(getattr(model_or_id_or_path, "dtype", None)),
            attn_implementation=engine_attn,
            device=engine_device,
            device_map=engine_device_map,
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
            evalution=evalution,
            engine=engine,
            model_config=model_config,
            model=model_or_id_or_path,
        )
        return engine, model_config, session

    raise ValueError(
        f"`model_or_id_or_path` is invalid. expected: `model instance or str` actual: `{model_or_id_or_path}`"
    )


def _build_evalution_session_from_model(*, evalution: Any, engine: Any, model_config: Any, model: Any):
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
    session = TransformersCompatSession(
        config=engine,
        model_config=model_config,
        model=inner_model,
        tokenizer=tokenizer,
        prepare_tokenizer=_clone_prepare_tokenizer(
            tokenizer=tokenizer,
            model_config=model_config,
            trust_remote_code=trust_remote_code,
        ),
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
    task: Any,
    apply_chat_template: bool,
    batch_size: int | None,
    generation_settings: Dict[str, Any],
    suite_kwargs: Dict[str, Any],
):
    benchmarks = evalution.benchmarks
    task_name = _task_name(task)
    max_new_tokens = int(generation_settings.get("max_new_tokens", 256))
    do_sample = bool(generation_settings.get("do_sample", False))
    temperature = float(generation_settings.get("temperature", 0.0))
    suite_options = dict(suite_kwargs or {})

    if task_name == EVAL.LM_EVAL.ARC_CHALLENGE.value:
        kwargs = {
            "apply_chat_template": apply_chat_template,
            "batch_size": batch_size,
        }
        kwargs.update(suite_options)
        return _ArcChallengeLoglikelihoodSuite(
            **kwargs,
        )
    if task_name == EVAL.LM_EVAL.ARC_EASY.value:
        kwargs = {"batch_size": batch_size}
        kwargs.update(suite_options)
        return benchmarks.arc_easy(**kwargs)
    if task_name == EVAL.LM_EVAL.BOOLQ.value:
        kwargs = {"batch_size": batch_size}
        kwargs.update(suite_options)
        return benchmarks.boolq(**kwargs)
    if task_name == EVAL.LM_EVAL.HELLASWAG.value:
        kwargs = {"batch_size": batch_size}
        kwargs.update(suite_options)
        return benchmarks.hellaswag(**kwargs)
    if task_name == EVAL.LM_EVAL.OPENBOOKQA.value:
        kwargs = {"batch_size": batch_size}
        kwargs.update(suite_options)
        return benchmarks.openbookqa(**kwargs)
    if task_name == EVAL.LM_EVAL.MMLU.value:
        kwargs = {"batch_size": batch_size}
        if _MMLU_LOCAL_DATASET.exists():
            kwargs["dataset_path"] = str(_MMLU_LOCAL_DATASET)
        kwargs.update(suite_options)
        return benchmarks.mmlu(**kwargs)
    if task_name == EVAL.LM_EVAL.MMLU_STEM.value:
        kwargs = {"subsets": "stem", "batch_size": batch_size}
        if _MMLU_LOCAL_DATASET.exists():
            kwargs["dataset_path"] = str(_MMLU_LOCAL_DATASET)
        kwargs.update(suite_options)
        return benchmarks.mmlu(**kwargs)
    if task_name == EVAL.LM_EVAL.GSM8K_COT.value:
        kwargs = {
            "variant": "cot",
            "apply_chat_template": apply_chat_template,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "do_sample": do_sample,
            "temperature": temperature,
        }
        if _GSM8K_LOCAL_DATASET.exists():
            kwargs["dataset_path"] = str(_GSM8K_LOCAL_DATASET)
            kwargs["dataset_name"] = "main"
        kwargs.update(suite_options)
        return benchmarks.gsm8k(**kwargs)
    if task_name == EVAL.LM_EVAL.GSM8K_PLATINUM_COT.value:
        kwargs = {
            "variant": "cot",
            "apply_chat_template": apply_chat_template,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "do_sample": do_sample,
            "temperature": temperature,
        }
        kwargs.update(suite_options)
        return benchmarks.gsm8k_platinum(**kwargs)
    if task_name == EVAL.LM_EVAL.GPQA.value:
        raise ValueError("Evalution does not currently provide a GPQA suite.")

    raise ValueError(f"Unsupported Evalution task: `{task_name}`")


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
    except Exception:
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


def evalplus(
        model,
        dataset: str,
        batch: int = 1,
        trust_remote_code: bool = False,
        output_file: Optional[str] = None,
        backend: str = 'gptqmodel'
):
    patch_evalplus(model)

    try:
        from evalplus.evaluate import evaluate
    except BaseException:
        raise ValueError("evalplus is not installed. Please install via `pip install gptqmodel[evalplus]`.")

    assert dataset in ["humaneval", "mbpp"], f"Invalid dataset {dataset}"

    evaluate(dataset=dataset, model=model, backend=backend, bs=batch, trust_remote_code=trust_remote_code, output_file=output_file,
             greedy=True)

    if output_file is None:
        base_name = model.strip("./").replace("/", "--") + "_gptqmodel_temp_0.0"
        legacy_file = os.path.join("evalplus_results", dataset, base_name + "_eval_results.json")
        new_file = os.path.join("evalplus_results", dataset, base_name + ".eval_results.json")
        # Check legacy format first for backwards compatibility, then new format
        output_file = legacy_file if os.path.exists(legacy_file) else new_file

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"No such file: {output_file}")

    try:
        with open(output_file, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON: {output_file}")

    try:
        pass_at_k = data["pass_at_k"]
        base = float(pass_at_k["base"]["pass@1"])
        plus = float(pass_at_k["plus"]["pass@1"])

        base_formatted = format(base, ".3f")
        plus_formatted = format(plus, ".3f")
    except KeyError as e:
        raise ValueError(f"Required key not found in JSON: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Data format error: {str(e)}")

    return base_formatted, plus_formatted, output_file


def evalplus_make_table(results):
    print("|    Tasks    | base tests | base + extra tests |")
    print("|-------------|------------|--------------------|")
    for task, metrics in results.items():
        print(f"| {task} | {metrics['base tests']} | {metrics['base + extra tests']} |")
