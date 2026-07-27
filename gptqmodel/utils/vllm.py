# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Mapping, Optional

import torch


VLLM_VERSION: Optional[str] = None
VLLM_IMPORT_ERROR: Optional[Exception] = None
VLLM_AVAILABLE = False

LLM = None
SamplingParams = None
TokensPrompt = None

try:
    VLLM_VERSION = version("vllm")
except PackageNotFoundError:
    pass

try:
    from vllm import LLM as _LLM
    from vllm import SamplingParams as _SamplingParams

    try:
        from vllm import TokensPrompt as _TokensPrompt
    except ImportError:
        try:
            from vllm.inputs import TokensPrompt as _TokensPrompt
        except ImportError:
            _TokensPrompt = None

    LLM = _LLM
    SamplingParams = _SamplingParams
    TokensPrompt = _TokensPrompt
    VLLM_AVAILABLE = True
except Exception as exc:
    VLLM_IMPORT_ERROR = exc


def _vllm_unavailable_message() -> str:
    if VLLM_VERSION is None and isinstance(VLLM_IMPORT_ERROR, ModuleNotFoundError):
        return "vLLM is not installed. Please install via `pip install -U vllm`."
    if VLLM_IMPORT_ERROR is not None:
        return (
            f"vLLM {VLLM_VERSION or 'with unknown version'} is installed but failed to import: "
            f"{type(VLLM_IMPORT_ERROR).__name__}: {VLLM_IMPORT_ERROR}. "
            "Check that vLLM's Python dependencies and its PyTorch/CUDA runtime build are mutually compatible."
        )
    return "vLLM is not installed. Please install via `pip install -U vllm`."


VLLM_INSTALL_HINT = _vllm_unavailable_message()


def _require_vllm() -> None:
    if not VLLM_AVAILABLE:
        raise ValueError(VLLM_INSTALL_HINT) from VLLM_IMPORT_ERROR


def _normalize_eos_token_ids(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return list(value)
    raise TypeError("`eos_token_id` must be an integer or a sequence of integers.")


# Returns SamplingParams but we cannot use this type hint since vLLM is optional.
def convert_hf_params_to_vllm(hf_params: Dict[str, Any]):
    _require_vllm()

    if hf_params.get("max_length") is not None:
        raise ValueError("vLLM does not support argument `max_length`. Please use `max_new_tokens` instead.")
    if hf_params.get("min_length") is not None:
        raise ValueError("vLLM does not support argument `min_length`. Please use `min_new_tokens` instead.")
    if hf_params.get("num_beams") not in (None, 1):
        raise ValueError(
            "GPT-QModel's vLLM generation adapter does not support Hugging Face beam search. "
            "Use vLLM's beam-search API directly."
        )
    if hf_params.get("do_sample") is False and (hf_params.get("num_return_sequences") or 1) > 1:
        raise ValueError(
            "vLLM requires sampling when `num_return_sequences` is greater than one. "
            "Set `do_sample=True` or request a single sequence."
        )

    field_map = {
        "num_return_sequences": "n",
        "repetition_penalty": "repetition_penalty",
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "min_p": "min_p",
        "max_new_tokens": "max_tokens",
        "max_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "min_tokens": "min_tokens",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "ignore_eos": "ignore_eos",
        "stop": "stop",
        "stop_token_ids": "stop_token_ids",
        "seed": "seed",
    }
    sampling_kwargs = {
        target: hf_params[source]
        for source, target in field_map.items()
        if hf_params.get(source) is not None
    }
    if hf_params.get("eos_token_id") is not None:
        eos_token_ids = _normalize_eos_token_ids(hf_params["eos_token_id"])
        existing_stop_ids = sampling_kwargs.get("stop_token_ids") or []
        sampling_kwargs["stop_token_ids"] = list(dict.fromkeys([*existing_stop_ids, *eos_token_ids]))
    if hf_params.get("do_sample") is False:
        # Hugging Face ignores temperature for greedy decoding. vLLM selects
        # greedy decoding by setting temperature to zero.
        sampling_kwargs["temperature"] = 0.0

    try:
        return SamplingParams(**sampling_kwargs)
    except TypeError as exc:
        requested = ", ".join(sorted(sampling_kwargs))
        raise ValueError(
            f"vLLM {VLLM_VERSION or 'unknown'} does not support the requested sampling parameters: {requested}."
        ) from exc


def load_model_by_vllm(
    model,
    **kwargs,
):
    _require_vllm()
    return LLM(
        model=model,
        **kwargs,
    )


def get_vllm_model_config(model):
    model_config = getattr(model, "model_config", None)
    if model_config is not None:
        return model_config

    engine = getattr(model, "llm_engine", None)
    if engine is None:
        raise AttributeError("vLLM LLM instance is missing `llm_engine`.")

    vllm_config = getattr(engine, "vllm_config", None) or getattr(model, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        model_config = getattr(engine, "model_config", None)
    if model_config is None:
        raise AttributeError("vLLM engine exposes neither `vllm_config.model_config` nor `model_config`.")
    return model_config


def get_vllm_device(model):
    engine = getattr(model, "llm_engine", None)
    if engine is None:
        return None

    vllm_config = getattr(engine, "vllm_config", None) or getattr(model, "vllm_config", None)
    device_config = getattr(vllm_config, "device_config", None)
    if device_config is None:
        device_config = getattr(engine, "device_config", None)
    return getattr(device_config, "device", None)


def _coerce_token_batch(value: Any) -> list[list[int]]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list) or not value:
        raise ValueError("Token prompts must be a non-empty tensor or list.")

    if all(isinstance(token, int) and not isinstance(token, bool) for token in value):
        return [list(value)]

    rows = []
    for row in value:
        if torch.is_tensor(row):
            row = row.detach().cpu().tolist()
        elif isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list) or not row:
            raise ValueError("Each token prompt must be a non-empty list of token IDs.")
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in row):
            raise TypeError("Token prompts may only contain integer token IDs.")
        rows.append(list(row))
    return rows


def _apply_attention_mask(token_batch: list[list[int]], attention_mask: Any) -> list[list[int]]:
    if attention_mask is None:
        return token_batch
    if torch.is_tensor(attention_mask):
        attention_mask = attention_mask.detach().cpu().tolist()
    elif isinstance(attention_mask, tuple):
        attention_mask = list(attention_mask)

    if len(token_batch) == 1 and isinstance(attention_mask, list) and attention_mask:
        if all(not isinstance(item, (list, tuple)) for item in attention_mask):
            attention_mask = [attention_mask]

    if not isinstance(attention_mask, list) or len(attention_mask) != len(token_batch):
        raise ValueError("`attention_mask` must have the same batch size as `input_ids`.")

    filtered_batch = []
    for token_ids, mask in zip(token_batch, attention_mask):
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().tolist()
        elif isinstance(mask, tuple):
            mask = list(mask)
        if not isinstance(mask, list) or len(mask) != len(token_ids):
            raise ValueError("Each `attention_mask` row must have the same length as its token prompt.")
        filtered = [token_id for token_id, keep in zip(token_ids, mask) if bool(keep)]
        if not filtered:
            raise ValueError("`attention_mask` removed every token from a prompt.")
        filtered_batch.append(filtered)
    return filtered_batch


def _normalize_vllm_inputs(prompts: Any, input_ids: Any, attention_mask: Any):
    if prompts is not None and input_ids is not None:
        raise ValueError("Pass only one of `prompts` or `input_ids`.")
    value = prompts if prompts is not None else input_ids
    if value is None:
        raise ValueError("Either prompts or input_ids must be provided.")

    if isinstance(value, str):
        if input_ids is not None:
            raise TypeError("`input_ids` cannot be a string.")
        return value, None
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
        if input_ids is not None:
            raise TypeError("`input_ids` cannot contain strings.")
        return list(value), None

    token_batch = _coerce_token_batch(value)
    return None, _apply_attention_mask(token_batch, attention_mask)


def _build_sampling_params(value: Any, kwargs: Mapping[str, Any]):
    if value is None:
        hf_keys = (
            "num_return_sequences",
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "max_length",
            "min_length",
            "max_new_tokens",
            "max_tokens",
            "min_new_tokens",
            "min_tokens",
            "eos_token_id",
            "frequency_penalty",
            "presence_penalty",
            "ignore_eos",
            "stop",
            "stop_token_ids",
            "seed",
            "do_sample",
            "num_beams",
        )
        hf_params = {key: kwargs[key] for key in hf_keys if kwargs.get(key) is not None}
        # GPT-QModel's public generate method follows Hugging Face semantics,
        # where sampling is disabled unless explicitly requested.
        hf_params.setdefault("do_sample", False)
        return convert_hf_params_to_vllm(hf_params)
    if isinstance(value, SamplingParams):
        return value
    if isinstance(value, Mapping):
        try:
            return SamplingParams(**dict(value))
        except TypeError as exc:
            raise ValueError("Invalid vLLM `sampling_params` mapping.") from exc
    raise TypeError("`sampling_params` must be a vLLM SamplingParams instance or a mapping.")


def _run_vllm_generation(model, text_prompts, token_batch, sampling_params, generation_kwargs):
    if token_batch is None:
        return model.generate(prompts=text_prompts, sampling_params=sampling_params, **generation_kwargs)
    if TokensPrompt is not None:
        token_prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in token_batch]
        return model.generate(prompts=token_prompts, sampling_params=sampling_params, **generation_kwargs)
    return model.generate(prompt_token_ids=token_batch, sampling_params=sampling_params, **generation_kwargs)


@torch.inference_mode()
def vllm_generate(model, **kwargs):
    _require_vllm()

    prompts = kwargs.pop("prompts", None)
    input_ids = kwargs.pop("input_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    text_prompts, token_batch = _normalize_vllm_inputs(prompts, input_ids, attention_mask)

    sampling_params = _build_sampling_params(kwargs.pop("sampling_params", None), kwargs)
    generate_keys = (
        "use_tqdm",
        "lora_request",
        "priority",
        "tokenization_kwargs",
        "mm_processor_kwargs",
    )
    generation_kwargs = {key: kwargs.pop(key) for key in generate_keys if key in kwargs}
    req_results = _run_vllm_generation(
        model,
        text_prompts,
        token_batch,
        sampling_params,
        generation_kwargs,
    )

    outputs = []
    for result in req_results:
        prompt_token_ids = list(result.prompt_token_ids)
        for output in result.outputs:
            outputs.append(prompt_token_ids + list(output.token_ids))
    if not outputs:
        return torch.empty((0, 0), dtype=torch.long)

    pad_token_id = kwargs.get("pad_token_id")
    if pad_token_id is None:
        tokenizer = model.get_tokenizer()
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    max_length = max(len(output) for output in outputs)
    padded = [output + [pad_token_id] * (max_length - len(output)) for output in outputs]
    return torch.tensor(padded, dtype=torch.long)
