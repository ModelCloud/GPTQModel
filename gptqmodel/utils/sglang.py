# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping, Optional

import torch
from transformers import AutoConfig


SGLANG_VERSION: Optional[str] = None
SGLANG_IMPORT_ERROR: Optional[Exception] = None
SGLANG_AVAILABLE = False
sgl = None

try:
    SGLANG_VERSION = version("sglang")
except PackageNotFoundError:
    pass

try:
    import sglang as _sgl

    sgl = _sgl
    SGLANG_AVAILABLE = True
except Exception as exc:
    SGLANG_IMPORT_ERROR = exc


def _sglang_unavailable_message() -> str:
    # A missing distribution raises ``ModuleNotFoundError(name="sglang")``.
    # A missing dependency while importing an installed distribution also
    # raises ModuleNotFoundError, but its name points at that dependency.
    missing_sglang = isinstance(SGLANG_IMPORT_ERROR, ModuleNotFoundError) and (
        getattr(SGLANG_IMPORT_ERROR, "name", None) == "sglang"
        or "No module named 'sglang'" in str(SGLANG_IMPORT_ERROR)
    )
    if SGLANG_VERSION is None and missing_sglang:
        return "SGLang is not installed. Please install via `pip install -U 'sglang[srt]'`."
    if SGLANG_IMPORT_ERROR is not None:
        return (
            f"SGLang {SGLANG_VERSION or 'with unknown version'} is installed but failed to import: "
            f"{type(SGLANG_IMPORT_ERROR).__name__}: {SGLANG_IMPORT_ERROR}."
        )
    return "SGLang is not installed. Please install via `pip install -U 'sglang[srt]'`."


SGLANG_INSTALL_HINT = _sglang_unavailable_message()
_ENGINE_MARKER = "_gptqmodel_uses_sglang_engine"
_MISSING = object()
_LEGACY_SGLANG_SAMPLING_PARAMS = frozenset(
    {
        "frequency_penalty",
        "ignore_eos",
        "json_schema",
        "max_tokens",
        "min_p",
        "min_tokens",
        "n",
        "presence_penalty",
        "regex",
        "stop",
        "stop_token_ids",
        "temperature",
        "top_k",
        "top_p",
    }
)


def _require_sglang() -> None:
    if not SGLANG_AVAILABLE:
        raise ValueError(SGLANG_INSTALL_HINT) from SGLANG_IMPORT_ERROR


def _normalize_dtype(dtype: Any) -> Any:
    if isinstance(dtype, torch.dtype):
        return str(dtype).removeprefix("torch.")
    if isinstance(dtype, str):
        return dtype.removeprefix("torch.")
    return dtype


def _move_alias(kwargs: dict[str, Any], source: str, target: str) -> None:
    if source not in kwargs:
        return
    if target in kwargs:
        raise ValueError(f"Pass only one of SGLang arguments `{source}` and `{target}`.")
    kwargs[target] = kwargs.pop(source)


def _normalize_sglang_engine_kwargs(kwargs: Mapping[str, Any], trust_remote_code: bool) -> dict[str, Any]:
    normalized = dict(kwargs)
    _move_alias(normalized, "tensor_parallel_size", "tp_size")
    _move_alias(normalized, "gpu_memory_utilization", "mem_fraction_static")
    _move_alias(normalized, "max_model_len", "context_length")
    _move_alias(normalized, "seed", "random_seed")

    if "enforce_eager" in normalized:
        enforce_eager = normalized.pop("enforce_eager")
        if "disable_cuda_graph" in normalized:
            raise ValueError("Pass only one of SGLang arguments `enforce_eager` and `disable_cuda_graph`.")
        normalized["disable_cuda_graph"] = bool(enforce_eager)

    dtype = normalized.get("dtype")
    if dtype is not None:
        normalized["dtype"] = _normalize_dtype(dtype)

    device = normalized.get("device")
    if isinstance(device, torch.device):
        if device.index is not None:
            normalized.setdefault("base_gpu_id", device.index)
        normalized["device"] = device.type
    elif isinstance(device, str) and ":" in device:
        try:
            parsed_device = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid SGLang device `{device}`.") from exc
        if parsed_device.index is not None:
            normalized.setdefault("base_gpu_id", parsed_device.index)
        normalized["device"] = parsed_device.type
    elif device == "rocm":
        normalized["device"] = "cuda"

    for key in ("base_gpu_id", "context_length", "random_seed", "tp_size"):
        if normalized.get(key) is not None:
            normalized[key] = int(normalized[key])
    if normalized.get("mem_fraction_static") is not None:
        normalized["mem_fraction_static"] = float(normalized["mem_fraction_static"])

    normalized.setdefault("trust_remote_code", trust_remote_code)
    return normalized


def load_model_by_sglang(
    model,
    trust_remote_code,
    config=None,
    **kwargs,
):
    _require_sglang()

    # from_quantized already loaded the config with its cache/revision and
    # offline options. Direct callers still get the historical convenience.
    hf_config = config
    if hf_config is None:
        hf_config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
    runtime_kwargs = _normalize_sglang_engine_kwargs(kwargs, trust_remote_code)
    engine_factory = getattr(sgl, "Engine", _MISSING)
    if engine_factory is not _MISSING:
        try:
            runtime = engine_factory(
                model_path=model,
                **runtime_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize SGLang Engine for `{model}`: {exc}") from exc
        setattr(runtime, _ENGINE_MARKER, True)
    else:
        try:
            runtime = sgl.Runtime(
                model_path=model,
                **runtime_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize legacy SGLang Runtime for `{model}`: {exc}") from exc
        setattr(runtime, _ENGINE_MARKER, False)
        set_default_backend = getattr(sgl, "set_default_backend", None)
        if callable(set_default_backend):
            set_default_backend(runtime)
    return runtime, hf_config


if SGLANG_AVAILABLE:

    @sgl.function
    def _legacy_generate(s, prompt, **kwargs):
        s += prompt
        s += sgl.gen("result", **kwargs)

else:

    def _legacy_generate(s, prompt, **kwargs):
        raise ValueError(SGLANG_INSTALL_HINT)


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


def _normalize_stop_token_ids(value: Any) -> list[int]:
    """Normalize SGLang/Hugging Face stop IDs to a de-duplicated list."""
    try:
        token_ids = _normalize_eos_token_ids(value)
    except TypeError as exc:
        raise TypeError("`stop_token_ids` must be an integer or a sequence of integers.") from exc
    return list(dict.fromkeys(token_ids))


def _build_sglang_sampling_params(value: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if value is None:
        sampling_params = {}
    elif isinstance(value, Mapping):
        sampling_params = dict(value)
    else:
        raise TypeError("SGLang `sampling_params` must be a mapping.")

    # SGLang's Engine API uses the *_new_tokens names. Normalize aliases in
    # user-provided sampling_params as well as top-level generate kwargs.
    _move_alias(sampling_params, "max_tokens", "max_new_tokens")
    _move_alias(sampling_params, "min_tokens", "min_new_tokens")
    _move_alias(sampling_params, "num_return_sequences", "n")

    if kwargs.get("max_length") is not None or sampling_params.get("max_length") is not None:
        raise ValueError("SGLang does not support argument `max_length`. Please use `max_new_tokens` instead.")
    if kwargs.get("min_length") is not None or sampling_params.get("min_length") is not None:
        raise ValueError("SGLang does not support argument `min_length`. Please use `min_new_tokens` instead.")
    if kwargs.get("max_new_tokens") is not None and kwargs.get("max_tokens") is not None:
        raise ValueError("Pass only one of SGLang arguments `max_new_tokens` and `max_tokens`.")
    if kwargs.get("min_new_tokens") is not None and kwargs.get("min_tokens") is not None:
        raise ValueError("Pass only one of SGLang arguments `min_new_tokens` and `min_tokens`.")

    field_map = {
        "num_return_sequences": "n",
        "repetition_penalty": "repetition_penalty",
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "min_p": "min_p",
        "max_new_tokens": "max_new_tokens",
        "max_tokens": "max_new_tokens",
        "min_new_tokens": "min_new_tokens",
        "min_tokens": "min_new_tokens",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "ignore_eos": "ignore_eos",
        "stop": "stop",
        "regex": "regex",
        "json_schema": "json_schema",
        "sampling_seed": "sampling_seed",
    }
    for source, target in field_map.items():
        if kwargs.get(source) is not None:
            sampling_params[target] = kwargs[source]

    stop_ids = []
    if "stop_token_ids" in sampling_params and sampling_params["stop_token_ids"] is not None:
        stop_ids.extend(_normalize_stop_token_ids(sampling_params["stop_token_ids"]))
    if kwargs.get("stop_token_ids") is not None:
        stop_ids.extend(_normalize_stop_token_ids(kwargs["stop_token_ids"]))
    if kwargs.get("eos_token_id") is not None:
        stop_ids.extend(_normalize_eos_token_ids(kwargs["eos_token_id"]))
    if stop_ids:
        sampling_params["stop_token_ids"] = list(dict.fromkeys(stop_ids))

    # do_sample=False is deterministic even when an explicit temperature was supplied.
    if kwargs.get("do_sample") is False:
        sampling_params["temperature"] = 0.0
    return sampling_params


def _coerce_token_batch(value: Any) -> tuple[list[list[int]], bool]:
    single_prompt = False
    if torch.is_tensor(value):
        single_prompt = value.ndim == 1
        value = value.detach().cpu().tolist()
    elif isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list) or not value:
        raise ValueError("Token prompts must be a non-empty tensor or list.")
    if all(isinstance(token, int) and not isinstance(token, bool) for token in value):
        return [list(value)], True

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
    return rows, single_prompt


def _apply_attention_mask(token_batch: list[list[int]], attention_mask: Any) -> list[list[int]]:
    if attention_mask is None:
        return token_batch
    if torch.is_tensor(attention_mask):
        attention_mask = attention_mask.detach().cpu().tolist()
    elif isinstance(attention_mask, tuple):
        attention_mask = list(attention_mask)
    if not isinstance(attention_mask, list):
        raise ValueError("`attention_mask` must be a sequence with the same batch size as `input_ids`.")
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
        if any(not isinstance(keep, (bool, int)) or keep not in (0, 1, False, True) for keep in mask):
            raise TypeError("`attention_mask` values must be boolean or 0/1 integers.")
        kept_positions = [index for index, keep in enumerate(mask) if bool(keep)]
        if not kept_positions:
            raise ValueError("`attention_mask` removed every token from a prompt.")
        first, last = kept_positions[0], kept_positions[-1]
        if any(not bool(mask[index]) for index in range(first, last + 1)):
            raise ValueError("`attention_mask` may only remove left or right padding, not tokens inside a prompt.")
        filtered = token_ids[first : last + 1]
        filtered_batch.append(filtered)
    return filtered_batch


def _normalize_sglang_inputs(prompts: Any, input_ids: Any, attention_mask: Any):
    if prompts is not None and input_ids is not None:
        raise ValueError("Pass only one of `prompts` or `input_ids`.")
    value = prompts if prompts is not None else input_ids
    if value is None:
        raise ValueError("Either prompts or input_ids must be provided.")

    if isinstance(value, str):
        if input_ids is not None:
            raise TypeError("`input_ids` cannot be a string.")
        if attention_mask is not None:
            raise ValueError("`attention_mask` is only supported with `input_ids`, not text prompts.")
        return value, None
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
        if input_ids is not None:
            raise TypeError("`input_ids` cannot contain strings.")
        if attention_mask is not None:
            raise ValueError("`attention_mask` is only supported with `input_ids`, not text prompts.")
        return list(value), None

    token_batch, single_prompt = _coerce_token_batch(value)
    token_batch = _apply_attention_mask(token_batch, attention_mask)
    return None, token_batch[0] if single_prompt else token_batch


def _extract_sglang_text(result: Any):
    if isinstance(result, Mapping):
        if "text" not in result:
            raise RuntimeError("SGLang generation result is missing the `text` field.")
        text = result["text"]
        if not isinstance(text, str):
            raise TypeError(f"SGLang generation result `text` must be a string, got {type(text)}.")
        return text
    if isinstance(result, list):
        return [_extract_sglang_text(item) for item in result]
    raise TypeError(f"Unexpected SGLang generation result type: {type(result)}.")


def _uses_engine_api(model: Any) -> bool:
    marker = getattr(model, _ENGINE_MARKER, None)
    if marker is not None:
        return bool(marker)
    return model.__class__.__module__.startswith("sglang.srt.entrypoints.")


def _legacy_sglang_sampling_params(sampling_params: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(sampling_params)
    _move_alias(normalized, "max_new_tokens", "max_tokens")
    _move_alias(normalized, "min_new_tokens", "min_tokens")

    try:
        signature = inspect.signature(sgl.gen)
        parameters = signature.parameters
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            # SGLang's decorated ``gen`` can expose only **kwargs even though
            # its legacy frontend has a finite set of supported fields.
            supported = set(_LEGACY_SGLANG_SAMPLING_PARAMS)
        else:
            supported = set(parameters)
    except (TypeError, ValueError):
        supported = set(_LEGACY_SGLANG_SAMPLING_PARAMS)
    unsupported = sorted(set(normalized) - supported)
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"The legacy SGLang Runtime frontend does not support sampling parameters: {names}.")
    return normalized


def _extract_legacy_sglang_result(state: Any) -> Any:
    # Runtime returns ProgramState, which supports item lookup but is not a Mapping.
    try:
        result = state["result"]
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Legacy SGLang generation state is missing the `result` field.") from exc
    if not isinstance(result, str):
        raise TypeError(f"Legacy SGLang generation result must be a string, got {type(result)}.")
    return result


@torch.inference_mode()
def sglang_generate(
    model,
    **kwargs,
):
    _require_sglang()

    prompts = kwargs.pop("prompts", None)
    input_ids = kwargs.pop("input_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    text_prompts, token_prompts = _normalize_sglang_inputs(prompts, input_ids, attention_mask)

    sampling_keys = {
        "do_sample",
        "eos_token_id",
        "frequency_penalty",
        "ignore_eos",
        "json_schema",
        "max_length",
        "max_new_tokens",
        "max_tokens",
        "min_length",
        "min_new_tokens",
        "min_p",
        "min_tokens",
        "num_return_sequences",
        "presence_penalty",
        "regex",
        "repetition_penalty",
        "sampling_seed",
        "stop",
        "stop_token_ids",
        "temperature",
        "top_k",
        "top_p",
    }
    sampling_params = _build_sglang_sampling_params(kwargs.pop("sampling_params", None), kwargs)
    request_kwargs = {key: value for key, value in kwargs.items() if key not in sampling_keys}
    request_kwargs.pop("pad_token_id", None)

    if _uses_engine_api(model):
        result = model.generate(
            prompt=text_prompts,
            input_ids=token_prompts,
            sampling_params=sampling_params,
            **request_kwargs,
        )
        return _extract_sglang_text(result)

    if token_prompts is not None:
        raise ValueError("The legacy SGLang Runtime frontend does not support `input_ids`; pass text prompts instead.")
    if request_kwargs:
        names = ", ".join(sorted(request_kwargs))
        raise ValueError(
            "The legacy SGLang Runtime frontend does not support request parameters: "
            f"{names}."
        )
    legacy_sampling_params = _legacy_sglang_sampling_params(sampling_params)
    if isinstance(text_prompts, list):
        # Runtime's historical frontend accepts one prompt per state.
        return [
            _extract_legacy_sglang_result(
                _legacy_generate.run(prompt=prompt, **legacy_sampling_params),
            )
            for prompt in text_prompts
        ]
    state = _legacy_generate.run(prompt=text_prompts, **legacy_sampling_params)
    return _extract_legacy_sglang_result(state)
