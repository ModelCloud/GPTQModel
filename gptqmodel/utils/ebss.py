# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Expert-Balanced Self-Sampling (EBSS) for MoE calibration.

Clean-room implementation of Algorithm 1 in:

    Zhixuan Chen, Xing Hu, Dawei Yang, Zukang Xu, Xu Chen, Zhihang Yuan,
    Sifan Zhou, and Jiangyong Yu.  "MoEQuant: Enhancing Quantization for
    Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and
    Affinity Guidance."  ICML 2025, PMLR 267:8245-8260.
    https://arxiv.org/abs/2505.03804

The public MoEQuant implementation uses a greedy corpus filter for its
calibration set; this module follows the paper's beam search instead.  It
keeps one KV cache per candidate, so every step costs one single-token
forward per retained branch rather than a full-vocabulary forward.

The generated token IDs are fed directly to ``CalibrationConfig`` as
``List[List[int]]``.  Decoding and re-tokenizing would change the model's own
distribution and is deliberately avoided.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ..quantization.config import EBSSConfig
from .logger import setup_logger
from .model import (
    MOE_TOPK_FIELD_NAMES,
    _get_local_expert_count,
    find_moe_routing_modules,
)


log = setup_logger()

_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _router_key(name: str) -> str:
    """Group one router's usage by its owning MoE block.

    Router names are model-global paths.  Keying by the parent path keeps
    models with several routers in one decoder layer independent, because
    those routers may own different expert counts.  A model with one router
    per layer still produces one balance term per layer.
    """

    parent = name.rsplit(".", 1)[0] if "." in name else ""
    return parent or name


def _router_top_k(router: nn.Module, indices: Optional[torch.Tensor]) -> int:
    for field_name in MOE_TOPK_FIELD_NAMES:
        value = getattr(router, field_name, None)
        if isinstance(value, int) and value > 0:
            return value
    if indices is not None and indices.ndim >= 1 and indices.shape[-1] > 0:
        return int(indices.shape[-1])
    raise ValueError(
        f"EBSS: router `{type(router).__name__}` exposes neither a positive top-k "
        "field nor integer routing indices."
    )


def _router_num_experts(router: nn.Module, indices: Optional[torch.Tensor]) -> int:
    value = _get_local_expert_count(router)
    if value is not None:
        return value
    if indices is not None and indices.numel() > 0:
        return int(indices.max().item()) + 1
    raise ValueError(
        f"EBSS: router `{type(router).__name__}` exposes no expert count; "
        "cannot compute the expert-balance term."
    )


def _extract_router_indices(output: Any, router: nn.Module) -> torch.Tensor:
    """Return the router's hard top-k assignments from any supported shape.

    Ports on the Transformers 5.23 ABI return either ``(logits, scores,
    indices)`` or a plain logits tensor.  Integer tensors in a tuple are the
    authoritative assignments; a logits-only router is resolved with its own
    ``top_k``.  Anything else fails closed instead of guessing.
    """

    indices: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    if isinstance(output, dict):
        for key in ("indices", "top_k_index", "selected_experts"):
            value = output.get(key)
            if torch.is_tensor(value) and value.dtype in _INTEGER_DTYPES:
                indices = value
                break
        for key in ("logits", "router_logits"):
            value = output.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                logits = value
                break
    elif isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value) and value.dtype in _INTEGER_DTYPES:
                indices = value
            elif torch.is_tensor(value) and value.is_floating_point() and logits is None:
                logits = value
    elif torch.is_tensor(output):
        if output.dtype in _INTEGER_DTYPES:
            indices = output
        elif output.is_floating_point():
            logits = output

    if indices is None and logits is not None:
        top_k = _router_top_k(router, None)
        indices = torch.topk(logits, min(top_k, int(logits.shape[-1])), dim=-1).indices

    if indices is None:
        raise ValueError(
            f"EBSS: router `{type(router).__name__}` returned no integer routing "
            "indices and no logits; cannot compute expert usage."
        )
    if indices.ndim == 1:
        return indices.reshape(-1, 1)
    if indices.ndim == 2:
        return indices
    if indices.ndim == 3 and indices.shape[-1] == 1:
        return indices.squeeze(-1)
    if indices.ndim == 3 and indices.shape[1] == 1:
        return indices.squeeze(1)
    raise ValueError(
        "EBSS: unsupported router-index shape "
        f"{tuple(indices.shape)}; expected [tokens], [tokens, top_k], or a "
        "singleton trailing/leading expert axis."
    )


class _RoutingObserver:
    """Collect one forward pass's hard expert assignments, keyed by layer."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[Any] = []
        self._current: Dict[str, torch.Tensor] = {}
        self._num_experts: Dict[str, int] = {}

    def __enter__(self) -> "_RoutingObserver":
        routers = find_moe_routing_modules(self.model)
        if not routers:
            raise ValueError(
                "EBSS requires a MoE model with discoverable routing modules; "
                "none were found."
            )
        try:
            for router in routers:
                name = self._module_name(router)
                self._handles.append(router.register_forward_hook(self._make_hook(name, router)))
        except BaseException:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._current.clear()
        return False

    def _module_name(self, target: nn.Module) -> str:
        for name, module in self.model.named_modules():
            if module is target:
                return name
        return type(target).__name__

    def _make_hook(self, name: str, router: nn.Module):
        key = _router_key(name)

        def hook(_module, _inputs, output):
            indices = _extract_router_indices(output, router)
            num_experts = _router_num_experts(router, indices)
            previous = self._num_experts.get(key)
            if previous is not None and previous != num_experts:
                raise ValueError(
                    f"EBSS: routers in `{key}` disagree on the expert count "
                    f"({previous} vs {num_experts}); cannot build one balance term."
                )
            self._num_experts[key] = num_experts
            flat = indices.reshape(-1)
            flat = flat[(flat >= 0) & (flat < num_experts)]
            if flat.numel() == 0:
                counts = torch.zeros(num_experts, dtype=torch.int64)
            else:
                counts = torch.bincount(flat.to(device="cpu"), minlength=num_experts).to(torch.int64)
            existing = self._current.get(key)
            self._current[key] = counts if existing is None else existing + counts

        return hook

    def begin_forward(self) -> None:
        self._current = {}

    def end_forward(self) -> Dict[str, torch.Tensor]:
        result = {key: value.clone() for key, value in self._current.items()}
        self._current = {}
        return result

    @property
    def num_experts(self) -> Dict[str, int]:
        return dict(self._num_experts)


@dataclass
class _Candidate:
    tokens: List[int]
    log_prob_sum: float
    usage: Dict[str, torch.Tensor]
    cache: Any
    logits: torch.Tensor


def _merge_usage(
    base: Dict[str, torch.Tensor],
    extra: Dict[str, torch.Tensor],
    num_experts: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    merged = {key: value.clone() for key, value in base.items()}
    for key, counts in extra.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = counts.clone()
        else:
            merged[key] = existing + counts
    for key in num_experts:
        merged.setdefault(key, torch.zeros(num_experts[key], dtype=torch.int64))
    return merged


def _expert_balance(usage: Dict[str, torch.Tensor], num_experts: Dict[str, int]) -> float:
    """Return the paper's layer-averaged expert-balance term (Eq. 3-4)."""

    sigmas: List[float] = []
    for key, counts in usage.items():
        expert_count = int(num_experts.get(key, 0))
        total = int(counts.sum().item())
        if expert_count <= 1 or total <= 0:
            sigmas.append(0.0)
            continue
        frequencies = counts.to(dtype=torch.float64) / float(total)
        mean = 1.0 / float(expert_count)
        variance = torch.sum((frequencies - mean) ** 2) * (
            float(expert_count) / float(expert_count - 1)
        )
        sigmas.append(float(torch.sqrt(variance).item()))
    if not sigmas:
        return 0.0
    return sum(sigmas) / float(len(sigmas))


def _supports_cache_position(model: nn.Module) -> bool:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return "cache_position" in parameters


def _input_device(model: nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is not None:
        for parameter in embeddings.parameters():
            if not parameter.is_meta:
                return parameter.device
    for parameter in model.parameters():
        if not parameter.is_meta:
            return parameter.device
    raise ValueError("EBSS requires materialized model weights; the model is still on meta.")


def _vocab_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return value
    embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if embeddings is not None and getattr(embeddings, "weight", None) is not None:
        return int(embeddings.weight.shape[0])
    raise ValueError("EBSS cannot resolve the model vocabulary size.")


def _forward_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    cache: Any = None,
    cache_position: Optional[torch.Tensor] = None,
    use_cache_position: bool = False,
) -> Any:
    kwargs: Dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
    if cache is not None:
        kwargs["past_key_values"] = cache
    if use_cache_position and cache_position is not None:
        kwargs["cache_position"] = cache_position
    return model(**kwargs)


def generate_ebss_calibration(
    model: nn.Module,
    config: EBSSConfig,
    *,
    progress: Optional[Any] = None,
) -> List[List[int]]:
    """Generate ``config.num_sentences`` token sequences with Algorithm 1.

    The first token of each sentence is drawn uniformly from the vocabulary.
    Its constant ``-log V`` prior cancels from every ranking, so ``R_S`` starts
    at zero.  The balance term uses the prefix's routing statistics because the
    next token's routing is only known after the branch is forwarded; the final
    sentence score uses the complete sequence.

    ``progress`` is an optional callable invoked as ``progress(sentence_index,
    total)`` after each sentence.  The model is temporarily switched to eval
    mode; its training flag and every routing hook are restored in ``finally``.
    """

    if not isinstance(config, EBSSConfig):
        raise TypeError("generate_ebss_calibration expects an EBSSConfig.")
    if config.width < 1:
        raise ValueError("EBSS width must be at least 1.")
    max_positions = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if (
        isinstance(max_positions, int)
        and max_positions > 0
        and int(config.seq_len) > max_positions
    ):
        raise ValueError(
            f"EBSS seq_len={int(config.seq_len)} exceeds the model's "
            f"max_position_embeddings={max_positions}."
        )

    device = _input_device(model)
    vocab_size = _vocab_size(model)
    was_training = bool(getattr(model, "training", False))
    use_cache_position = _supports_cache_position(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    sentences: List[List[int]] = []

    observer = _RoutingObserver(model)
    model.eval()
    try:
        with observer:
            with torch.inference_mode():
                for sentence_index in range(config.num_sentences):
                    first = int(torch.randint(0, vocab_size, (1,), generator=generator).item())
                    observer.begin_forward()
                    output = _forward_step(
                        model,
                        torch.tensor([[first]], dtype=torch.long, device=device),
                        use_cache_position=use_cache_position,
                    )
                    cache = getattr(output, "past_key_values", None)
                    if cache is None:
                        raise ValueError(
                            "EBSS requires a model forward that returns `past_key_values`; "
                            f"{type(model).__name__} returned none."
                        )
                    logits = output.logits[0, -1].to(dtype=torch.float32)
                    if logits.numel() < vocab_size:
                        vocab_size = int(logits.numel())
                    logits = logits[:vocab_size]
                    usage = observer.end_forward()
                    candidates = [
                        _Candidate(
                            tokens=[first],
                            log_prob_sum=0.0,
                            usage=_merge_usage({}, usage, observer.num_experts),
                            cache=cache,
                            logits=logits,
                        )
                    ]

                    for step in range(1, int(config.seq_len)):
                        width = min(int(config.width), len(candidates) * vocab_size)
                        flat_scores: List[torch.Tensor] = []
                        for candidate in candidates:
                            sigma = _expert_balance(candidate.usage, observer.num_experts)
                            log_probs = torch.log_softmax(candidate.logits, dim=-1)
                            length = len(candidate.tokens)
                            flat_scores.append(
                                -(candidate.log_prob_sum + log_probs) / float(length + 1)
                                + sigma / float(config.temperature)
                            )
                        scores = torch.cat(flat_scores)
                        top_scores, top_indices = torch.topk(scores, width, largest=True, sorted=True)

                        next_candidates: List[_Candidate] = []
                        for rank in range(width):
                            flat_index = int(top_indices[rank].item())
                            parent_index, token = divmod(flat_index, vocab_size)
                            parent = candidates[parent_index]
                            parent_log_probs = torch.log_softmax(parent.logits, dim=-1)
                            token_log_prob = float(parent_log_probs[token].item())

                            child_cache = copy.deepcopy(parent.cache)
                            observer.begin_forward()
                            output = _forward_step(
                                model,
                                torch.tensor([[token]], dtype=torch.long, device=device),
                                cache=child_cache,
                                cache_position=torch.tensor(
                                    [len(parent.tokens)], dtype=torch.long, device=device
                                ),
                                use_cache_position=use_cache_position,
                            )
                            child_cache = getattr(output, "past_key_values", None)
                            if child_cache is None:
                                raise ValueError(
                                    "EBSS incremental forward returned no `past_key_values`."
                                )
                            child_usage = observer.end_forward()
                            next_candidates.append(
                                _Candidate(
                                    tokens=parent.tokens + [token],
                                    log_prob_sum=parent.log_prob_sum + token_log_prob,
                                    usage=_merge_usage(
                                        parent.usage, child_usage, observer.num_experts
                                    ),
                                    cache=child_cache,
                                    logits=output.logits[0, -1].to(dtype=torch.float32)[:vocab_size],
                                )
                            )
                        candidates = next_candidates

                    def final_score(candidate: _Candidate) -> float:
                        sigma = _expert_balance(candidate.usage, observer.num_experts)
                        return (
                            -candidate.log_prob_sum / float(len(candidate.tokens))
                            + sigma / float(config.temperature)
                        )

                    best = max(candidates, key=final_score)
                    sentences.append(list(best.tokens))
                    if progress is not None:
                        progress(sentence_index + 1, int(config.num_sentences))
    finally:
        model.train(was_training)

    log.info(
        "MoEQuant EBSS: generated %d sentences of %d tokens (width=%d, temperature=%.3f, seed=%d).",
        len(sentences),
        int(config.seq_len),
        int(config.width),
        float(config.temperature),
        int(config.seed),
    )
    return sentences
