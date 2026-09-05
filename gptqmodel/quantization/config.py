# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import json
import math
import os.path
import tempfile
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from functools import total_ordering
from os.path import join
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import pcre
import torch
from packaging import version

from ..adapter.adapter import Lora, normalize_adapter
from ..utils.logger import setup_logger
from .diagnostics import (
    QuantizationDiagnosticsMode,
    normalize_quantization_diagnostics_mode,
)
from .fused_forward_config import FusedForwardConfig
from .qvq_activation import (
    QVQ_FP8_ACTIVATION_FORMAT,
    QVQ_FP8_ACTIVATION_SCALE_METHOD,
    normalize_qvq_fp8_activation_format,
    normalize_qvq_fp8_activation_scale_method,
)
from .qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
from .qvq_rates import QVQ_BITS, normalize_qvq_rate
from .qvq_yaqa import (
    YAQA_DEFAULT_RATE_REGULARIZATION,
    YAQA_DEFAULT_REGULARIZATION,
    YAQA_PAPER_MINIMUM_SEQUENCES,
)

log = setup_logger()


@dataclass
class ChatTemplateConfig:
    """Controls optional YAQA down-weighting of formatter/control tokens."""

    enabled: bool = False
    content_weight: float = 0.97

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("ChatTemplateConfig: `enabled` must be boolean.")
        if (
            isinstance(self.content_weight, bool)
            or not isinstance(self.content_weight, (int, float))
            or not math.isfinite(float(self.content_weight))
            or not 0.0 < float(self.content_weight) < 1.0
        ):
            raise ValueError("ChatTemplateConfig: `content_weight` must be finite and in (0, 1).")
        self.content_weight = float(self.content_weight)


VITERBI_PRUNING_MODES = ("auto", "off", "required")
VITERBI_PRUNING_STRATEGIES = ("norm_band",)
VITERBI_PRUNING_FALLBACKS = ("baseline", "error")


@dataclass
class ViterbiPruningConfig:
    """Exact survivor-pruning policy for the QVQ V2 segmented grid recurrence.

    ``mode`` selects the policy that reaches the CUDA V2 segmented grid
    dispatch:

    - ``auto`` (default) reproduces today's automatic behavior exactly. The
      exact norm-band survivor pruning runs for eligible unconstrained,
      unweighted, FP16-codebook W2.5/W3 two-bank/four-bank grid calls, and
      every other call keeps the unmodified baseline recurrence.
    - ``off`` deterministically suppresses norm-band dispatch, so every call
      uses the baseline recurrence.
    - ``required`` demands norm-band dispatch for every requested call and
      raises a clear error instead of silently falling back to the baseline
      recurrence.

    ``fallback`` only decides what ``auto`` does with a call that cannot use
    norm-band pruning: ``baseline`` keeps the exact baseline recurrence and
    ``error`` raises. ``off`` and ``required`` fully determine their own
    behavior, so they must leave ``fallback`` at its ``baseline`` default
    rather than encoding a redundant or contradictory combination.

    ``strategy`` names the pruning family; only the exact ``norm_band`` band
    is implemented. ``exact`` guards the future introduction of an approximate
    strategy and must stay ``True`` until one exists.

    Precedence: this configuration is authoritative. The deprecated
    ``GPTQMODEL_QVQ_DISABLE_OCTET_GRID`` A/B escape hatch is honored only when
    ``mode="auto"``; ``off`` and ``required`` ignore it entirely. Under
    ``mode="auto"`` with ``fallback="error"`` a variable that disables the fast
    path makes the call unable to prune, which raises like any other
    ineligible call.
    """

    mode: str = "auto"
    strategy: str = "norm_band"
    exact: bool = True
    fallback: str = "baseline"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("ViterbiPruningConfig: `mode` must be a string.")
        self.mode = self.mode.strip().lower()
        if self.mode not in VITERBI_PRUNING_MODES:
            raise ValueError(
                "ViterbiPruningConfig: `mode` must be one of "
                f"{list(VITERBI_PRUNING_MODES)}, got {self.mode!r}."
            )
        if not isinstance(self.strategy, str):
            raise TypeError("ViterbiPruningConfig: `strategy` must be a string.")
        self.strategy = self.strategy.strip().lower()
        if self.strategy not in VITERBI_PRUNING_STRATEGIES:
            raise ValueError(
                "ViterbiPruningConfig: `strategy` must be one of "
                f"{list(VITERBI_PRUNING_STRATEGIES)}, got {self.strategy!r}."
            )
        if not isinstance(self.exact, bool):
            raise TypeError("ViterbiPruningConfig: `exact` must be boolean.")
        if not self.exact:
            raise ValueError(
                "ViterbiPruningConfig: `exact=False` is rejected because no approximate "
                "survivor-pruning strategy exists; the `norm_band` strategy is exact."
            )
        if not isinstance(self.fallback, str):
            raise TypeError("ViterbiPruningConfig: `fallback` must be a string.")
        self.fallback = self.fallback.strip().lower()
        if self.fallback not in VITERBI_PRUNING_FALLBACKS:
            raise ValueError(
                "ViterbiPruningConfig: `fallback` must be one of "
                f"{list(VITERBI_PRUNING_FALLBACKS)}, got {self.fallback!r}."
            )
        if self.mode != "auto" and self.fallback != "baseline":
            raise ValueError(
                f"ViterbiPruningConfig: `mode={self.mode!r}` already determines the behavior of "
                "ineligible calls, so `fallback` must stay at its default `baseline`; "
                "`fallback` applies to `mode='auto'` only."
            )


@dataclass
class YaqaConfig:
    """YAQA-v3 full-model Fisher collection controls."""

    seed: int = 0
    regularization: float = YAQA_DEFAULT_REGULARIZATION
    # The full-depth Llama sweep selected stronger damping for W1--W4. Rates
    # above W4 use the global 0.05 fallback. Explicit overrides still win.
    regularization_by_rate: tuple[tuple[float, float], ...] = YAQA_DEFAULT_RATE_REGULARIZATION
    minimum_sequences: int = YAQA_PAPER_MINIMUM_SEQUENCES
    batch_size: int = 8
    chat_template: ChatTemplateConfig = field(default_factory=ChatTemplateConfig)
    activation_checkpointing: bool = True
    mps_cleanup_interval: int = 8
    sequence_sort: str = "desc"
    # Optional per-source Fisher importance weights. The source column is read
    # from each raw YAQA calibration row and preserved as one scalar per
    # independent sequence. Sketch-B applies the weight to each sequence Gram;
    # rows are never physically duplicated.
    source_weight_column: str | None = None
    source_weights: tuple[tuple[str, float], ...] = ()
    max_factor_bytes_per_pass: int | None = None
    v2b2_family_mode: str = "reselect"
    sample_strategy: str = "full"
    spectral_refinement: bool = False
    spectral_ranks: tuple[int, ...] = (8, 16, 32)
    spectral_lambdas: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    spectral_push: bool = False
    spectral_push_alphas: tuple[float, ...] = (0.25, 0.5, 1.0)
    spectral_localized: bool = False
    spectral_localized_alphas: tuple[float, ...] = (0.25, 0.5, 1.0)
    spectral_localized_max_segments: int = 8
    spectral_localized_max_changes: int = 1
    spectral_localized_replay_candidates: int = 0
    spectral_localized_direct_replay_candidates: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("YaqaConfig: `seed` must be an integer.")
        if isinstance(self.regularization, bool) or not isinstance(self.regularization, (int, float)):
            raise TypeError("YaqaConfig: `regularization` must be a real scalar.")
        self.regularization = float(self.regularization)
        if not math.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("YaqaConfig: `regularization` must be finite and nonnegative.")
        if not isinstance(self.regularization_by_rate, (tuple, list)):
            raise TypeError("YaqaConfig: `regularization_by_rate` must be a sequence of `(bits, value)` pairs.")
        normalized_rate_overrides = []
        seen_rates = set()
        for override in self.regularization_by_rate:
            if not isinstance(override, (tuple, list)) or len(override) != 2:
                raise ValueError("YaqaConfig: every regularization rate override must be a `(bits, value)` pair.")
            rate, value = override
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or float(rate) <= 0
            ):
                raise ValueError("YaqaConfig: override rates must be finite and positive.")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError("YaqaConfig: override regularization must be finite and nonnegative.")
            normalized_rate = float(rate)
            if normalized_rate in seen_rates:
                raise ValueError(f"YaqaConfig: duplicate regularization override for rate {normalized_rate}.")
            seen_rates.add(normalized_rate)
            normalized_rate_overrides.append((normalized_rate, float(value)))
        self.regularization_by_rate = tuple(sorted(normalized_rate_overrides))
        if (
            isinstance(self.minimum_sequences, bool)
            or not isinstance(self.minimum_sequences, int)
            or self.minimum_sequences < 1
        ):
            raise ValueError("YaqaConfig: `minimum_sequences` must be a positive integer.")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("YaqaConfig: `batch_size` must be a positive integer.")
        if isinstance(self.chat_template, dict):
            self.chat_template = ChatTemplateConfig(**self.chat_template)
        elif not isinstance(self.chat_template, ChatTemplateConfig):
            raise TypeError("YaqaConfig: `chat_template` must be a ChatTemplateConfig.")
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("YaqaConfig: `activation_checkpointing` must be boolean.")
        if (
            isinstance(self.mps_cleanup_interval, bool)
            or not isinstance(self.mps_cleanup_interval, int)
            or self.mps_cleanup_interval < 1
        ):
            raise ValueError("YaqaConfig: `mps_cleanup_interval` must be a positive integer.")
        if not isinstance(self.sequence_sort, str):
            raise TypeError("YaqaConfig: `sequence_sort` must be a string.")
        self.sequence_sort = self.sequence_sort.strip().lower()
        if self.sequence_sort not in {"none", "asc", "desc"}:
            raise ValueError("YaqaConfig: `sequence_sort` must be one of `none`, `asc`, or `desc`.")
        if self.source_weight_column is not None:
            if not isinstance(self.source_weight_column, str):
                raise TypeError("YaqaConfig: `source_weight_column` must be a string or None.")
            self.source_weight_column = self.source_weight_column.strip()
            if not self.source_weight_column:
                raise ValueError("YaqaConfig: `source_weight_column` must not be empty.")
        if isinstance(self.source_weights, dict):
            source_weight_items = self.source_weights.items()
        elif isinstance(self.source_weights, (tuple, list)):
            source_weight_items = self.source_weights
        else:
            raise TypeError("YaqaConfig: `source_weights` must be a mapping or sequence of `(source, weight)` pairs.")
        normalized_source_weights = []
        seen_sources = set()
        for item in source_weight_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("YaqaConfig: every source weight must be a `(source, weight)` pair.")
            source, weight = item
            if not isinstance(source, str) or not source.strip():
                raise ValueError("YaqaConfig: source-weight names must be non-empty strings.")
            source = source.strip()
            if source in seen_sources:
                raise ValueError(f"YaqaConfig: duplicate source weight for {source!r}.")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0
            ):
                raise ValueError("YaqaConfig: source weights must be finite and positive.")
            seen_sources.add(source)
            normalized_source_weights.append((source, float(weight)))
        self.source_weights = tuple(normalized_source_weights)
        if bool(self.source_weight_column) != bool(self.source_weights):
            raise ValueError(
                "YaqaConfig: `source_weight_column` and non-empty `source_weights` must be configured together."
            )
        if self.max_factor_bytes_per_pass is not None and (
            isinstance(self.max_factor_bytes_per_pass, bool)
            or not isinstance(self.max_factor_bytes_per_pass, int)
            or self.max_factor_bytes_per_pass < 1
        ):
            raise ValueError("YaqaConfig: `max_factor_bytes_per_pass` must be a positive integer or None.")
        if not isinstance(self.v2b2_family_mode, str):
            raise TypeError("YaqaConfig: `v2b2_family_mode` must be a string.")
        self.v2b2_family_mode = self.v2b2_family_mode.strip().lower()
        if self.v2b2_family_mode not in {"fixed_block_ldlq", "reselect"}:
            raise ValueError(
                "YaqaConfig: `v2b2_family_mode` must be `fixed_block_ldlq` or `reselect`."
            )
        if not isinstance(self.sample_strategy, str):
            raise TypeError("YaqaConfig: `sample_strategy` must be a string.")
        self.sample_strategy = self.sample_strategy.strip().lower()
        if self.sample_strategy not in {"full", "32_16x16", "64_16x16", "96_16x16", "128_16x16", "256_16x16"}:
            raise ValueError(
                "YaqaConfig: `sample_strategy` must be `full`, `32_16x16`, `64_16x16`, `96_16x16`, "
                "`128_16x16`, or `256_16x16`."
            )
        if self.v2b2_family_mode != "reselect" and self.sample_strategy != "full":
            raise ValueError("YaqaConfig: sampled family selection requires `v2b2_family_mode=reselect`.")
        if not isinstance(self.spectral_refinement, bool):
            raise TypeError("YaqaConfig: `spectral_refinement` must be boolean.")
        if not isinstance(self.spectral_push, bool):
            raise TypeError("YaqaConfig: `spectral_push` must be boolean.")
        if not isinstance(self.spectral_localized, bool):
            raise TypeError("YaqaConfig: `spectral_localized` must be boolean.")
        if sum((self.spectral_refinement, self.spectral_push, self.spectral_localized)) > 1:
            raise ValueError("YaqaConfig: spectral refinement experiments are mutually exclusive.")
        if not isinstance(self.spectral_ranks, (tuple, list)) or not self.spectral_ranks:
            raise ValueError("YaqaConfig: `spectral_ranks` must be a non-empty sequence.")
        if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in self.spectral_ranks):
            raise ValueError("YaqaConfig: every spectral rank must be a positive integer.")
        self.spectral_ranks = tuple(dict.fromkeys(self.spectral_ranks))
        if not isinstance(self.spectral_lambdas, (tuple, list)) or not self.spectral_lambdas:
            raise ValueError("YaqaConfig: `spectral_lambdas` must be a non-empty sequence.")
        if any(
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(float(strength))
            or float(strength) <= 0
            for strength in self.spectral_lambdas
        ):
            raise ValueError("YaqaConfig: every spectral lambda must be finite and positive.")
        self.spectral_lambdas = tuple(dict.fromkeys(float(strength) for strength in self.spectral_lambdas))
        if not isinstance(self.spectral_push_alphas, (tuple, list)) or not self.spectral_push_alphas:
            raise ValueError("YaqaConfig: `spectral_push_alphas` must be a non-empty sequence.")
        if any(
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0
            for alpha in self.spectral_push_alphas
        ):
            raise ValueError("YaqaConfig: every spectral push alpha must be finite and positive.")
        self.spectral_push_alphas = tuple(dict.fromkeys(float(alpha) for alpha in self.spectral_push_alphas))
        if not isinstance(self.spectral_localized_alphas, (tuple, list)) or not self.spectral_localized_alphas:
            raise ValueError("YaqaConfig: `spectral_localized_alphas` must be a non-empty sequence.")
        if any(
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0
            for alpha in self.spectral_localized_alphas
        ):
            raise ValueError("YaqaConfig: every localized spectral alpha must be finite and positive.")
        self.spectral_localized_alphas = tuple(
            dict.fromkeys(float(alpha) for alpha in self.spectral_localized_alphas)
        )
        if (
            isinstance(self.spectral_localized_max_segments, bool)
            or not isinstance(self.spectral_localized_max_segments, int)
            or self.spectral_localized_max_segments < 1
        ):
            raise ValueError("YaqaConfig: `spectral_localized_max_segments` must be a positive integer.")
        if (
            isinstance(self.spectral_localized_max_changes, bool)
            or not isinstance(self.spectral_localized_max_changes, int)
            or self.spectral_localized_max_changes < 1
            or self.spectral_localized_max_changes > self.spectral_localized_max_segments
        ):
            raise ValueError(
                "YaqaConfig: `spectral_localized_max_changes` must be between 1 and "
                "`spectral_localized_max_segments`."
            )
        if (
            isinstance(self.spectral_localized_replay_candidates, bool)
            or not isinstance(self.spectral_localized_replay_candidates, int)
            or self.spectral_localized_replay_candidates < 0
        ):
            raise ValueError(
                "YaqaConfig: `spectral_localized_replay_candidates` must be a nonnegative integer."
            )
        if (
            isinstance(self.spectral_localized_direct_replay_candidates, bool)
            or not isinstance(self.spectral_localized_direct_replay_candidates, int)
            or self.spectral_localized_direct_replay_candidates < 0
            or self.spectral_localized_direct_replay_candidates
            > self.spectral_localized_replay_candidates
        ):
            raise ValueError(
                "YaqaConfig: `spectral_localized_direct_replay_candidates` must be between zero and "
                "`spectral_localized_replay_candidates`."
            )

    def regularization_for_rate(self, bits: float) -> float:
        """Return an exact-rate override, or the configured global value."""

        if isinstance(bits, bool) or not isinstance(bits, (int, float)) or not math.isfinite(float(bits)):
            raise ValueError("YaqaConfig: `bits` must be a finite numeric rate.")
        for rate, value in self.regularization_by_rate:
            if math.isclose(float(bits), rate, rel_tol=0.0, abs_tol=1e-6):
                return value
        return self.regularization


MODULE_GRANULAR_REPLAY_SUBSETS = {
    "attention_qk": ("q_proj", "k_proj"),
    "attention_vo": ("v_proj", "o_proj"),
    "attention_qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp_gate_up": ("gate_proj", "up_proj"),
    "mlp_down": ("down_proj",),
    "mlp_gate_up_down": ("gate_proj", "up_proj", "down_proj"),
}


@dataclass
class ModuleGranularReplayConfig:
    """Offline propagated selection of complete packed candidates within coupled subsets.

    This is the public name for the P27 policy. It does not alter the QVQ checkpoint
    layout or inference kernel: every accepted candidate remains an ordinary
    V2B2-P32 module. Canonical V2+YAQA is always candidate zero and the atomic
    rollback artifact.
    """

    subsets: tuple[str, ...] = ("attention_qkvo",)
    strategy: str = "greedy"
    module_order: tuple[str, ...] = (
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    alternative_bank_ids: tuple[int, ...] = (1, 2, 3)
    replay_horizon: str = "final_logits"
    search_folds: int = 2
    minimum_relative_kl_improvement: float = 0.001
    topn_regression_limit: float = 0.0025
    require_disjoint_confirmation: bool = True
    fallback: str = "canonical_v2_yaqa"

    def __post_init__(self) -> None:
        if not isinstance(self.subsets, (tuple, list)) or not self.subsets:
            raise ValueError("ModuleGranularReplayConfig: `subsets` must be a nonempty sequence.")
        self.subsets = tuple(str(value).strip().lower() for value in self.subsets)
        if len(set(self.subsets)) != len(self.subsets):
            raise ValueError("ModuleGranularReplayConfig: `subsets` must not contain duplicates.")
        unknown_subsets = set(self.subsets) - set(MODULE_GRANULAR_REPLAY_SUBSETS)
        if unknown_subsets:
            raise ValueError(
                "ModuleGranularReplayConfig: unsupported subsets: "
                f"{sorted(unknown_subsets)}; expected {sorted(MODULE_GRANULAR_REPLAY_SUBSETS)}."
            )

        if not isinstance(self.strategy, str):
            raise TypeError("ModuleGranularReplayConfig: `strategy` must be a string.")
        self.strategy = self.strategy.strip().lower()
        if self.strategy not in {"greedy", "atomic_swiglu"}:
            raise ValueError(
                "ModuleGranularReplayConfig: strategy must be `greedy` or `atomic_swiglu`."
            )
        if self.strategy == "atomic_swiglu" and "mlp_gate_up_down" not in self.subsets:
            raise ValueError(
                "ModuleGranularReplayConfig: `atomic_swiglu` requires the `mlp_gate_up_down` subset."
            )

        if not isinstance(self.module_order, (tuple, list)) or not self.module_order:
            raise ValueError("ModuleGranularReplayConfig: `module_order` must be a nonempty sequence.")
        self.module_order = tuple(str(value).strip().lower() for value in self.module_order)
        if len(set(self.module_order)) != len(self.module_order):
            raise ValueError("ModuleGranularReplayConfig: `module_order` must not contain duplicates.")
        supported_roles = {role for roles in MODULE_GRANULAR_REPLAY_SUBSETS.values() for role in roles}
        unknown_roles = set(self.module_order) - supported_roles
        if unknown_roles:
            raise ValueError(
                f"ModuleGranularReplayConfig: unsupported module roles: {sorted(unknown_roles)}."
            )
        selected_roles = {role for subset in self.subsets for role in MODULE_GRANULAR_REPLAY_SUBSETS[subset]}
        missing_roles = selected_roles - set(self.module_order)
        if missing_roles:
            raise ValueError(
                "ModuleGranularReplayConfig: `module_order` must include every selected subset role; "
                f"missing {sorted(missing_roles)}."
            )

        if not isinstance(self.alternative_bank_ids, (tuple, list)) or not self.alternative_bank_ids:
            raise ValueError("ModuleGranularReplayConfig: `alternative_bank_ids` must be a nonempty sequence.")
        if any(
            isinstance(bank_id, bool) or not isinstance(bank_id, int) or bank_id not in (1, 2, 3)
            for bank_id in self.alternative_bank_ids
        ):
            raise ValueError("ModuleGranularReplayConfig: alternative bank IDs must be integers in {1, 2, 3}.")
        self.alternative_bank_ids = tuple(self.alternative_bank_ids)
        if len(set(self.alternative_bank_ids)) != len(self.alternative_bank_ids):
            raise ValueError("ModuleGranularReplayConfig: `alternative_bank_ids` must not contain duplicates.")

        if not isinstance(self.replay_horizon, str):
            raise TypeError("ModuleGranularReplayConfig: `replay_horizon` must be a string.")
        self.replay_horizon = self.replay_horizon.strip().lower()
        if self.replay_horizon != "final_logits":
            raise ValueError(
                "ModuleGranularReplayConfig: only the validated `final_logits` replay horizon is supported."
            )
        if isinstance(self.search_folds, bool) or not isinstance(self.search_folds, int) or self.search_folds < 2:
            raise ValueError("ModuleGranularReplayConfig: `search_folds` must be an integer of at least two.")

        for field_name in ("minimum_relative_kl_improvement", "topn_regression_limit"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"ModuleGranularReplayConfig: `{field_name}` must be a real scalar.")
            value = float(value)
            if not math.isfinite(value) or value < 0 or value >= 1:
                raise ValueError(f"ModuleGranularReplayConfig: `{field_name}` must be finite and in [0, 1).")
            setattr(self, field_name, value)

        if self.require_disjoint_confirmation is not True:
            raise ValueError(
                "ModuleGranularReplayConfig: disjoint confirmation is required for propagated promotion."
            )
        if not isinstance(self.fallback, str):
            raise TypeError("ModuleGranularReplayConfig: `fallback` must be a string.")
        self.fallback = self.fallback.strip().lower()
        if self.fallback != "canonical_v2_yaqa":
            raise ValueError(
                "ModuleGranularReplayConfig: only the exact `canonical_v2_yaqa` fallback is supported."
            )

    def roles(self) -> tuple[str, ...]:
        """Return selected roles in the configured greedy order."""

        selected = {role for subset in self.subsets for role in MODULE_GRANULAR_REPLAY_SUBSETS[subset]}
        return tuple(role for role in self.module_order if role in selected)

    def includes_module(self, module_name: str) -> bool:
        """Return whether a fully qualified module name belongs to the selected subsets."""

        return module_name.rsplit(".", 1)[-1] in self.roles()


@dataclass
class SmoothSwiGLUConfig:
    """Offline, function-preserving preconditioning for Llama-style SwiGLU."""

    enabled: bool = False
    group_size: int = 16
    candidate_exponents: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    scale_min: float = 0.5
    scale_max: float = 2.0
    max_calibration_tokens: int = 2048
    dense_parity_relative_l2_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("SmoothSwiGLUConfig: `enabled` must be boolean.")
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int) or self.group_size < 1:
            raise ValueError("SmoothSwiGLUConfig: `group_size` must be a positive integer.")
        if not isinstance(self.candidate_exponents, (tuple, list)) or not self.candidate_exponents:
            raise ValueError("SmoothSwiGLUConfig: `candidate_exponents` must be nonempty.")
        self.candidate_exponents = tuple(float(value) for value in self.candidate_exponents)
        if any(not math.isfinite(value) for value in self.candidate_exponents):
            raise ValueError("SmoothSwiGLUConfig: candidate exponents must be finite.")
        if not any(math.isclose(value, 0.0, abs_tol=1e-8) for value in self.candidate_exponents):
            raise ValueError("SmoothSwiGLUConfig: candidate exponents must include zero.")
        for field_name in ("scale_min", "scale_max"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"SmoothSwiGLUConfig: `{field_name}` must be a real scalar.")
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"SmoothSwiGLUConfig: `{field_name}` must be finite and positive.")
            setattr(self, field_name, value)
        if self.scale_min > self.scale_max:
            raise ValueError("SmoothSwiGLUConfig: `scale_min` must not exceed `scale_max`.")
        if (
            isinstance(self.max_calibration_tokens, bool)
            or not isinstance(self.max_calibration_tokens, int)
            or self.max_calibration_tokens < 1
        ):
            raise ValueError("SmoothSwiGLUConfig: `max_calibration_tokens` must be a positive integer.")
        if (
            isinstance(self.dense_parity_relative_l2_tolerance, bool)
            or not isinstance(self.dense_parity_relative_l2_tolerance, (int, float))
        ):
            raise TypeError(
                "SmoothSwiGLUConfig: `dense_parity_relative_l2_tolerance` must be a real scalar."
            )
        self.dense_parity_relative_l2_tolerance = float(self.dense_parity_relative_l2_tolerance)
        if not math.isfinite(self.dense_parity_relative_l2_tolerance) or self.dense_parity_relative_l2_tolerance <= 0:
            raise ValueError(
                "SmoothSwiGLUConfig: `dense_parity_relative_l2_tolerance` must be finite and positive."
            )


class _SharedTemporaryDirectory:
    """Share one TemporaryDirectory handle across copied config objects."""

    def __init__(self, *, prefix: str):
        self._temp_dir = tempfile.TemporaryDirectory(prefix=prefix)

    @property
    def name(self) -> str:
        return self._temp_dir.name

    def cleanup(self) -> None:
        self._temp_dir.cleanup()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


def _create_temp_offload_dir() -> _SharedTemporaryDirectory:
    return _SharedTemporaryDirectory(prefix="gptqmodel_")


_DECODER_TARGET_DTYPE_MAP = {
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

BITS_FIELD_CODE = "bits"
GROUP_SIZE_FIELD_CODE = "group_size"
FORMAT_FIELD_CODE = "format"
SYMMETRIC_FIELD_CODE = "sym"
# Deprecated JSON alias retained for backward compatibility.
FORMAT_FIELD_CHECKPOINT = "checkpoint_format"
# Hard-deprecated legacy alias. Presence should fail fast.
FORMAT_FIELD_COMPAT_MARLIN = "is_marlin_format"
# Canonical method field; `quant_method` is a deprecated JSON alias.
METHOD_FIELD_CODE = "method"
QUANT_METHOD_FIELD = "quant_method"
PACK_DTYPE_FIELD = "pack_dtype"
QUANT_CONFIG_FILENAME = "quantize_config.json"
QUANT_CONFIG_FILENAME_COMPAT = [QUANT_CONFIG_FILENAME, "quant_config.json", "config.json"]
# This is AwqBackendPackingMethod, not the GPT-QModel backend enum.
# It's used to distinguish between quantization by llm-awq and autoawq; llm-awq actually uses GEMV_FAST for packing.
AWQ_PACKING_BACKEND_FIELD = "backend"

MIN_VERSION_WITH_V2 = "0.9.0"

META_FIELD = "meta"
# quantizer is the tool that did the quantization
META_FIELD_QUANTIZER = "quantizer"

META_QUANTIZER_GPTQMODEL = "gptqmodel"

META_FIELD_URI = "uri"
META_VALUE_URI = "https://github.com/modelcloud/gptqmodel"

META_FIELD_DAMP_PERCENT = "damp_percent"
META_FIELD_DAMP_AUTO_INCREMENT = "damp_auto_increment"

META_FIELD_STATIC_GROUPS = "static_groups"
META_FIELD_TRUE_SEQUENTIAL = "true_sequential"

META_FIELD_MSE = "mse"
META_FIELD_SCALE_SEARCH = "scale_search"
META_FIELD_ACT_GROUP_AWARE = "act_group_aware"

# timestamp is the date/hour/minute the quantization config was saved
META_FIELD_TIMESTAMP = "timestamp"

# calibration_paths is the list of dataset source paths used during quantization/requantization
META_FIELD_CALIBRATION_PATHS = "calibration_paths"

META_FIELD_GPTAQ_ENABLED = "gptaq"

META_FIELD_FOEM_ENABLED = "foem"

ADAPTER_FIELD = "adapter"


# saved formats
class FORMAT(str, Enum):
    """Checkpoint and runtime tensor layout identifiers."""

    GPTQ = "gptq"
    # v2 format fixed sym = False quantization
    GPTQ_V2 = "gptq_v2"
    # planar (split-plane, word-aligned high-plane) layout; distinct from the
    # continuous gptq/gptq_v2 layouts. Zeros use v2 semantics on disk.
    GPTQ_P = "gptq_p"
    GGUF = "gguf"
    FP8 = "fp8"
    BITSANDBYTES = "bitsandbytes"
    MARLIN = "marlin"
    BITBLAS = "bitblas"
    QQQ = "qqq"
    EXL3 = "exl3"
    QVQ = "qvq"
    QVQ_V4 = "qvq_v4"
    QVQ_V4_L18 = "qvq_v4_l18"
    QVQ_DUAL_V2 = "qvq_dual_v2"
    QVQ_V2B4_P64 = "qvq_v2b4_p64"
    QVQ_V2B2_P32 = "qvq_v2b2_p32"
    MXFP4 = "mxfp4"

    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    LLM_AWQ = "llm-awq"
    PAROQUANT = "paroquant"


# quant methods
class METHOD(str, Enum):
    """Supported quantization algorithms exposed by config payloads."""

    GPTQ = "gptq"
    GGUF = "gguf"
    FP8 = "fp8"
    BITSANDBYTES = "bitsandbytes"
    QQQ = "qqq"
    AWQ = "awq"
    EXL3 = "exl3"
    QVQ = "qvq"
    PARO = "paroquant"
    MXFP4 = "mxfp4"


class ScaleSearchConfig(str, Enum):
    """Objectives available when searching GPTQ weight scales and clipping ranges."""

    MSE = "mse"
    ACTIVATION = "activation"
    HESSIAN = "hessian"
    HYBRID = "hybrid"
    MARLIN = "marlin"
    MARLIN_MSE = "marlin_mse"
    MARLIN_ACTIVATION = "marlin_activation"


class AdaptiveClippingMetric(str, Enum):
    """Loss objective used by adaptive weight clipping search."""

    GPTQ_ERROR = "gptq_error"
    MSE = "mse"
    HESSIAN_DIAG = "hessian_diag"


# Accuracy-first GPTQ defaults. Static 5% damping and activation-aware scale
# search remain the general defaults; adaptive damping and clipping require an
# explicit config. Keep the scale selector distinct from the public ``None``
# opt-out and the legacy ``mse`` API.
GPTQ_DEFAULT_DAMP_PERCENT = 0.05
GPTQ_DEFAULT_DAMP_AUTO_INCREMENT = 0.01
GPTQ_DEFAULT_SCALE_SEARCH = ScaleSearchConfig.ACTIVATION
_UNSET_SCALE_SEARCH = object()


class VramStrategy(str, Enum):
    """Placement strategies shared by dense and MoE device pools."""

    EXCLUSIVE = "exclusive"
    BALANCED = "balanced"


class QuantizeEmbed(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


@dataclass
class QuantizeEmbedConfig:
    embed_quant_mode: QuantizeEmbed = QuantizeEmbed.OUTPUT
    embed_only: bool = True


class ShardStrategy(str, Enum):
    """Safetensors checkpoint sharding strategy used during save and reshard."""

    PER_LAYER = "per_layer"
    """One output shard per transformer layer. Non-layer tensors (embeddings,
    final norm, lm_head) are grouped into a separate non-layer shard."""

    PER_LAYER_MOE = "per_layer_moe"
    """One dense/shared shard per transformer layer plus bounded shards for
    routed expert projection modules declared by the model's module tree."""


class FallbackStrategy(str, Enum):
    """
    +-----------+----------------------+---------------------------+------------------------------+
    | strategy  | center               | scale                     | strengths / weaknesses       |
    +-----------+----------------------+---------------------------+------------------------------+
    | rtn       | min/max (quantizer)  | min/max (quantizer)        | simple, but outlier-driven   |
    | midpoint  | (min+max)/2          | (max-min)                  | symmetric, outlier-sensitive |
    | mean      | mean(w)              | 2*max(|w-mean|)            | stable for symmetric data    |
    | median    | median(w)            | 2*max(|w-median|)          | robust center vs outliers    |
    | stdclip   | mean(w)              | 2*sigma*std                | tames tails, may clip signal |
    +-----------+----------------------+---------------------------+------------------------------+
    """
    RTN = "rtn" # round to nearest
    MIDPOINT = "midpoint"
    MEAN = "mean"
    MEDIAN = "median"
    STDCLIP = "stdclip"


class WeightOnlyMethod(str, Enum):
    """Weight-only quantization backends available to fallback flows."""

    RTN = "rtn"
    GGUF = "gguf"
    FP8 = "fp8"
    BITSANDBYTES = "bitsandbytes"
    NVFP4 = "nvfp4"


class PreProcessorCode(str, Enum):
    """Identifiers for preprocessing passes that run before quantization."""

    SMOOTHER = "smoother"
    AUTO_MODULE_DECODER = "auto_module_decoder"
    TENSOR_PARALLEL_PADDER = "tensor_parallel_padder"
    ANALYSIS = "analysis"


_GGUF_BITS_ALIAS_INFO = {
    "q1_0": {"bits": 1, "version": "q", "variant": "0", "quality": None},
    "q1_0_g128": {"bits": 1, "version": "q", "variant": "0", "quality": "g128"},
    "q2_0": {"bits": 2, "version": "q", "variant": "0", "quality": None},
    "q4_0": {"bits": 4, "version": "q", "variant": "0", "quality": None},
    "q8_0": {"bits": 8, "version": "q", "variant": "0", "quality": None},
    "q4_k": {"bits": 4, "version": "q", "variant": "k", "quality": None},
    "q4_k_s": {"bits": 4, "version": "q", "variant": "k", "quality": "s"},
    "q4_k_m": {"bits": 4, "version": "q", "variant": "k", "quality": "m"},
    "q5_k": {"bits": 5, "version": "q", "variant": "k", "quality": None},
    "q5_k_s": {"bits": 5, "version": "q", "variant": "k", "quality": "s"},
    "q5_k_m": {"bits": 5, "version": "q", "variant": "k", "quality": "m"},
    "q6_k": {"bits": 6, "version": "q", "variant": "k", "quality": None},
}
_GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH = {
    1: "q1_0",
    2: "q2_0",
    4: "q4_0",
    5: "q5_k_m",
    6: "q6_k",
    8: "q8_0",
}
_GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS = {
    "q1_0": 1.5,
    "q1_0_g128": 1.125,
    "q2_0": 2.125,
    "q4_0": 4.5,
    "q8_0": 8.5,
    "q4_k": 4.5,
    "q4_k_s": 4.5,
    "q4_k_m": 4.5,
    "q5_k": 5.5,
    "q5_k_s": 5.0,
    "q5_k_m": 5.5,
    "q6_k": 6.0,
}


@total_ordering
class BaseComplexBits(ABC):
    """Comparable bit-spec base class for non-scalar bit encodings."""

    @classmethod
    @abstractmethod
    def from_string(cls, value: str) -> "BaseComplexBits":
        """Parse a serialized bit specification into an instance."""

        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        """Serialize the bit specification into its canonical string form."""

        raise NotImplementedError

    @property
    def width(self) -> int:
        """Return the integer width represented by this bit encoding."""

        return self.bits

    @property
    def name(self) -> str:
        """Return the canonical string name for this bit encoding."""

        return self.to_string()

    def _coerce_bits(self, other: Any) -> Any:
        """Convert compatible operands into raw bit widths for arithmetic."""

        if isinstance(other, BaseComplexBits):
            return other.bits
        if isinstance(other, int):
            return other
        if isinstance(other, str) and other.strip().isdigit():
            return int(other.strip())
        return NotImplemented

    def __str__(self) -> str:
        """Render the canonical string form for logging and serialization."""

        return self.to_string()

    def __hash__(self) -> int:
        """Hash bit encodings by their integer width."""

        return hash(self.bits)

    def __int__(self) -> int:
        """Expose the bit width as an integer."""

        return self.bits

    def __index__(self) -> int:
        """Allow the bit width to participate in index-style conversions."""

        return self.bits

    def __float__(self) -> float:
        """Expose the bit width as a float."""

        return float(self.bits)

    def __eq__(self, other: Any) -> bool:
        """Compare complex bit encodings against strings, ints, or peers."""

        if isinstance(other, BaseComplexBits):
            return self.to_string() == other.to_string()
        if isinstance(other, int):
            return self.bits == other
        if isinstance(other, str):
            normalized = other.strip().lower().replace("-", "_")
            if normalized.isdigit():
                return self.bits == int(normalized)
            return self.to_string() == normalized
        return False

    def __lt__(self, other: Any) -> bool:
        """Order bit encodings by their effective width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits < coerced

    def __add__(self, other: Any) -> int:
        """Add the effective bit width to another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits + coerced

    def __radd__(self, other: Any) -> int:
        """Support right-hand addition with scalar-like operands."""

        return self.__add__(other)

    def __sub__(self, other: Any) -> int:
        """Subtract another scalar-like operand from this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits - coerced

    def __rsub__(self, other: Any) -> int:
        """Support right-hand subtraction against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced - self.bits

    def __mul__(self, other: Any) -> int:
        """Multiply the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits * coerced

    def __rmul__(self, other: Any) -> int:
        """Support right-hand multiplication with scalar-like operands."""

        return self.__mul__(other)

    def __floordiv__(self, other: Any) -> int:
        """Floor-divide the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits // coerced

    def __rfloordiv__(self, other: Any) -> int:
        """Support right-hand floor division against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced // self.bits

    def __truediv__(self, other: Any) -> float:
        """True-divide the bit width by another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits / coerced

    def __rtruediv__(self, other: Any) -> float:
        """Support right-hand true division against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced / self.bits

    def __mod__(self, other: Any) -> int:
        """Take the modulo of the bit width with another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits % coerced

    def __rmod__(self, other: Any) -> int:
        """Support right-hand modulo against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced % self.bits

    def __pow__(self, other: Any) -> int:
        """Raise the bit width to another scalar-like operand."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return self.bits ** coerced

    def __rpow__(self, other: Any) -> int:
        """Support right-hand exponentiation against this bit width."""

        coerced = self._coerce_bits(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced ** self.bits


@dataclass(frozen=True, eq=False)
class GGUFBits(BaseComplexBits):
    """Structured GGUF bit specification with version and variant tags."""

    bits: int
    version: str
    variant: str
    quality: Optional[str] = None

    def __post_init__(self):
        """Validate the GGUF bit-spec components after construction."""

        if self.bits <= 0:
            raise ValueError("GGUFBits: `bits` must be a positive integer.")
        if self.version not in {"q", "iq"}:
            raise ValueError("GGUFBits: `version` must be `q` or `iq`.")
        if self.variant not in {"0", "k"}:
            raise ValueError("GGUFBits: `variant` must be `0` or `k`.")
        if self.quality not in {None, "xs", "s", "m", "l", "g128"}:
            raise ValueError("GGUFBits: `quality` must be one of `[None, xs, s, m, l, g128]`.")

    @classmethod
    def from_string(cls, value: str) -> "GGUFBits":
        """Parse a GGUF alias such as ``q4_k_m`` into a typed bit spec."""

        normalized = str(value).strip().lower().replace("-", "_")
        info = _GGUF_BITS_ALIAS_INFO.get(normalized)
        if info is None:
            supported = ", ".join(sorted(_GGUF_BITS_ALIAS_INFO))
            raise ValueError(f"Unsupported GGUF bits `{value}`. Supported values: {supported}.")
        return cls(
            bits=info["bits"],
            version=info["version"],
            variant=info["variant"],
            quality=info["quality"],
        )

    def to_string(self) -> str:
        """Serialize this GGUF bit spec back to its alias form."""

        alias = f"{self.version}{self.bits}_{self.variant}"
        if self.quality is not None:
            alias = f"{alias}_{self.quality}"
        return alias

    @classmethod
    def from_alias(cls, value: str) -> "GGUFBits":
        """Backward-compatible alias parser for GGUF bit specs."""

        return cls.from_string(value)

    def serialize(self) -> str:
        """Return the canonical serialized form used in config payloads."""

        return self.to_string()

    def __repr__(self) -> str:
        """Return a debug-friendly constructor-style representation."""

        return f"GGUFBits({self.to_string()!r})"

    def to_public_format(self) -> str:
        """Return the GGUF public subtype string without the width prefix."""

        public_format = f"{self.version}_{self.variant}"
        if self.quality is not None:
            public_format = f"{public_format}_{self.quality}"
        return public_format


# Backward-compatible alias for the earlier wrapper-based refactor.
QuantBits = GGUFBits


_GGUF_PUBLIC_FORMAT_RE = pcre.compile(r"^(q|iq)_(0|k)(?:_(xs|s|m|l|g128))?$")


def _gguf_public_format_from_bits(bits: GGUFBits) -> str:
    """Project a full GGUF bit spec into its public subtype token."""

    return bits.to_public_format()


def _normalize_gguf_public_format(value: Any) -> Optional[str]:
    """Normalize GGUF subtype aliases into their public format string."""

    if value is None:
        return None

    if isinstance(value, GGUFBits):
        return _gguf_public_format_from_bits(value)

    if isinstance(value, FORMAT):
        value = value.value

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", FORMAT.GGUF.value}:
        return None
    if normalized in _GGUF_BITS_ALIAS_INFO:
        return _gguf_public_format_from_bits(GGUFBits.from_alias(normalized))
    if _GGUF_PUBLIC_FORMAT_RE.fullmatch(normalized):
        return normalized

    raise ValueError(
        "GGUFConfig: `format` must be a GGUF subtype like `q_0`, `q_k`, `q_k_s`, or `q_k_m`."
    )


def _default_gguf_public_format(bits: int) -> str:
    """Return the default GGUF subtype for a supported bit width."""

    alias = _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH.get(bits)
    if alias is None:
        raise ValueError(f"GGUFConfig: no default GGUF format exists for `{bits}`-bit quantization.")
    return _gguf_public_format_from_bits(GGUFBits.from_alias(alias))


def _gguf_bits_from_components(bits: int, public_format: str) -> GGUFBits:
    """Build a validated ``GGUFBits`` object from width and subtype parts."""

    match = _GGUF_PUBLIC_FORMAT_RE.fullmatch(public_format)
    if match is None:
        raise ValueError(
            "GGUFConfig: `format` must be a GGUF subtype like `q_0`, `q_k`, `q_k_s`, or `q_k_m`."
        )

    version_name, variant, quality = match.groups()
    bits_spec = GGUFBits(bits=bits, version=version_name, variant=variant, quality=quality)
    if bits_spec.to_string() not in _GGUF_BITS_ALIAS_INFO:
        raise ValueError(
            f"Unsupported GGUF combination: bits={bits}, format={public_format}."
        )
    return bits_spec


def _normalize_gguf_config_spec(
    bits: Union[int, str, GGUFBits],
    format_value: Optional[Union[str, FORMAT, GGUFBits]],
) -> Tuple[int, str, GGUFBits]:
    """Resolve GGUF bits and format inputs into a consistent typed triple."""

    bits_spec_from_bits: Optional[GGUFBits] = None
    normalized_bits = bits

    if isinstance(bits, GGUFBits):
        bits_spec_from_bits = bits
        normalized_bits = bits.bits
    elif isinstance(bits, str):
        raw_bits = bits.strip().lower().replace("-", "_")
        if raw_bits.isdigit():
            normalized_bits = int(raw_bits)
        else:
            bits_spec_from_bits = GGUFBits.from_alias(raw_bits)
            normalized_bits = bits_spec_from_bits.bits
    elif not isinstance(bits, int):
        raise ValueError(f"GGUFConfig: unsupported bits specification `{bits}`.")

    normalized_bits = int(normalized_bits)
    if normalized_bits not in [1, 2, 3, 4, 5, 6, 8]:
        raise ValueError("GGUFConfig: `bits` must resolve to one of `[1, 2, 3, 4, 5, 6, 8]`.")

    normalized_format = _normalize_gguf_public_format(format_value)
    if normalized_format is None:
        if bits_spec_from_bits is not None:
            bits_spec = bits_spec_from_bits
            normalized_format = _gguf_public_format_from_bits(bits_spec)
        else:
            normalized_format = _default_gguf_public_format(normalized_bits)
            bits_spec = _gguf_bits_from_components(normalized_bits, normalized_format)
    else:
        bits_spec = _gguf_bits_from_components(normalized_bits, normalized_format)
        if bits_spec_from_bits is not None and bits_spec_from_bits != bits_spec:
            raise ValueError(
                f"GGUFConfig: incompatible GGUF bits/format combination: bits={bits}, format={format_value}."
            )

    return normalized_bits, normalized_format, bits_spec


def _normalize_quant_bits(
    bits: Union[int, float, str, GGUFBits],
    format_value: Optional[Union[str, FORMAT]] = None,
) -> Union[int, float, GGUFBits]:
    """Normalize generic bit fields into ints or structured GGUF specs."""

    if isinstance(format_value, str):
        format_value = _normalize_format(format_value)

    if format_value in {
        FORMAT.QVQ,
        FORMAT.QVQ_V4,
        FORMAT.QVQ_V4_L18,
        FORMAT.QVQ_DUAL_V2,
        FORMAT.QVQ_V2B4_P64,
        FORMAT.QVQ_V2B2_P32,
    }:
        if isinstance(bits, GGUFBits):
            raise ValueError("QuantizeConfig: GGUF bit encodings require `format=gguf`.")
        return normalize_qvq_rate(bits)

    if isinstance(bits, GGUFBits):
        normalized = bits
    elif isinstance(bits, float):
        if format_value == FORMAT.EXL3:
            normalized = bits
        elif bits.is_integer():
            normalized = int(bits)
        else:
            raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")
    elif isinstance(bits, int):
        normalized = bits
    elif isinstance(bits, str):
        raw = bits.strip().lower().replace("-", "_")
        normalized = int(raw) if raw.isdigit() else GGUFBits.from_alias(raw)
    else:
        raise ValueError(f"QuantizeConfig: unsupported bits specification `{bits}`.")

    normalized_width = normalized.bits if isinstance(normalized, GGUFBits) else normalized
    valid_bit_widths = [1, 2, 3, 4, 5, 6, 7, 8]
    if normalized_width not in valid_bit_widths:
        raise ValueError(f"QuantizeConfig: `bits` must resolve to one of `{valid_bit_widths}`.")

    if format_value == FORMAT.GGUF and not isinstance(normalized, GGUFBits):
        if not isinstance(normalized_width, int):
            raise ValueError("QuantizeConfig: GGUF bit widths must be integers.")
        default_alias = _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH.get(normalized_width)
        if default_alias is None:
            raise ValueError(
                f"QuantizeConfig: no default GGUF bits alias exists for `{normalized_width}`-bit quantization."
            )
        normalized = GGUFBits.from_alias(default_alias)

    if isinstance(normalized, GGUFBits) and format_value is not None and format_value != FORMAT.GGUF:
        raise ValueError("QuantizeConfig: GGUF bit encodings require `format=gguf`.")

    return normalized


def resolve_quant_format(
    format_value: Optional[Union[str, FORMAT]],
    method: Optional[Union[str, METHOD]] = None,
    quant_method: Optional[Union[str, METHOD]] = None,
) -> FORMAT:
    """Infer the effective quantization format from method and format hints."""

    if method is None:
        method = quant_method

    if isinstance(method, str):
        method = _normalize_quant_method(method)

    if method == METHOD.GGUF:
        return FORMAT.GGUF
    if method == METHOD.FP8:
        return FORMAT.FP8
    if method == METHOD.BITSANDBYTES:
        return FORMAT.BITSANDBYTES
    if method == METHOD.EXL3:
        return FORMAT.EXL3
    if method == METHOD.QVQ:
        return FORMAT.QVQ
    if method == METHOD.PARO:
        return FORMAT.PAROQUANT
    if method == METHOD.MXFP4:
        return FORMAT.MXFP4

    if isinstance(format_value, FORMAT):
        return format_value

    try:
        if _normalize_gguf_public_format(format_value) is not None:
            return FORMAT.GGUF
    except ValueError:
        pass

    if _looks_like_fp8_fmt(format_value):
        return FORMAT.FP8
    if _looks_like_bitsandbytes_format(format_value):
        return FORMAT.BITSANDBYTES

    if format_value is None:
        return FORMAT.GPTQ

    return _normalize_format(format_value)


def _looks_like_gguf_bits(bits: Any) -> bool:
    """Return ``True`` when a value resembles a GGUF alias or bit spec."""

    if isinstance(bits, GGUFBits):
        return True
    if not isinstance(bits, str):
        return False
    normalized = bits.strip().lower().replace("-", "_")
    return normalized in _GGUF_BITS_ALIAS_INFO


def quant_bits_width(bits: Union[int, float, str, GGUFBits]) -> int:
    """Return the integer width represented by a quant bits field."""

    if isinstance(bits, float):
        if bits <= 0:
            raise ValueError("QuantizeConfig: EXL3 bits per weight must be greater than 0.")
        return max(1, int(math.floor(bits)))
    normalized = _normalize_quant_bits(bits)
    if isinstance(normalized, GGUFBits):
        return normalized.bits
    if isinstance(normalized, float):
        return max(1, int(math.floor(normalized)))
    return normalized


def serialize_quant_bits(bits: Union[int, float, str, GGUFBits]) -> Union[int, float, str]:
    """Serialize a quant bits field for JSON-compatible output payloads."""

    if isinstance(bits, float):
        return float(bits)
    normalized = _normalize_quant_bits(bits)
    return normalized.serialize() if isinstance(normalized, GGUFBits) else normalized


def _normalize_exl3_bits(bits: Union[int, float, str]) -> float:
    """Normalize EXL3 fractional bits-per-weight values."""

    if isinstance(bits, str):
        bits = float(bits.strip())
    elif isinstance(bits, int):
        bits = float(bits)
    elif not isinstance(bits, float):
        raise ValueError(f"EXL3Config: unsupported bits specification `{bits}`.")

    if not math.isfinite(bits):
        raise ValueError("EXL3Config: `bits` must be finite.")
    if bits < 1.0 or bits > 8.0:
        raise ValueError("EXL3Config: `bits` must be between 1.0 and 8.0.")
    return float(bits)


# Canonical FP8 aliases are normalized here before validating torch runtime
# support so config payloads can use either shorthand or exact dtype names.
_FP8_FMT_ALIASES = {
    "e4m3": "float8_e4m3fn",
    "float8_e4m3": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",
    "e4m3fnuz": "float8_e4m3fnuz",
    "float8_e4m3fnuz": "float8_e4m3fnuz",
    "e5m2fnuz": "float8_e5m2fnuz",
    "float8_e5m2fnuz": "float8_e5m2fnuz",
    "e8m0": "float8_e8m0fnu",
    "e8m0fnu": "float8_e8m0fnu",
    "float8_e8m0": "float8_e8m0fnu",
    "float8_e8m0fnu": "float8_e8m0fnu",
}
_FP8_WEIGHT_SCALE_METHODS = {"tensor", "row", "block"}
_FP8_SCALE_SEMANTICS = {"inverse"}
_BITSANDBYTES_4BIT_FORMATS = {"fp4", "nf4"}
_BITSANDBYTES_8BIT_FORMATS = {"int8"}
_BITSANDBYTES_FORMATS = _BITSANDBYTES_4BIT_FORMATS | _BITSANDBYTES_8BIT_FORMATS
_BITSANDBYTES_BLOCK_SIZES = {32, 64, 128, 256, 512, 1024, 2048, 4096}


def _looks_like_fp8_fmt(value: Any) -> bool:
    """Return ``True`` when a value matches a supported FP8 format alias."""

    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in _FP8_FMT_ALIASES


def _normalize_fp8_fmt(value: Optional[str]) -> str:
    """Resolve FP8 format aliases to the canonical PyTorch dtype name."""

    if isinstance(value, FORMAT):
        if value != FORMAT.FP8:
            raise ValueError(f"FP8Config: unsupported `format` `{value}`.")
        value = None

    normalized = "float8_e4m3fn" if value is None else str(value).strip().lower()
    if normalized in {"", FORMAT.FP8.value}:
        normalized = "float8_e4m3fn"
    resolved = _FP8_FMT_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(sorted(_FP8_FMT_ALIASES))
        raise ValueError(f"FP8Config: unsupported `format` `{value}`. Supported values: {supported}.")
    if not hasattr(torch, resolved):
        raise ValueError(f"FP8Config: current PyTorch build does not provide `{resolved}`.")
    return resolved


def _normalize_fp8_weight_block_size(value: Optional[Union[List[int], Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    """Validate and normalize FP8 block-scale dimensions."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("FP8Config: `weight_block_size` must be a 2-item list/tuple or None.")
    rows, cols = int(value[0]), int(value[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("FP8Config: `weight_block_size` entries must be positive integers.")
    return rows, cols


def _normalize_fp8_weight_scale_method(
    value: Optional[str],
    *,
    weight_block_size: Optional[Tuple[int, int]],
) -> str:
    """Resolve the FP8 weight scaling strategy from config inputs."""

    normalized = "block" if weight_block_size is not None and value is None else (value or "row")
    normalized = str(normalized).strip().lower()
    if normalized not in _FP8_WEIGHT_SCALE_METHODS:
        supported = ", ".join(sorted(_FP8_WEIGHT_SCALE_METHODS))
        raise ValueError(
            f"FP8Config: `weight_scale_method` must be one of {{{supported}}}, got `{value}`."
        )
    if normalized == "block" and weight_block_size is None:
        raise ValueError("FP8Config: `weight_scale_method='block'` requires `weight_block_size`.")
    if normalized != "block" and weight_block_size is not None:
        raise ValueError(
            "FP8Config: `weight_block_size` is only valid when `weight_scale_method='block'`."
        )
    return normalized


def _normalize_fp8_scale_semantics(value: Optional[str]) -> str:
    """Normalize FP8 scale semantics to the supported enum-like string."""

    normalized = "inverse" if value is None else str(value).strip().lower()
    if normalized not in _FP8_SCALE_SEMANTICS:
        supported = ", ".join(sorted(_FP8_SCALE_SEMANTICS))
        raise ValueError(
            f"FP8Config: `weight_scale_semantics` must be one of {{{supported}}}, got `{value}`."
        )
    return normalized


def _looks_like_bitsandbytes_format(value: Any) -> bool:
    """Return ``True`` when a value matches a bitsandbytes format alias."""

    if value is None:
        return False
    normalized = str(value).strip().lower().replace("-", "_")
    return normalized in _BITSANDBYTES_FORMATS


def _normalize_bitsandbytes_format(value: Optional[str], *, bits: Optional[int] = None) -> str:
    """Normalize bitsandbytes format aliases for the requested bit width."""

    default_format = "int8" if bits == 8 else "fp4"
    normalized = default_format if value is None else str(value).strip().lower().replace("-", "_")
    if normalized in {"", FORMAT.BITSANDBYTES.value}:
        normalized = default_format

    if bits == 4:
        allowed_formats = _BITSANDBYTES_4BIT_FORMATS
    elif bits == 8:
        allowed_formats = _BITSANDBYTES_8BIT_FORMATS
    else:
        allowed_formats = _BITSANDBYTES_FORMATS

    if normalized not in allowed_formats:
        supported = ", ".join(sorted(allowed_formats))
        raise ValueError(
            f"BitsAndBytesConfig: `format` must be one of {{{supported}}}, got `{value}`."
        )
    return normalized


def _normalize_bitsandbytes_quant_type(value: Optional[str]) -> str:
    """Normalize the legacy 4-bit bitsandbytes quant type field."""

    return _normalize_bitsandbytes_format(value, bits=4)


def _normalize_bitsandbytes_block_size(value: Optional[int]) -> int:
    """Validate and normalize the bitsandbytes block size setting."""

    normalized = 64 if value is None else int(value)
    if normalized not in _BITSANDBYTES_BLOCK_SIZES:
        supported = ", ".join(str(item) for item in sorted(_BITSANDBYTES_BLOCK_SIZES))
        raise ValueError(
            f"BitsAndBytesConfig: `block_size` must be one of {{{supported}}}, got `{value}`."
        )
    return normalized


@dataclass
class SmoothMethod:
    """Base smoother descriptor shared by all smoothing strategies."""

    name: str
    # Apply the smoother only when group size >= this threshold.
    group_size_threshold: int = 128


@dataclass
class SmoothPercentile(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | clip(|w|) at p-th percentile             |
    | config         | SmoothPercentile(percentile=p)           |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | percentile (p) | percentile of |w| used as clip threshold |
    | effect         | higher p = less clipping                  |
    +----------------+-------------------------------------------+
    """
    percentile: float = 99.0

    def __init__(self, percentile: float = 99.0, group_size_threshold: int = 128):
        """Configure percentile clipping with an optional group-size floor."""

        super().__init__(name="percentile", group_size_threshold=group_size_threshold)
        self.percentile = percentile


@dataclass
class SmoothPercentileAsymmetric(SmoothMethod):
    """
    +-------------------+-------------------------------------------+
    | math              | clip to [p_low, p_high] percentiles      |
    | config            | SmoothPercentileAsymmetric(low, high)    |
    +-------------------+-------------------------------------------+
    +-------------------+-------------------------------------------+
    | low/high          | percentile bounds on raw weights         |
    | effect            | asymmetric clipping of tails             |
    +-------------------+-------------------------------------------+
    """
    low: float = 0.5
    high: float = 99.5

    def __init__(self, low: float = 0.5, high: float = 99.5, group_size_threshold: int = 128):
        """Configure asymmetric percentile clipping bounds."""

        super().__init__(name="percentile_asym", group_size_threshold=group_size_threshold)
        self.low = low
        self.high = high


@dataclass
class SmoothMAD(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | median +/- K * MAD                        |
    | config         | SmoothMAD(k=K)                            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | K              | width multiplier for MAD window           |
    | effect         | higher K = less clipping                  |
    +----------------+-------------------------------------------+
    """
    k: float = 2.75

    def __init__(self, k: float = 2.75, group_size_threshold: int = 128):
        """Configure MAD-based clipping width and activation threshold."""

        super().__init__(name="mad", group_size_threshold=group_size_threshold)
        self.k = k


@dataclass
class SmoothMSE(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | grid-search shrink p in [1..maxshrink]    |
    | config         | SmoothMSE(steps=N, maxshrink=S)           |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | steps (N)      | number of shrink candidates               |
    | maxshrink (S)  | smallest range multiplier                 |
    | effect         | more steps = better fit, slower           |
    +----------------+-------------------------------------------+
    """
    steps: int = 32
    maxshrink: float = 0.8

    def __init__(self, steps: int = 32, maxshrink: float = 0.8, group_size_threshold: int = 128):
        """Configure search granularity for MSE-based shrinking."""

        super().__init__(name="mse", group_size_threshold=group_size_threshold)
        self.steps = steps
        self.maxshrink = maxshrink


@dataclass
class SmoothOutlier(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | clip by kth |w|, keep (100-pct)% mass     |
    | config         | SmoothOutlier(pct=p)                      |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | pct (p)        | top-pct of |w| treated as outliers        |
    | effect         | higher p = more clipping                  |
    +----------------+-------------------------------------------+
    """
    pct: float = 1.0

    def __init__(self, pct: float = 1.0, group_size_threshold: int = 128):
        """Configure top-percent outlier clipping behavior."""

        super().__init__(name="outlier", group_size_threshold=group_size_threshold)
        self.pct = pct


@dataclass
class SmoothSoftNorm(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | z=(w-mean)/rms, clip z to +/-K            |
    | config         | SmoothSoftNorm(k=K)                       |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | K              | z-score clip limit                        |
    | effect         | higher K = less clipping                  |
    +----------------+-------------------------------------------+
    """
    k: float = 3.0

    def __init__(self, k: float = 3.0, group_size_threshold: int = 128):
        """Configure z-score clipping strength for soft normalization."""

        super().__init__(name="softnorm", group_size_threshold=group_size_threshold)
        self.k = k


@dataclass
class SmoothLog(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | log1p(mu*|w|) percentile, invert to clip  |
    | config         | SmoothLog(percentile=p, mu=mu)            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | percentile (p) | percentile in log space for clip          |
    | mu             | log companding strength                   |
    | effect         | higher mu compresses outliers more        |
    +----------------+-------------------------------------------+
    """
    percentile: float = 99.0
    mu: float = 8.0

    def __init__(self, percentile: float = 99.0, mu: float = 8.0, group_size_threshold: int = 128):
        """Configure log-domain smoothing with percentile and companding strength."""

        super().__init__(name="log", group_size_threshold=group_size_threshold)
        self.percentile = percentile
        self.mu = mu


@dataclass
class SmoothRowCol(SmoothMethod):
    """
    +----------------+-------------------------------------------+
    | math           | divide by row/col RMS, re-scale after     |
    | config         | SmoothRowCol(axis="row"|"col")            |
    +----------------+-------------------------------------------+
    +----------------+-------------------------------------------+
    | axis           | apply RMS scale per "row" or "col"        |
    | effect         | normalizes dynamic range before quant     |
    +----------------+-------------------------------------------+
    """
    axis: str = "row"

    def __init__(self, axis: str = "row", group_size_threshold: int = 128):
        """Configure RMS normalization over rows or columns."""

        super().__init__(name="rowcol", group_size_threshold=group_size_threshold)
        self.axis = axis


class GcMode(str, Enum):
    """Policies for when staged garbage collection should run."""

    INTERVAL = "interval"
    ON_STAGE_END = "on_stage_end"


@dataclass
class Fallback:
    """Low-sample fallback strategy for modules with weak calibration coverage."""

    strategy: FallbackStrategy = FallbackStrategy.RTN # enable fallback by default due to moe routing behavior breaking calibration based quantization

    # int/float = if captured module fwd tokens is less than value, trigger strategy
    # string = if string is int/float followed by %, then if captured module fwd tokens is less than value in percentage relative to calibration, trigger strategy
    threshold: int | float | str = "0.5%" # if less than 0.5% of calibration reaches module (think moe) then we trigger per-module fallback quantization

    # Smoothers can help some low-sample fallback cases, but a static default can
    # hurt whole-model RTN quality. Leave smoothing opt-in.
    smooth: Optional[SmoothMethod] = None


@dataclass
class WeightOnlyConfig:
    """Configuration for weight-only fallback quantization flows."""

    method: WeightOnlyMethod = WeightOnlyMethod.RTN
    # Whole-model RTN is noticeably more stable without a smoother by default.
    smooth: Optional[SmoothMethod] = None

    def __post_init__(self):
        """Normalize the weight-only method and optional smoother settings."""

        if isinstance(self.method, str):
            try:
                self.method = WeightOnlyMethod(self.method.lower())
            except ValueError as exc:
                raise ValueError(
                    f"WeightOnlyConfig: `method` must be one of {[v.value for v in WeightOnlyMethod]}."
                ) from exc
        elif not isinstance(self.method, WeightOnlyMethod):
            raise ValueError(
                f"WeightOnlyConfig: `method` must be one of {[v.value for v in WeightOnlyMethod]}."
            )

        self.smooth = _parse_smooth_method(self.smooth)


@dataclass
class BasePreProcessorConfig:
    """Base payload for preprocessing stages emitted into config JSON."""

    code: ClassVar[str] = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the preprocessor config into a minimal dictionary."""

        return {"code": self.code}


@dataclass
class SmootherConfig(BasePreProcessorConfig):
    """Serialized wrapper for a configured smoothing preprocessor."""

    code: ClassVar[str] = PreProcessorCode.SMOOTHER.value
    smooth: Optional[SmoothMethod] = None

    def __post_init__(self):
        """Normalize the smoother payload into a typed smoother instance."""

        self.smooth = _parse_smooth_method(self.smooth)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the smoother config, including the smoother payload."""

        payload = super().to_dict()
        payload["smooth"] = _serialize_smooth_method(self.smooth)
        return payload


@dataclass
class AutoModuleDecoderConfig(BasePreProcessorConfig):
    """Configure automatic module-local decode behavior for checkpoint dtypes such as FP8."""

    code: ClassVar[str] = PreProcessorCode.AUTO_MODULE_DECODER.value
    source_dtype: str = "auto"
    target_dtype: Union[str, torch.dtype] = torch.bfloat16

    def __post_init__(self):
        """Normalize the decoder payload into canonical string and dtype values."""

        source_dtype = str(self.source_dtype).strip().lower()
        if source_dtype != "auto":
            raise ValueError(
                f"AutoModuleDecoderConfig: unsupported `source_dtype` `{self.source_dtype}`."
            )
        self.source_dtype = source_dtype

        target_dtype = self.target_dtype
        if isinstance(target_dtype, torch.dtype):
            normalized_dtype = target_dtype
        else:
            normalized_dtype = _DECODER_TARGET_DTYPE_MAP.get(str(target_dtype).strip().lower())
        if normalized_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                "AutoModuleDecoderConfig: `target_dtype` must be `torch.float16` or `torch.bfloat16`."
            )
        self.target_dtype = normalized_dtype

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decoder config with a stable dtype string payload."""

        payload = super().to_dict()
        payload["source_dtype"] = self.source_dtype
        payload["target_dtype"] = str(self.target_dtype).split(".")[-1]
        return payload


@dataclass
class TensorParallelPadderConfig(BasePreProcessorConfig):
    """Configure tensor-parallel-safe column padding derived from module weight shapes."""

    code: ClassVar[str] = PreProcessorCode.TENSOR_PARALLEL_PADDER.value


@dataclass
class AnalysisConfig(BasePreProcessorConfig):
    """Configure granular pre-quantization error analysis for the active quant config."""

    code: ClassVar[str] = PreProcessorCode.ANALYSIS.value
    top_k: int = 32
    top_k_regions: int = 128
    regions_per_module: int = 8
    emit_markdown: bool = True
    emit_json: bool = True
    include_endpoints: bool = True
    bad_block_rel_rmse_threshold: float = 0.10
    recommendation_percentile: float = 95.0
    min_recommendation_risk: float = 20.0
    promotion_bits: int = 8
    promotion_group_size: int = 32
    fusion_profile: str = "model_definition"
    max_chunk_values: int = 8 * 1024 * 1024
    max_sample_values: int = 1024 * 1024

    def __post_init__(self):
        """Validate report shape and scoring thresholds."""

        for name in (
            "top_k",
            "top_k_regions",
            "regions_per_module",
            "promotion_bits",
            "promotion_group_size",
            "max_chunk_values",
            "max_sample_values",
        ):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise ValueError(f"AnalysisConfig: `{name}` must be an integer.")
            if value <= 0:
                raise ValueError(f"AnalysisConfig: `{name}` must be greater than 0.")
        self.emit_markdown = bool(self.emit_markdown)
        self.emit_json = bool(self.emit_json)
        self.include_endpoints = bool(self.include_endpoints)
        self.bad_block_rel_rmse_threshold = float(self.bad_block_rel_rmse_threshold)
        if self.bad_block_rel_rmse_threshold <= 0:
            raise ValueError("AnalysisConfig: `bad_block_rel_rmse_threshold` must be greater than 0.")
        self.recommendation_percentile = float(self.recommendation_percentile)
        if not 0 < self.recommendation_percentile <= 100:
            raise ValueError("AnalysisConfig: `recommendation_percentile` must be in (0, 100].")
        self.min_recommendation_risk = float(self.min_recommendation_risk)
        if not 0 <= self.min_recommendation_risk <= 100:
            raise ValueError("AnalysisConfig: `min_recommendation_risk` must be in [0, 100].")
        self.fusion_profile = str(self.fusion_profile).strip().lower()
        if self.fusion_profile == "vllm_sglang":
            self.fusion_profile = "model_definition"
        if self.fusion_profile not in {"none", "model_definition"}:
            raise ValueError("AnalysisConfig: `fusion_profile` must be one of `none` or `model_definition`.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the analysis preprocessor config."""

        payload = super().to_dict()
        payload.update(
            {
                "top_k": self.top_k,
                "emit_markdown": self.emit_markdown,
                "emit_json": self.emit_json,
                "bad_block_rel_rmse_threshold": self.bad_block_rel_rmse_threshold,
            }
        )
        optional_defaults = {
            "top_k_regions": 128,
            "regions_per_module": 8,
            "include_endpoints": True,
            "recommendation_percentile": 95.0,
            "min_recommendation_risk": 20.0,
            "promotion_bits": 8,
            "promotion_group_size": 32,
            "fusion_profile": "model_definition",
            "max_chunk_values": 8 * 1024 * 1024,
            "max_sample_values": 1024 * 1024,
        }
        for name, default in optional_defaults.items():
            value = getattr(self, name)
            if value != default:
                payload[name] = value
        return payload


class LengthAwareMode(str, Enum):
    """Length-aware Hessian weighting modes."""

    DISABLED = "disabled"
    SINGLE = "single"
    EQUAL_PER_BUCKET_WEIGHT = "equal_per_bucket_weight"


@dataclass
class LengthAwareConfig:
    """Length-aware Hessian weighting configuration for GPTQ calibration."""

    mode: Union[str, LengthAwareMode] = field(default_factory=lambda: LengthAwareMode.SINGLE)
    min_length: Optional[int] = None
    min_bucket_size: int = 16
    max_bucket_ratio: float = 2.0
    bucket_weight_exponent: float = 1.0
    target_bucket_count: Optional[int] = None
    bucket_boundaries: Optional[List[Optional[Union[int, float]]]] = None
    bucket_scales: Optional[List[float]] = None
    bucket_weights: Optional[List[float]] = None

    def __post_init__(self):
        if isinstance(self.mode, str) and not isinstance(self.mode, LengthAwareMode):
            try:
                self.mode = LengthAwareMode(self.mode.lower())
            except ValueError as exc:
                raise ValueError(f"LengthAwareConfig: invalid mode `{self.mode}`.") from exc
        if not isinstance(self.mode, LengthAwareMode):
            raise ValueError("LengthAwareConfig: `mode` must be a LengthAwareMode or string.")
        if self.min_length is not None and (not isinstance(self.min_length, int) or self.min_length <= 0):
            raise ValueError("LengthAwareConfig: `min_length` must be a positive integer or None.")
        if not isinstance(self.min_bucket_size, int) or self.min_bucket_size <= 0:
            raise ValueError("LengthAwareConfig: `min_bucket_size` must be a positive integer.")
        if (
            not isinstance(self.max_bucket_ratio, (int, float))
            or not math.isfinite(self.max_bucket_ratio)
            or self.max_bucket_ratio <= 1.0
        ):
            raise ValueError("LengthAwareConfig: `max_bucket_ratio` must be a finite number > 1.0.")
        if self.target_bucket_count is not None and (not isinstance(self.target_bucket_count, int) or self.target_bucket_count <= 0):
            raise ValueError("LengthAwareConfig: `target_bucket_count` must be a positive integer or None.")
        if (
            not isinstance(self.bucket_weight_exponent, (int, float))
            or not math.isfinite(self.bucket_weight_exponent)
            or self.bucket_weight_exponent < 0
        ):
            raise ValueError("LengthAwareConfig: `bucket_weight_exponent` must be a finite non-negative number.")
        if self.bucket_boundaries is not None:
            normalized_boundaries: List[Optional[Union[int, float]]] = []
            for boundary in self.bucket_boundaries:
                if boundary is None:
                    normalized_boundaries.append(float("inf"))
                elif isinstance(boundary, (int, float)) and boundary >= 0:
                    normalized_boundaries.append(float(boundary))
                else:
                    raise ValueError("LengthAwareConfig: `bucket_boundaries` must be non-negative numbers or None.")
            self.bucket_boundaries = normalized_boundaries
            if len(self.bucket_boundaries) < 2:
                raise ValueError("LengthAwareConfig: `bucket_boundaries` must define at least one bucket.")
            if any(
                left >= right
                for left, right in zip(self.bucket_boundaries, self.bucket_boundaries[1:])
            ):
                raise ValueError("LengthAwareConfig: `bucket_boundaries` must be strictly increasing.")
            if self.bucket_scales is not None and len(self.bucket_scales) != len(self.bucket_boundaries) - 1:
                raise ValueError("LengthAwareConfig: `bucket_scales` length must match `bucket_boundaries`.")
            if self.bucket_weights is not None and len(self.bucket_weights) != len(self.bucket_boundaries) - 1:
                raise ValueError("LengthAwareConfig: `bucket_weights` length must match `bucket_boundaries`.")
        elif self.bucket_scales is not None or self.bucket_weights is not None:
            raise ValueError(
                "LengthAwareConfig: `bucket_boundaries` are required when bucket scales or weights are provided."
            )
        if self.bucket_scales is not None and not all(
            isinstance(s, (int, float)) and math.isfinite(s) and s > 0 for s in self.bucket_scales
        ):
            raise ValueError("LengthAwareConfig: `bucket_scales` must be finite positive numbers.")
        if self.bucket_weights is not None and not all(
            isinstance(w, (int, float)) and math.isfinite(w) and w > 0 for w in self.bucket_weights
        ):
            raise ValueError("LengthAwareConfig: `bucket_weights` must be finite positive numbers.")

    def __bool__(self) -> bool:
        return self.mode is not LengthAwareMode.DISABLED

    @staticmethod
    def _value_change_split(
        sorted_lengths: List[int],
        lo: int,
        hi: int,
        min_bucket_size: int,
        target: Optional[int] = None,
    ) -> Optional[int]:
        """Return a split index in [lo, hi] at a token-length value change, or None.

        Splits inside a run of identical token lengths would make `from_lengths()`
        bucket counts (taken from index slices) disagree with runtime
        `bisect_right` partitioning (which uses value boundaries).
        """
        if hi - lo < 2 * min_bucket_size:
            return None
        candidates = [
            i for i in range(lo + min_bucket_size, hi - min_bucket_size + 1)
            if sorted_lengths[i - 1] < sorted_lengths[i]
        ]
        if not candidates:
            return None
        if target is None:
            mid = (lo + hi) // 2
            return min(candidates, key=lambda i: (abs(i - mid), i))
        return min(candidates, key=lambda i: (abs(sorted_lengths[i] - target), i))

    @staticmethod
    def _adaptive_bucket_intervals(
        lengths: List[int],
        min_bucket_size: int,
        max_bucket_ratio: float,
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        sorted_lengths = sorted(int(length) for length in lengths if length > 0)
        if not sorted_lengths:
            raise ValueError("LengthAwareConfig: all lengths must be positive.")
        intervals: List[Tuple[int, int]] = []

        def _split(lo: int, hi: int) -> None:
            if hi - lo < 2 * min_bucket_size:
                intervals.append((lo, hi))
                return
            min_len = sorted_lengths[lo]
            max_len = sorted_lengths[hi - 1]
            if max_len <= min_len or max_len / min_len <= max_bucket_ratio:
                intervals.append((lo, hi))
                return
            target = math.isqrt(min_len * max_len)
            mid = LengthAwareConfig._value_change_split(
                sorted_lengths, lo, hi, min_bucket_size, target
            )
            if mid is None:
                intervals.append((lo, hi))
                return
            _split(lo, mid)
            _split(mid, hi)

        _split(0, len(sorted_lengths))
        return intervals, sorted_lengths

    @staticmethod
    def _targeted_bucket_intervals(
        sorted_lengths: List[int],
        target_bucket_count: int,
        min_bucket_size: int,
    ) -> List[Tuple[int, int]]:
        """Split sorted lengths into balanced, value-safe quantile buckets.

        Dynamic programming chooses cumulative cuts nearest the equal-count
        quantiles. Cuts may only land where token length changes, because
        runtime lookup is value-based and cannot assign identical lengths to
        different buckets. If the requested count is infeasible, return the
        largest feasible partition so the caller can report the exact collapse.
        """

        sample_count = len(sorted_lengths)
        value_change_indexes = [
            index
            for index in range(1, sample_count)
            if sorted_lengths[index - 1] < sorted_lengths[index]
        ]

        def _balanced_partition(bucket_count: int) -> Optional[List[Tuple[int, int]]]:
            if bucket_count == 1:
                return [(0, sample_count)]
            if sample_count < bucket_count * min_bucket_size:
                return None

            # DP state is keyed by the most recent value-safe cut. A prefix
            # minimum keeps each cut stage linear in the distinct-length count.
            previous_costs = {0: 0.0}
            parents: List[Dict[int, int]] = []
            for cut_number in range(1, bucket_count):
                ideal_cut = sample_count * cut_number / bucket_count
                current_costs: Dict[int, float] = {}
                current_parents: Dict[int, int] = {}
                previous_items = sorted(previous_costs.items())
                previous_cursor = 0
                best_previous_index: Optional[int] = None
                best_previous_cost = float("inf")

                for cut_index in value_change_indexes:
                    if cut_index < cut_number * min_bucket_size:
                        continue
                    if sample_count - cut_index < (bucket_count - cut_number) * min_bucket_size:
                        continue
                    while (
                        previous_cursor < len(previous_items)
                        and previous_items[previous_cursor][0] <= cut_index - min_bucket_size
                    ):
                        candidate_index, candidate_cost = previous_items[previous_cursor]
                        if candidate_cost < best_previous_cost:
                            best_previous_index = candidate_index
                            best_previous_cost = candidate_cost
                        previous_cursor += 1
                    if best_previous_index is None:
                        continue
                    current_costs[cut_index] = best_previous_cost + (cut_index - ideal_cut) ** 2
                    current_parents[cut_index] = best_previous_index

                if not current_costs:
                    return None
                previous_costs = current_costs
                parents.append(current_parents)

            last_cut = min(previous_costs, key=lambda index: (previous_costs[index], index))
            cuts = [last_cut]
            for parent_map in reversed(parents[1:]):
                last_cut = parent_map[last_cut]
                cuts.append(last_cut)
            cuts.reverse()
            points = [0, *cuts, sample_count]
            return list(zip(points, points[1:]))

        for bucket_count in range(target_bucket_count, 0, -1):
            intervals = _balanced_partition(bucket_count)
            if intervals is not None:
                return intervals
        return [(0, sample_count)]

    @classmethod
    def from_lengths(
        cls,
        lengths: List[int],
        mode: Union[str, LengthAwareMode] = LengthAwareMode.SINGLE,
        min_length: Optional[int] = None,
        min_bucket_size: int = 16,
        max_bucket_ratio: float = 2.0,
        bucket_weight_exponent: float = 1.0,
        target_bucket_count: Optional[int] = None,
    ) -> "LengthAwareConfig":
        """Build a LengthAwareConfig with adaptive buckets computed from token lengths."""
        sorted_lengths = sorted(int(length) for length in lengths if length > 0)
        if not sorted_lengths:
            raise ValueError("LengthAwareConfig: all lengths must be positive.")
        if isinstance(mode, str) and not isinstance(mode, LengthAwareMode):
            mode = LengthAwareMode(mode.lower())
        if target_bucket_count is not None:
            intervals = cls._targeted_bucket_intervals(
                sorted_lengths, target_bucket_count=target_bucket_count, min_bucket_size=min_bucket_size
            )
            if len(intervals) != target_bucket_count:
                raise ValueError(
                    "LengthAwareConfig: calibration bucket calculation collapsed from "
                    f"target_bucket_count={target_bucket_count} to resolved_bucket_count={len(intervals)} "
                    f"non-empty value-based bucket(s) from {len(sorted_lengths)} sequence(s), "
                    f"{len(set(sorted_lengths))} unique length(s), range=[{sorted_lengths[0]},{sorted_lengths[-1]}], "
                    f"min_bucket_size={min_bucket_size}. Add more length-diverse calibration data, lower "
                    "target_bucket_count, or lower min_bucket_size when calibration row count is the constraint."
                )
        else:
            intervals = cls._adaptive_bucket_intervals(
                lengths, min_bucket_size=min_bucket_size, max_bucket_ratio=max_bucket_ratio
            )[0]
        # Bucket boundaries must be monotonic; intervals may be created out of order.
        intervals = sorted(intervals, key=lambda x: x[0])
        boundaries: List[Optional[int]] = [0]
        scales: List[float] = []
        counts: List[int] = []
        for lo, hi in intervals:
            bucket = sorted_lengths[lo:hi]
            scales.append(float(sum(bucket) / len(bucket)))
            counts.append(len(bucket))
            if hi < len(sorted_lengths):
                boundaries.append(int(sorted_lengths[hi]))
        boundaries.append(None)
        total = len(sorted_lengths)
        non_empty_buckets = sum(1 for count in counts if count > 0)
        if non_empty_buckets == 0 or total == 0:
            weights = None
        else:
            weights = [float((non_empty_buckets * count / total) ** bucket_weight_exponent) for count in counts]
        return cls(
            mode=mode,
            min_length=min_length,
            min_bucket_size=min_bucket_size,
            max_bucket_ratio=max_bucket_ratio,
            bucket_weight_exponent=bucket_weight_exponent,
            target_bucket_count=target_bucket_count,
            bucket_boundaries=boundaries,
            bucket_scales=scales,
            bucket_weights=weights if mode is LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        bucket_boundaries = None
        if self.bucket_boundaries is not None:
            bucket_boundaries = [
                None if b == float("inf") else int(b) if b.is_integer() else b
                for b in self.bucket_boundaries
            ]
        return {
            "mode": self.mode.value,
            "min_length": self.min_length,
            "min_bucket_size": self.min_bucket_size,
            "max_bucket_ratio": self.max_bucket_ratio,
            "bucket_weight_exponent": self.bucket_weight_exponent,
            "target_bucket_count": self.target_bucket_count,
            "bucket_boundaries": bucket_boundaries,
            "bucket_scales": self.bucket_scales,
            "bucket_weights": self.bucket_weights,
        }


@dataclass
class HessianConfig:
    """Controls for chunked Hessian accumulation during GPTQ calibration."""

    chunk_size: Optional[int] = field(default=None, metadata={"help": "Maximum rows per Hessian chunk"})
    chunk_bytes: Optional[int] = field(default=None, metadata={"help": "Memory budget (in bytes) for Hessian chunk staging"})
    staging_dtype: Union[str, torch.dtype] = field(
        default=torch.float32,
        metadata={"help": "Stage Hessian chunks in a lower precision dtype when supported"},
    )
    dedup_shared_inputs: bool = field(
        default=True,
        metadata={
            "help": "Collect the Hessian once per explicit `:in=<tag>` shared-input group and copy it to the "
                    "other group members instead of accumulating it separately for each module"
        },
    )
    length_aware: Union[bool, str, LengthAwareConfig] = field(
        default_factory=lambda: LengthAwareConfig(
            mode=LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT,
            target_bucket_count=6,
            bucket_weight_exponent=0.2,
        ),
        metadata={"help": "Enable length-aware per-sequence Hessian normalization (MaCa); "
                          "use a LengthAwareConfig for advanced modes."},
    )

    def __post_init__(self):
        """Validate Hessian chunking and staging dtype settings."""

        if not isinstance(self.dedup_shared_inputs, bool):
            raise ValueError("HessianConfig: `dedup_shared_inputs` must be a bool.")

        if self.chunk_size is not None:
            if not isinstance(self.chunk_size, int):
                raise ValueError("HessianConfig: `chunk_size` must be an integer or None.")
            if self.chunk_size <= 0:
                raise ValueError("HessianConfig: `chunk_size` must be a positive integer.")

        if self.chunk_bytes is not None:
            if not isinstance(self.chunk_bytes, int):
                raise ValueError("HessianConfig: `chunk_bytes` must be an integer or None.")
            if self.chunk_bytes <= 0:
                raise ValueError("HessianConfig: `chunk_bytes` must be a positive integer amount of bytes.")

        if isinstance(self.staging_dtype, str):
            self.staging_dtype = self.staging_dtype.lower()
            if self.staging_dtype not in ["float32", "float16", "bfloat16"]:
                raise ValueError("HessianConfig: `staging_dtype` must be float32, float16, or bfloat16.")
            self.staging_dtype = getattr(torch, self.staging_dtype)
        elif isinstance(self.staging_dtype, torch.dtype):
            if self.staging_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                raise ValueError("HessianConfig: `staging_dtype` must be float32, float16, or bfloat16.")
        else:
            raise ValueError("HessianConfig: `staging_dtype` must be a torch.dtype or string.")

        if self.length_aware is None:
            self.length_aware = LengthAwareConfig(mode=LengthAwareMode.DISABLED)
        elif isinstance(self.length_aware, bool):
            self.length_aware = LengthAwareConfig(
                mode=LengthAwareMode.SINGLE if self.length_aware else LengthAwareMode.DISABLED
            )
        elif isinstance(self.length_aware, str):
            self.length_aware = LengthAwareConfig(mode=self.length_aware)
        elif isinstance(self.length_aware, dict):
            self.length_aware = LengthAwareConfig(**self.length_aware)
        elif isinstance(self.length_aware, LengthAwareConfig):
            self.length_aware = copy.deepcopy(self.length_aware)
        else:
            raise ValueError("HessianConfig: `length_aware` must be None, bool, string, dict, or LengthAwareConfig.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_bytes": self.chunk_bytes,
            "staging_dtype": str(self.staging_dtype).split(".")[-1],
            "dedup_shared_inputs": self.dedup_shared_inputs,
            "length_aware": self.length_aware.to_dict() if self.length_aware.mode is not LengthAwareMode.DISABLED else None,
        }


@dataclass
class DampConfig:
    """Static Hessian damping configuration.

    Static damping uses a single percdamp value: ``min == max`` and ``step`` is
    the increment used for Cholesky failure recovery. The same type serves as
    the base for :class:`AdaptiveDampingConfig`.
    """

    min: float = field(default=GPTQ_DEFAULT_DAMP_PERCENT)
    max: float = field(default=GPTQ_DEFAULT_DAMP_PERCENT)
    step: float = field(default=GPTQ_DEFAULT_DAMP_AUTO_INCREMENT)

    def __post_init__(self):
        if not (0 < self.min < 1):
            raise ValueError("DampConfig: `min` must be between 0 and 1.")
        if not (0 < self.max < 1):
            raise ValueError("DampConfig: `max` must be between 0 and 1.")
        if self.max < self.min:
            raise ValueError("DampConfig: `max` must be greater than or equal to `min`.")
        if not (0 <= self.step < 1):
            raise ValueError("DampConfig: `step` must be between 0 and 1.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": self.min,
            "max": self.max,
            "step": self.step,
        }


@dataclass
class AdaptiveDampingConfig(DampConfig):
    """Calibration-aware Hessian damping controls for GPTQ.

    The default adaptive path changes only the regularization of GPTQ's
    activation Hessian.  For calibration activations ``X`` and
    ``H = (2 / N) X.T @ X`` it selects

        r = lambda_max(H) / mean(diag(H))
        damp_percent = clamp(base_percdamp * r ** spectral_alpha, min, max)
        H_damped = H + damp_percent * mean(diag(H)) * I

    GPTQ then uses the canonical inverse-Hessian error correction unchanged:
    ``E_j = (W_j - Q_j) / U_jj`` and
    ``W[:, j:] -= E_j outer U[j, j:]``, where
    ``U = chol(inv(H_damped), upper=True)``. Thus the adaptive signal comes
    from collected calibration activations, never from weights alone.

    Module-role and group-size priors are metadata heuristics, while online
    feedback scales the canonical GPTQ correction. They remain available for
    experiments but are disabled by default so opting into adaptive damping
    does not silently replace GPTQ's error-correction mathematics. If online
    feedback is explicitly enabled, its canonical group loss is
    ``0.5 * sum(E ** 2)``. Optional raw-Hessian diagonal weighting is also
    experimental because ``E`` already contains inverse-Hessian geometry.

    ``lambda_max`` is estimated with ``method`` (``power_iteration``,
    ``lanczos``, or ``diagonal``). See ``docs/adaptive_damping_math.md`` for
    derivation, invariants, fallbacks, and the validation matrix.
    """

    enabled: bool = field(default=True)
    base_percdamp: float = field(default=0.05)
    module_prior_enabled: bool = field(default=False)
    module_factors: Dict[str, float] = field(
        default_factory=lambda: {
            "q": 1.0,
            "k": 1.0,
            "v": 1.0,
            "o": 1.0,
            "gate": 1.1,
            "up": 1.1,
            "down": 0.9,
        }
    )
    method: str = field(default="power_iteration")
    eigen_iterations: int = field(default=10)
    spectral_alpha: float = field(default=0.25)
    min: float = field(default=0.02)
    max: float = field(default=0.08)
    step: float = field(default=0.01)
    group_error_enabled: bool = field(default=True)
    group_error_gamma: float = field(default=0.1)
    group_error_ema_decay: float = field(default=0.9)
    group_error_factor_min: float = field(default=0.9)
    group_error_factor_max: float = field(default=1.1)
    online_feedback_enabled: bool = field(default=False)
    group_error_scale_min: float = field(default=0.8)
    group_error_scale_max: float = field(default=1.2)
    group_error_use_hessian_weighting: bool = field(default=False)
    group_error_measure_raw_residual: bool = field(default=True)
    group_size_prior_enabled: bool = field(default=False)
    group_size_prior_beta: float = field(default=0.25)
    group_size_prior_reference: int = field(default=128)

    def __post_init__(self):
        super().__post_init__()
        self.enabled = bool(self.enabled)
        self.module_prior_enabled = bool(self.module_prior_enabled)
        if not (0 < self.base_percdamp < 1):
            raise ValueError("AdaptiveDampingConfig: `base_percdamp` must be between 0 and 1.")
        if not (0 < self.spectral_alpha):
            raise ValueError("AdaptiveDampingConfig: `spectral_alpha` must be positive.")
        if not isinstance(self.eigen_iterations, int) or self.eigen_iterations <= 0:
            raise ValueError("AdaptiveDampingConfig: `eigen_iterations` must be a positive integer.")
        if self.method not in {"power_iteration", "diagonal", "lanczos"}:
            raise ValueError(
                "AdaptiveDampingConfig: `method` must be one of {'power_iteration', 'diagonal', 'lanczos'}."
            )
        if not isinstance(self.module_factors, dict):
            raise ValueError("AdaptiveDampingConfig: `module_factors` must be a dict.")
        for k, v in self.module_factors.items():
            if not isinstance(k, str) or not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(
                    "AdaptiveDampingConfig: `module_factors` must map module short names to positive floats."
                )
        self.online_feedback_enabled = bool(self.online_feedback_enabled)
        self.group_error_enabled = bool(self.group_error_enabled)
        self.group_size_prior_enabled = bool(self.group_size_prior_enabled)
        if self.group_error_gamma <= 0:
            raise ValueError("AdaptiveDampingConfig: `group_error_gamma` must be positive.")
        if not (0 < self.group_error_ema_decay < 1):
            raise ValueError("AdaptiveDampingConfig: `group_error_ema_decay` must be between 0 and 1.")
        if self.group_size_prior_beta < 0:
            raise ValueError("AdaptiveDampingConfig: `group_size_prior_beta` must be non-negative.")
        if self.group_size_prior_reference <= 0:
            raise ValueError("AdaptiveDampingConfig: `group_size_prior_reference` must be positive.")
        if not (0 < self.group_error_factor_min < self.group_error_factor_max):
            raise ValueError(
                "AdaptiveDampingConfig: `group_error_factor_min` must be positive and less than `group_error_factor_max`."
            )
        if not (0 < self.group_error_scale_min < self.group_error_scale_max):
            raise ValueError(
                "AdaptiveDampingConfig: `group_error_scale_min` must be positive and less than `group_error_scale_max`."
            )
        self.group_error_use_hessian_weighting = bool(self.group_error_use_hessian_weighting)
        self.group_error_measure_raw_residual = bool(self.group_error_measure_raw_residual)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "enabled": self.enabled,
                "base_percdamp": self.base_percdamp,
                "module_prior_enabled": self.module_prior_enabled,
                "module_factors": dict(self.module_factors),
                "method": self.method,
                "eigen_iterations": self.eigen_iterations,
                "spectral_alpha": self.spectral_alpha,
                "group_error_enabled": self.group_error_enabled,
                "group_error_gamma": self.group_error_gamma,
                "group_error_ema_decay": self.group_error_ema_decay,
                "group_error_factor_min": self.group_error_factor_min,
                "group_error_factor_max": self.group_error_factor_max,
                "online_feedback_enabled": self.online_feedback_enabled,
                "group_error_scale_min": self.group_error_scale_min,
                "group_error_scale_max": self.group_error_scale_max,
                "group_error_use_hessian_weighting": self.group_error_use_hessian_weighting,
                "group_error_measure_raw_residual": self.group_error_measure_raw_residual,
                "group_size_prior_enabled": self.group_size_prior_enabled,
                "group_size_prior_beta": self.group_size_prior_beta,
                "group_size_prior_reference": self.group_size_prior_reference,
            }
        )
        return d


@dataclass
class AdaptiveClippingConfig:
    """Calibration-aware adaptive weight-clipping search for GPTQ.

    Searches a per-group clipping threshold before scale/zero selection so that
    outlier weights do not dominate the quantization range. ``gptq_error`` is
    the safe default: it evaluates every candidate with the same sequential
    inverse-Hessian correction used by GPTQ. ``hessian_diag`` is an explicit
    diagonal approximation and ``mse`` is an explicit weight-only objective.

    The feature remains opt-in at ``QuantizeConfig.adaptive_clipping``. When
    ``gptq_error`` is selected but a valid damped inverse-Cholesky factor is not
    available, the search conservatively uses the full, unclipped range rather
    than silently changing to a weight-only objective.
    """

    enabled: bool = field(default=True)
    metric: Union[str, AdaptiveClippingMetric] = field(default=AdaptiveClippingMetric.GPTQ_ERROR)
    per_group: bool = field(default=True)
    candidates: Tuple[float, ...] = field(default=(0.99, 0.995, 0.999, 1.0))

    def __post_init__(self):
        self.enabled = bool(self.enabled)
        if isinstance(self.metric, AdaptiveClippingMetric):
            self.metric = self.metric.value
        valid_metrics = {
            AdaptiveClippingMetric.GPTQ_ERROR.value,
            AdaptiveClippingMetric.MSE.value,
            AdaptiveClippingMetric.HESSIAN_DIAG.value,
        }
        if self.metric not in valid_metrics:
            raise ValueError(
                "AdaptiveClippingConfig: `metric` must be one of "
                "{'gptq_error', 'mse', 'hessian_diag'}."
            )
        self.per_group = bool(self.per_group)
        if not isinstance(self.candidates, (list, tuple)) or len(self.candidates) == 0:
            raise ValueError("AdaptiveClippingConfig: `candidates` must be a non-empty list or tuple of floats in (0, 1].")
        normalized = []
        for c in self.candidates:
            if not isinstance(c, (int, float)) or not (0 < c <= 1.0):
                raise ValueError("AdaptiveClippingConfig: each candidate must be in (0, 1].")
            normalized.append(float(c))
        self.candidates = tuple(normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "metric": self.metric,
            "per_group": self.per_group,
            "candidates": list(self.candidates),
        }


@dataclass
class GPTAQConfig:
    alpha: float = field(default=0.25)
    device: Union[str, torch.device] = field(default="auto")

    def __post_init__(self):
        if not isinstance(self.alpha, (int, float)):
            raise ValueError("GPTAQConfig: `alpha` must be a numeric value.")
        if isinstance(self.device, str):
            if not self.device:
                raise ValueError("GPTAQConfig: `device` must be a non-empty string or torch.device.")
        elif not isinstance(self.device, torch.device):
            raise ValueError("GPTAQConfig: `device` must be a string or torch.device.")


@dataclass
class FOEMConfig:
    r"""Configuration parameters for the FOEM calibration process, including `alpha` and `beta`.

    The parameter `alpha` follows the same definition and role as in GPTAQ.
    Note: although GPTAQ does not explicitly mention this coefficient in the paper,
    its official implementation applies it to the rightmost term of Eq.18.

    The parameter `beta` is introduced by FOEM. Please refer to the paper for details:
    https://ojs.aaai.org/index.php/AAAI/article/view/40123.

    Special cases:
        - alpha = 0, beta = 0:
            Equivalent to GPTQ.
        - alpha > 0, beta = 0:
            Equivalent to GPTAQ. The recommended value for `alpha` is 0.25.
        - alpha = 0, beta > 0:
            Equivalent to FOEM. Empirically, setting `beta` in the range [0.1, 0.25] yields good performance.
        - alpha > 0, beta > 0:
            Equivalent to FOEM + GPTAQ. Using the default best settings
            (alpha = 0.25, beta = 0.2) generally produces strong results,
            although it is not consistently superior to using FOEM alone.

    Args:
        alpha (float, optional): Default is 0.
        beta (float, optional): Default is 0.2.
    """
    alpha: float = field(default=0)
    beta: float = field(default=0.2)
    device: Union[str, torch.device] = field(default="auto")

    def __post_init__(self):
        if not isinstance(self.alpha, (int, float)):
            raise ValueError("FOEMConfig: `alpha` must be a numeric value.")
        if not isinstance(self.beta, (int, float)):
            raise ValueError("FOEMConfig: `beta` must be a numeric value.")
        if isinstance(self.device, str):
            if not self.device:
                raise ValueError("FOEMConfig: `device` must be a non-empty string or torch.device.")
        elif not isinstance(self.device, torch.device):
            raise ValueError("FOEMConfig: `device` must be a string or torch.device.")


@dataclass
class MoERoutingConfig:
    """Base configuration for model routing behavior during MoE quantization."""


MOE_ALL_EXPERTS = "all"


@dataclass
class ExpertsRoutingOverride(MoERoutingConfig):
    num_experts_per_tok: Union[int, str] = MOE_ALL_EXPERTS

    def __post_init__(self):
        # Handle string values
        if isinstance(self.num_experts_per_tok, str):
            raw = self.num_experts_per_tok.strip()

            # Numeric string -> int (must be > 0)
            if raw.isdigit():
                value = int(raw)
                if value <= 0:
                    raise ValueError(
                        f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                        f"got '{self.num_experts_per_tok}'"
                    )
                self.num_experts_per_tok = value
                return

            # Normalize keyword string
            value = raw.lower()
            if value != MOE_ALL_EXPERTS:
                raise ValueError(
                    f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                    f"got '{self.num_experts_per_tok}'"
                )

            self.num_experts_per_tok = value
            return

        # Validate integer values
        if not isinstance(self.num_experts_per_tok, int) or self.num_experts_per_tok <= 0:
            raise ValueError(
                f"num_experts_per_tok must be a positive int or '{MOE_ALL_EXPERTS}', "
                f"got {self.num_experts_per_tok}"
            )


# MoE quantization: forward whole calibration dataset to each expert instead of only routed data
# This ensures all experts receive sufficient calibration samples but increases quantization time
@dataclass
class ExpertsRoutingBypass(MoERoutingConfig):
    """Route every calibration observation to every routed expert."""


@dataclass
class MoEExecutionConfig:
    """Controls how MoE quantization work is scheduled without changing routing semantics."""

    # Number of projection modules processed in one subset. None keeps the full
    # module-tree subset; zero disables batching and a positive value limits peak VRAM.
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of modules to process in a single batch during MoE quantization"}
    )
    # Parallel capture is effective only with free-threaded Python and multiple
    # visible CUDA devices; the runtime eligibility gate remains authoritative.
    parallel_input_capture: bool = field(
        default=True,
        metadata={
            "help": (
                "Run independent MoE bypass input-capture groups concurrently across CUDA devices. "
                "Requires a free-threaded Python runtime with GIL disabled and multiple visible CUDA GPUs."
            )
        },
    )
    parallel_input_capture_streams: int = field(default=2)
    parallel_output_replay: bool = field(default=True)

    def __post_init__(self):
        if self.batch_size is not None:
            if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 0:
                raise ValueError("MoEExecutionConfig: `batch_size` must be a non-negative integer or None.")
        if not isinstance(self.parallel_input_capture, bool):
            raise ValueError("MoEExecutionConfig: `parallel_input_capture` must be a boolean.")
        if isinstance(self.parallel_input_capture_streams, bool) or not isinstance(self.parallel_input_capture_streams, int) or self.parallel_input_capture_streams <= 0:
            raise ValueError("MoEExecutionConfig: `parallel_input_capture_streams` must be a positive integer.")
        if not isinstance(self.parallel_output_replay, bool):
            raise ValueError("MoEExecutionConfig: `parallel_output_replay` must be a boolean.")


@dataclass
class MoEConfig:
    routing: MoERoutingConfig
    execution: MoEExecutionConfig = field(default_factory=MoEExecutionConfig)

    def __post_init__(self):
        if not isinstance(self.routing, MoERoutingConfig):
            raise ValueError(
                f"routing must be an instance of MoERoutingConfig, "
                f"got {type(self.routing).__name__}"
            )
        if not isinstance(self.execution, MoEExecutionConfig):
            raise ValueError(
                f"execution must be an instance of MoEExecutionConfig, "
                f"got {type(self.execution).__name__}"
            )

    def routing_bypass(self) -> bool:
        return isinstance(self.routing, ExpertsRoutingBypass)

    def routing_override(self, num_experts: int) -> Union[int, None]:
        """
        Resolve MoE routing top-k override.

        Returns the effective number of experts per token if routing override
        is enabled, otherwise None.

        - "all" resolves to `num_experts`
        - integer value is returned directly
        """
        if isinstance(self.routing, ExpertsRoutingOverride):
            # Resolve "all" to full expert count
            if isinstance(self.routing.num_experts_per_tok, str) and self.routing.num_experts_per_tok.lower().strip() == MOE_ALL_EXPERTS:
                return num_experts

            assert isinstance(self.routing.num_experts_per_tok, int)
            top_k = self.routing.num_experts_per_tok

            # Clamp to valid range and warn user if needed
            if top_k > num_experts:
                log.info(f"MoEConfig: MoE routing override num_experts_per_tok ({top_k}) exceeds "
                    f"num_experts ({num_experts}); clamping to {num_experts}.",)
                top_k = num_experts

            return top_k

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing": {
                "class": self.routing.__class__.__name__,
                **asdict(self.routing),
            },
            "execution": asdict(self.execution),
        }


QUANT_METHOD_FORMAT_MAPPING = {
    METHOD.GPTQ: {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.GPTQ_P,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
    },
    METHOD.FP8: {
        FORMAT.FP8,
    },
    METHOD.BITSANDBYTES: {
        FORMAT.BITSANDBYTES,
    },
    METHOD.EXL3: {
        FORMAT.EXL3,
    },
    METHOD.QVQ: {
        FORMAT.QVQ,
        FORMAT.QVQ_V4,
        FORMAT.QVQ_V4_L18,
        FORMAT.QVQ_DUAL_V2,
        FORMAT.QVQ_V2B4_P64,
        FORMAT.QVQ_V2B2_P32,
    },
    METHOD.GGUF: {
        FORMAT.GGUF,
    },
    METHOD.QQQ: {
        FORMAT.QQQ,
    },
    METHOD.AWQ: {
        FORMAT.GEMM,
        FORMAT.GEMV,
        FORMAT.GEMV_FAST,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
        FORMAT.LLM_AWQ,
    },
    METHOD.PARO: {
        FORMAT.PAROQUANT,
    },
    METHOD.MXFP4: {
        FORMAT.MXFP4,
    },
}

GPTQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GPTQ,
    FORMAT.GPTQ_V2,
    FORMAT.GPTQ_P,
    FORMAT.MARLIN,
    FORMAT.BITBLAS,
)
AWQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GEMM,
    FORMAT.GEMV,
    FORMAT.GEMV_FAST,
    FORMAT.MARLIN,
    FORMAT.BITBLAS,
    FORMAT.LLM_AWQ,
)
PAROQUANT_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.PAROQUANT,
)
# Keep ParoQuant channel-scale clamps configurable so users can relax or
# tighten the safeguard without patching the optimizer code.
PAROQUANT_OPT_SCALE_CLAMP_MIN_DEFAULT = 1e-2
PAROQUANT_OPT_SCALE_CLAMP_MAX_DEFAULT = 1e2
QQQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.QQQ,
)
FP8_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.FP8,
)
BITSANDBYTES_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.BITSANDBYTES,
)
EXL3_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.EXL3,
)
QVQ_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.QVQ,
    FORMAT.QVQ_V4,
    FORMAT.QVQ_V4_L18,
    FORMAT.QVQ_DUAL_V2,
    FORMAT.QVQ_V2B4_P64,
    FORMAT.QVQ_V2B2_P32,
)
RTN_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GPTQ,
    FORMAT.GPTQ_V2,
    FORMAT.GPTQ_P,
    FORMAT.GEMM,
    FORMAT.GEMV,
    FORMAT.GEMV_FAST,
    FORMAT.LLM_AWQ,
)
GGUF_EXPORT_FORMATS: Tuple[FORMAT, ...] = (
    FORMAT.GGUF,
)

_UNAMBIGUOUS_EXPORT_METHOD_BY_FORMAT = {
    FORMAT.GPTQ: METHOD.GPTQ,
    FORMAT.GPTQ_V2: METHOD.GPTQ,
    FORMAT.GPTQ_P: METHOD.GPTQ,
    FORMAT.FP8: METHOD.FP8,
    FORMAT.BITSANDBYTES: METHOD.BITSANDBYTES,
    FORMAT.EXL3: METHOD.EXL3,
    FORMAT.QVQ: METHOD.QVQ,
    FORMAT.QVQ_V4: METHOD.QVQ,
    FORMAT.QVQ_V4_L18: METHOD.QVQ,
    FORMAT.QVQ_DUAL_V2: METHOD.QVQ,
    FORMAT.QVQ_V2B4_P64: METHOD.QVQ,
    FORMAT.QVQ_V2B2_P32: METHOD.QVQ,
    FORMAT.GGUF: METHOD.GGUF,
    FORMAT.BITBLAS: METHOD.GPTQ,
    FORMAT.GEMM: METHOD.AWQ,
    FORMAT.GEMV: METHOD.AWQ,
    FORMAT.GEMV_FAST: METHOD.AWQ,
    FORMAT.LLM_AWQ: METHOD.AWQ,
    FORMAT.PAROQUANT: METHOD.PARO,
    FORMAT.QQQ: METHOD.QQQ,
    FORMAT.MXFP4: METHOD.MXFP4,
}

# Inference-only methods should go here. QVQ owns a dedicated calibration
# processor and must never be routed through the affine GPTQ processor.
QUANTIZE_BLACK_LIST = {}

# compat
QUANT_CONFIG_ARG_SYNONYMS = {
    "w_bit": BITS_FIELD_CODE,

    # QQQ compat
    "wbits": BITS_FIELD_CODE,
    "q_group_size": GROUP_SIZE_FIELD_CODE,

    # AWQ compat
    "version" : FORMAT_FIELD_CODE,

    # map deprecated aliases to canonical fields
    FORMAT_FIELD_CHECKPOINT: FORMAT_FIELD_CODE,
    QUANT_METHOD_FIELD: METHOD_FIELD_CODE,
    "bnb_quant_type": FORMAT_FIELD_CODE,
    "bnb_block_size": "block_size",
    "bnb_compress_statistics": "compress_statistics",

    # QVQ draft compatibility: `activation` is the canonical field.
    "activation_quantization": "activation",
}

# compat (values are negated)
QUANT_CONFIG_ARG_SYNONYMS_NEGATED = {
    # AWQ compat
    "zero_point": SYMMETRIC_FIELD_CODE,
}
DYNAMIC_FIELD_SYNONYMS = {}

# Sentinel used by the dynamic override cache to indicate no pattern matched.
_DYNAMIC_NO_MATCH = object()


@dataclass
class _DynamicResolutionCacheEntry:
    # Keep the owner alive while its id is a cache key. Without this reference,
    # CPython may recycle the id for an unrelated config and return stale rules.
    dynamic: Dict[str, Dict[str, Any]]
    patterns: List[Tuple[bool, Any, Dict[str, Any], Optional[str], int]]
    exact_lookup: Dict[str, Tuple[int, Union[Dict[str, Any], bool]]]
    all_exact: bool
    regex_patterns: List[Tuple[int, bool, Any, Dict[str, Any]]]
    override_cache: OrderedDict[str, Any] = field(default_factory=OrderedDict)


# Dynamic rules are shared by deep-copied per-module configs. Bound both cache
# dimensions so repeated model/config lifecycles cannot retain memory forever.
_DYNAMIC_CACHE_MAX_CONFIGS = 256
_DYNAMIC_CACHE_MAX_OVERRIDES = 65_536
_DYNAMIC_CACHE_LOCK = threading.RLock()
_DYNAMIC_CACHE: OrderedDict[int, _DynamicResolutionCacheEntry] = OrderedDict()
_DYNAMIC_CACHE_OVERRIDE_COUNT = 0


def _extract_literal_regex_pattern(raw: str) -> Optional[str]:
    """If `raw` is a regex that matches a single literal string, return that string."""
    # PCRE `.match()` is start-anchored only, so a trailing `$` is required for a
    # true exact full-string match.  A leading `^` is also required to avoid
    # matching arbitrary prefixes.
    if not raw.startswith("^") or not raw.endswith("$"):
        return None
    raw = raw[1:-1]
    out = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\":
            if i + 1 >= n:
                return None
            nxt = raw[i + 1]
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
            if nxt.isalnum():
                # Regex escape such as \d, \w, \1, \x, etc.
                return None
            # Escaped special character -> literal.
            out.append(nxt)
            i += 2
            continue
        if ch in ".^$*+?{}[]()|":
            return None
        out.append(ch)
        i += 1
    return "".join(out)


def _build_dynamic_cache_entry(
    dynamic: Dict[str, Dict[str, Any]],
) -> _DynamicResolutionCacheEntry:
    patterns = []
    exact_lookup: Dict[str, Tuple[int, Union[Dict[str, Any], bool]]] = {}
    regex_patterns: List[Tuple[int, bool, Any, Dict[str, Any]]] = []
    all_exact = True
    for index, (pattern, source_overrides) in enumerate(list(dynamic.items())):
        is_negative = pattern.startswith("-:") or source_overrides is False
        if source_overrides is False:
            overrides = {}
        elif isinstance(source_overrides, dict):
            overrides = dict(source_overrides)
        else:
            raise TypeError(
                f"QuantizeConfig: dynamic override `{pattern}` must be a dictionary or False, "
                f"got {type(source_overrides).__name__}."
            )
        raw = pattern[2:] if pattern.startswith(("-:", "+:")) else pattern
        exact_literal = _extract_literal_regex_pattern(raw)
        if exact_literal is None:
            all_exact = False
            try:
                compiled = pcre.compile(raw)
            except Exception as exc:
                raise ValueError(f"QuantizeConfig: invalid dynamic pattern `{pattern}`") from exc
            regex_patterns.append((index, is_negative, compiled, overrides))
        else:
            compiled = None
            if exact_literal not in exact_lookup:
                exact_lookup[exact_literal] = (
                    index,
                    False if is_negative else overrides,
                )
        patterns.append((is_negative, compiled, overrides, exact_literal, index))
    return _DynamicResolutionCacheEntry(
        dynamic=dynamic,
        patterns=patterns,
        exact_lookup=exact_lookup,
        all_exact=all_exact,
        regex_patterns=regex_patterns,
    )


def _trim_dynamic_cache_locked() -> None:
    global _DYNAMIC_CACHE_OVERRIDE_COUNT

    while len(_DYNAMIC_CACHE) > _DYNAMIC_CACHE_MAX_CONFIGS:
        _, evicted = _DYNAMIC_CACHE.popitem(last=False)
        _DYNAMIC_CACHE_OVERRIDE_COUNT -= len(evicted.override_cache)

    while _DYNAMIC_CACHE_OVERRIDE_COUNT > _DYNAMIC_CACHE_MAX_OVERRIDES:
        _, oldest = next(iter(_DYNAMIC_CACHE.items()))
        if oldest.override_cache:
            oldest.override_cache.popitem(last=False)
            _DYNAMIC_CACHE_OVERRIDE_COUNT -= 1
        else:
            _DYNAMIC_CACHE.move_to_end(id(oldest.dynamic))


def _get_dynamic_cache_entry(dynamic: Dict[str, Dict[str, Any]]) -> _DynamicResolutionCacheEntry:
    """Return the bounded, identity-safe cache entry for one dynamic config."""

    cache_key = id(dynamic)
    with _DYNAMIC_CACHE_LOCK:
        cached = _DYNAMIC_CACHE.get(cache_key)
        if cached is not None and cached.dynamic is dynamic:
            _DYNAMIC_CACHE.move_to_end(cache_key)
            return cached

    candidate = _build_dynamic_cache_entry(dynamic)
    with _DYNAMIC_CACHE_LOCK:
        cached = _DYNAMIC_CACHE.get(cache_key)
        if cached is not None and cached.dynamic is dynamic:
            _DYNAMIC_CACHE.move_to_end(cache_key)
            return cached
        _DYNAMIC_CACHE[cache_key] = candidate
        _DYNAMIC_CACHE.move_to_end(cache_key)
        _trim_dynamic_cache_locked()
        return candidate


def _get_dynamic_patterns(dynamic: Dict[str, Dict[str, Any]]) -> List[Tuple[bool, Any, Dict[str, Any], Optional[str], int]]:
    """Return compiled PCRE patterns (plus optional exact literals)."""

    return _get_dynamic_cache_entry(dynamic).patterns


def _resolve_dynamic_override(
    dynamic: Dict[str, Dict[str, Any]],
    module_name: str,
) -> Union[Dict[str, Any], bool, None]:
    """Return the first matching dynamic override dict, False for negative, or None."""

    if dynamic is None:
        return None

    global _DYNAMIC_CACHE_OVERRIDE_COUNT
    cache_key = id(dynamic)
    entry = _get_dynamic_cache_entry(dynamic)
    with _DYNAMIC_CACHE_LOCK:
        cached = entry.override_cache.get(module_name, _DYNAMIC_NO_MATCH)
        if cached is not _DYNAMIC_NO_MATCH:
            entry.override_cache.move_to_end(module_name)
            return cached

    # Fast path: every pattern is an exact literal module name.
    if entry.all_exact:
        exact_entry = entry.exact_lookup.get(module_name)
        matched = exact_entry[1] if exact_entry is not None else None
    else:
        # Mixed fallback: find the earliest matching pattern among exact
        # literals (O(1) lookup) and ordered regex patterns.
        exact_entry = entry.exact_lookup.get(module_name)
        best_index = exact_entry[0] if exact_entry is not None else None
        matched = exact_entry[1] if exact_entry is not None else None

        for index, is_negative, compiled, overrides in entry.regex_patterns:
            if best_index is not None and index > best_index:
                break
            if compiled.match(module_name):
                matched = False if is_negative else dict(overrides)
                break

    with _DYNAMIC_CACHE_LOCK:
        if _DYNAMIC_CACHE.get(cache_key) is entry:
            existing = entry.override_cache.get(module_name, _DYNAMIC_NO_MATCH)
            if existing is _DYNAMIC_NO_MATCH:
                entry.override_cache[module_name] = matched
                _DYNAMIC_CACHE_OVERRIDE_COUNT += 1
            else:
                matched = existing
                entry.override_cache.move_to_end(module_name)
            _DYNAMIC_CACHE.move_to_end(cache_key)
            _trim_dynamic_cache_locked()
    return matched


def dict_scale_dtype_to_str(d: Dict[str, Any]) -> None:
    """
    Checks whether the passed dictionary and its nested dicts have a *scale_dtype* key and if it's not None,
    converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
    string, which can then be stored in the json format.
    """
    if d.get("scale_dtype", None) is not None and not isinstance(d["scale_dtype"], str):
        d["scale_dtype"] = str(d["scale_dtype"]).split(".")[1]
    for value in d.values():
        if isinstance(value, dict):
            dict_scale_dtype_to_str(value)


def _build_smooth_method_from_dict(payload: Dict[str, Any]) -> Optional[SmoothMethod]:
    method_type = payload.get("type") or payload.get("name")
    if not method_type:
        return None
    method_type = str(method_type).strip().lower()
    group_size_threshold_raw = payload.get("group_size_threshold", 128)
    group_size_threshold = int(group_size_threshold_raw) if group_size_threshold_raw is not None else 128
    if method_type == "percentile":
        return SmoothPercentile(
            percentile=float(payload.get("percentile", 99.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type in ("percentile_asym", "percentile_asymmetric"):
        return SmoothPercentileAsymmetric(
            low=float(payload.get("low", 0.5)),
            high=float(payload.get("high", 99.5)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "mad":
        return SmoothMAD(
            k=float(payload.get("k", 3.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "mse":
        return SmoothMSE(
            steps=int(payload.get("steps", 32)),
            maxshrink=float(payload.get("maxshrink", 0.8)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "outlier":
        return SmoothOutlier(
            pct=float(payload.get("pct", 1.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "softnorm":
        return SmoothSoftNorm(
            k=float(payload.get("k", 3.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "log":
        return SmoothLog(
            percentile=float(payload.get("percentile", 99.0)),
            mu=float(payload.get("mu", 8.0)),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "rowcol":
        return SmoothRowCol(
            axis=str(payload.get("axis", "row")),
            group_size_threshold=group_size_threshold,
        )
    if method_type == "none":
        return None
    raise ValueError(f"QuantizeConfig: Unknown smooth type `{method_type}`.")


def _parse_smooth_method(setting: Any) -> Optional[SmoothMethod]:
    if setting is None:
        return None
    if isinstance(setting, SmoothMethod):
        return setting
    if isinstance(setting, str):
        return _build_smooth_method_from_dict({"type": setting})
    if isinstance(setting, dict):
        return _build_smooth_method_from_dict(setting)
    raise ValueError("QuantizeConfig: `fallback.smooth` must be a SmoothMethod, string, or dict.")


def _serialize_smooth_method(method: Optional[SmoothMethod]) -> Optional[Dict[str, Any]]:
    if method is None:
        return None

    payload = {"type": method.name, "group_size_threshold": method.group_size_threshold}
    if isinstance(method, SmoothPercentile):
        payload["percentile"] = method.percentile
    elif isinstance(method, SmoothPercentileAsymmetric):
        payload["low"] = method.low
        payload["high"] = method.high
    elif isinstance(method, SmoothMAD):
        payload["k"] = method.k
    elif isinstance(method, SmoothMSE):
        payload["steps"] = method.steps
        payload["maxshrink"] = method.maxshrink
    elif isinstance(method, SmoothOutlier):
        payload["pct"] = method.pct
    elif isinstance(method, SmoothSoftNorm):
        payload["k"] = method.k
    elif isinstance(method, SmoothLog):
        payload["percentile"] = method.percentile
        payload["mu"] = method.mu
    elif isinstance(method, SmoothRowCol):
        payload["axis"] = method.axis
    return payload


def _normalize_smoother_config(
    payload: Optional[Union[SmootherConfig, SmoothMethod, Dict[str, Any], str]]
) -> Optional[SmootherConfig]:
    if payload is None:
        return None
    if isinstance(payload, SmootherConfig):
        return payload
    if isinstance(payload, dict) and "smooth" in payload and "type" not in payload:
        return SmootherConfig(smooth=payload.get("smooth"))
    return SmootherConfig(smooth=payload)


def _normalize_preprocessor_config(payload: Any) -> BasePreProcessorConfig:
    if isinstance(payload, BasePreProcessorConfig):
        return payload
    if isinstance(payload, SmoothMethod):
        return SmootherConfig(smooth=payload)
    if isinstance(payload, str):
        normalized = payload.strip().lower()
        if normalized == PreProcessorCode.SMOOTHER.value:
            return SmootherConfig(smooth=None)
        if normalized == PreProcessorCode.AUTO_MODULE_DECODER.value:
            return AutoModuleDecoderConfig()
        if normalized == PreProcessorCode.TENSOR_PARALLEL_PADDER.value:
            return TensorParallelPadderConfig()
        if normalized == PreProcessorCode.ANALYSIS.value:
            return AnalysisConfig()
        return SmootherConfig(smooth=payload)
    if isinstance(payload, dict):
        code = str(payload.get("code", "")).strip().lower()
        if code == PreProcessorCode.AUTO_MODULE_DECODER.value:
            return AutoModuleDecoderConfig(
                source_dtype=payload.get("source_dtype", "auto"),
                target_dtype=payload.get("target_dtype", torch.bfloat16),
            )
        if code == PreProcessorCode.TENSOR_PARALLEL_PADDER.value:
            return TensorParallelPadderConfig()
        if code == PreProcessorCode.ANALYSIS.value:
            return AnalysisConfig(
                top_k=payload.get("top_k", 32),
                top_k_regions=payload.get("top_k_regions", 128),
                regions_per_module=payload.get("regions_per_module", 8),
                emit_markdown=payload.get("emit_markdown", True),
                emit_json=payload.get("emit_json", True),
                include_endpoints=payload.get("include_endpoints", True),
                bad_block_rel_rmse_threshold=payload.get("bad_block_rel_rmse_threshold", 0.10),
                recommendation_percentile=payload.get("recommendation_percentile", 95.0),
                min_recommendation_risk=payload.get("min_recommendation_risk", 20.0),
                promotion_bits=payload.get("promotion_bits", 8),
                promotion_group_size=payload.get("promotion_group_size", 32),
                fusion_profile=payload.get("fusion_profile", "model_definition"),
                max_chunk_values=payload.get("max_chunk_values", 8 * 1024 * 1024),
                max_sample_values=payload.get("max_sample_values", 1024 * 1024),
            )
        if code and code != PreProcessorCode.SMOOTHER.value:
            raise ValueError(f"QuantizeConfig: unsupported preprocessor code `{code}`.")
        if "smooth" in payload:
            return SmootherConfig(smooth=payload.get("smooth"))
        if "type" in payload:
            return SmootherConfig(smooth=payload)
        return SmootherConfig(smooth=None)
    raise ValueError("QuantizeConfig: `preprocessors` entries must be preprocessor configs, smooth configs, dicts, or strings.")


def _normalize_preprocessors(payload: Optional[List[Any]]) -> List[BasePreProcessorConfig]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("QuantizeConfig: `preprocessors` must be a list or None.")
    return [_normalize_preprocessor_config(item) for item in payload]


def _validate_unique_preprocessors(preprocessors: List[BasePreProcessorConfig]) -> None:
    codes_seen = set()
    for preprocessor in preprocessors:
        if preprocessor.code in codes_seen:
            raise ValueError(f"QuantizeConfig: duplicate preprocessor `{preprocessor.code}` is not allowed.")
        codes_seen.add(preprocessor.code)


def dynamic_get(dynamic: Dict[str, Dict[str, Union[int, bool]]], module_name: str, key: str = None,
                default: Union[int, bool] = None, sub_key: str = None) -> Union[Dict, int, bool]:

    if dynamic is None:
        return default

    overrides = _resolve_dynamic_override(dynamic, module_name)
    if overrides is False:
        return False
    if overrides is None:
        return default

    if key is None:
        return overrides

    if key in overrides:
        sub_value = overrides[key]
    elif key in DYNAMIC_FIELD_SYNONYMS:
        sub_value = None
        for legacy_key in DYNAMIC_FIELD_SYNONYMS[key]:
            if legacy_key in overrides:
                sub_value = overrides[legacy_key]
                break
    else:
        return default

    if sub_key:
        if isinstance(sub_value, Dict):
            return sub_value.get(sub_key, default)
        log.info(f"QuantConfig: Dynamic `sub_key`: `{sub_key}` failed extraction from  `sub_value`: `{sub_value}`")
        return default

    return sub_value


def _normalize_quant_method(value: Union[str, METHOD]) -> METHOD:
    if isinstance(value, str):
        value = value.lower()
        if value in {"qvq_v2b2_g32", "v2b2_g32", "v2b2-g32"}:
            return METHOD.QVQ
        if value == FORMAT.MARLIN:
            return METHOD.GPTQ
        if value == FORMAT.BITBLAS:
            return METHOD.GPTQ
        if value == FORMAT.FP8:
            return METHOD.FP8
        if value == FORMAT.BITSANDBYTES:
            return METHOD.BITSANDBYTES
        if value == FORMAT.EXL3:
            return METHOD.EXL3
        if value == FORMAT.QVQ:
            return METHOD.QVQ
        if value == FORMAT.QVQ_V4:
            return METHOD.QVQ
        if value == FORMAT.QVQ_V4_L18:
            return METHOD.QVQ
        if value == FORMAT.QVQ_DUAL_V2:
            return METHOD.QVQ
        if value == FORMAT.QVQ_V2B4_P64:
            return METHOD.QVQ
        if value == FORMAT.QVQ_V2B2_P32:
            return METHOD.QVQ
        if value == FORMAT.PAROQUANT:
            return METHOD.PARO
        if value == FORMAT.MXFP4:
            return METHOD.MXFP4
        try:
            return METHOD(value)
        except ValueError as exc:
            raise ValueError(f"QuantizeConfig: Unknown quantization method: `{value}`.") from exc
    if not isinstance(value, METHOD):
        raise ValueError(f"QuantizeConfig: Unsupported `method`: {value}")
    return value


def normalize_scale_search(
    value: Optional[Union[str, ScaleSearchConfig]],
) -> Optional[ScaleSearchConfig]:
    """Normalize the public scale-search selector while preserving ``None`` as disabled."""

    if value is None:
        return None
    if isinstance(value, ScaleSearchConfig):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return ScaleSearchConfig(normalized)
        except ValueError as exc:
            raise ValueError(
                "QuantizeConfig: `scale_search` must be one of "
                f"{[method.value for method in ScaleSearchConfig]} or None, got `{value}`."
            ) from exc
    raise ValueError(
        "QuantizeConfig: `scale_search` must be a ScaleSearchConfig, string, or None."
    )


def _normalize_adjacent_model(value: Optional[Any]) -> Optional[Any]:
    """Restore a validated AdjacentExact policy from serialized metadata."""

    if value is None:
        return None

    from .adjacent_model import AdjacentModelConfig

    if isinstance(value, AdjacentModelConfig):
        return value
    if isinstance(value, dict):
        return AdjacentModelConfig.from_dict(value)
    raise TypeError("QuantizeConfig: `adjacent_model` must be an AdjacentModelConfig or dictionary.")


def _serialize_adjacent_model(value: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Serialize an AdjacentExact policy without its per-run statistics."""

    normalized = _normalize_adjacent_model(value)
    return normalized.to_dict() if normalized is not None else None


def _normalize_format(value: Union[str, FORMAT]) -> FORMAT:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"qvq_v2b2_g32", "v2b2_g32", "v2b2-g32"}:
            normalized = FORMAT.QVQ_V2B2_P32.value
        try:
            return FORMAT(normalized)
        except ValueError as exc:
            raise ValueError(f"QuantizeConfig: Unknown quantization format: `{value}`.") from exc
    if not isinstance(value, FORMAT):
        raise ValueError(f"QuantizeConfig: Unknown quantization format: `{value}`.")
    return value


def _normalize_pack_dtype(pack_dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if pack_dtype is None:
        return torch.int32
    if isinstance(pack_dtype, str):
        pack_dtype = pack_dtype.lower()
        if pack_dtype not in ["int64", "int32", "int16", "int8"]:
            raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")
        return getattr(torch, pack_dtype)
    if isinstance(pack_dtype, torch.dtype):
        if pack_dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
            raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")
        return pack_dtype
    raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {pack_dtype}")


def _normalize_paroquant_best_state_dtype(best_state_dtype: Optional[Union[str, torch.dtype]]) -> str:
    """Canonicalize the ParoQuant best-state snapshot dtype into a serialized string."""
    if best_state_dtype is None:
        return "fp32"
    if isinstance(best_state_dtype, str):
        normalized = best_state_dtype.strip().lower()
        if normalized in {"fp16", "float16"}:
            return "fp16"
        if normalized in {"bf16", "bfloat16"}:
            return "bf16"
        if normalized in {"fp32", "float32"}:
            return "fp32"
    elif isinstance(best_state_dtype, torch.dtype):
        if best_state_dtype == torch.float16:
            return "fp16"
        if best_state_dtype == torch.bfloat16:
            return "bf16"
        if best_state_dtype == torch.float32:
            return "fp32"
    raise ValueError(
        "ParoConfig: `opt_best_state_dtype` must be one of {'fp16', 'bf16', 'fp32'} "
        "or torch.float16/torch.bfloat16/torch.float32."
    )


def _normalize_fallback(fallback: Optional[Union[Fallback, Dict[str, Any], str, int, float]]) -> Optional[Fallback]:
    if fallback is None:
        return None
    if isinstance(fallback, dict):
        strategy = fallback.get("strategy", FallbackStrategy.RTN)
        threshold = fallback.get("threshold", "1.0%")
        smooth = fallback.get("smooth")
        if smooth is None:
            smooth = fallback.get("smooth_method")
        if smooth is None and "clip_method" in fallback:
            smooth = fallback.get("clip_method")
        smooth = _parse_smooth_method(smooth)
        if smooth is None:
            if "smooth_percentile" in fallback:
                smooth = SmoothPercentile(percentile=float(fallback.get("smooth_percentile", 99.0)))
            elif "smooth_mad_k" in fallback:
                smooth = SmoothMAD(k=float(fallback.get("smooth_mad_k", 3.0)))
            elif "smooth_mse_steps" in fallback or "smooth_mse_maxshrink" in fallback:
                smooth = SmoothMSE(
                    steps=int(fallback.get("smooth_mse_steps", 32)),
                    maxshrink=float(fallback.get("smooth_mse_maxshrink", 0.8)),
                )
            elif "smooth_outlier_pct" in fallback:
                smooth = SmoothOutlier(pct=float(fallback.get("smooth_outlier_pct", 1.0)))
            elif "smooth_rms_k" in fallback:
                smooth = SmoothSoftNorm(k=float(fallback.get("smooth_rms_k", 3.0)))
            elif "smooth_log_mu" in fallback:
                smooth = SmoothLog(
                    percentile=float(fallback.get("smooth_percentile", 99.0)),
                    mu=float(fallback.get("smooth_log_mu", 8.0)),
                )
            elif "smooth_axis" in fallback:
                smooth = SmoothRowCol(axis=str(fallback.get("smooth_axis", "row")))
        fallback = Fallback(strategy=strategy, threshold=threshold, smooth=smooth)
    elif isinstance(fallback, (str, int, float)):
        fallback = Fallback(strategy=FallbackStrategy.RTN, threshold=fallback)
    elif not isinstance(fallback, Fallback):
        raise ValueError("QuantizeConfig: `fallback` must be a Fallback config, dict, string, int, float, or None.")

    if isinstance(fallback.strategy, str):
        try:
            fallback.strategy = FallbackStrategy(fallback.strategy.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `fallback.strategy` must be one of {[v.value for v in FallbackStrategy]}."
            ) from exc
    elif not isinstance(fallback.strategy, FallbackStrategy):
        raise ValueError(
            f"QuantizeConfig: `fallback.strategy` must be one of {[v.value for v in FallbackStrategy]}."
        )

    fallback.smooth = _parse_smooth_method(fallback.smooth)
    return fallback


def _normalize_weight_only(
    weight_only: Optional[Union[WeightOnlyConfig, Dict[str, Any], str]]
) -> Optional[WeightOnlyConfig]:
    if weight_only is None:
        return None
    if isinstance(weight_only, dict):
        method = weight_only.get("method", WeightOnlyMethod.RTN)
        smooth = weight_only.get("smooth")
        if smooth is None:
            smooth = weight_only.get("smooth_method")
        return WeightOnlyConfig(method=method, smooth=smooth)
    if isinstance(weight_only, str):
        return WeightOnlyConfig(method=weight_only)
    if not isinstance(weight_only, WeightOnlyConfig):
        raise ValueError(
            "QuantizeConfig: `weight_only` must be a WeightOnlyConfig, dict, string, or None."
        )
    return weight_only


def _normalize_hessian(hessian: Optional[Union[HessianConfig, Dict[str, Any]]]) -> HessianConfig:
    if hessian is None:
        return HessianConfig()
    if isinstance(hessian, dict):
        return HessianConfig(**hessian)
    if not isinstance(hessian, HessianConfig):
        raise ValueError("QuantizeConfig: `hessian` must be a HessianConfig, dict, or None.")
    return hessian


def _normalize_gptaq(gptaq: Optional[Union[GPTAQConfig, Dict[str, Any]]]) -> Optional[GPTAQConfig]:
    if gptaq is None:
        return None
    if isinstance(gptaq, dict):
        return GPTAQConfig(**gptaq)
    if not isinstance(gptaq, GPTAQConfig):
        raise ValueError("QuantizeConfig: `gptaq` must be a GPTAQConfig, dict, or None.")
    return gptaq


def _normalize_foem(foem: Optional[Union[FOEMConfig, Dict[str, Any]]]) -> Optional[FOEMConfig]:
    if foem is None:
        return None
    if isinstance(foem, dict):
        return FOEMConfig(**foem)
    if not isinstance(foem, FOEMConfig):
        raise ValueError("QuantizeConfig: `foem` must be a FOEMConfig, dict, or None.")
    return foem


def _normalize_damp(
    damp: Optional[Union[DampConfig, AdaptiveDampingConfig, Dict[str, Any]]]
) -> Union[DampConfig, AdaptiveDampingConfig]:
    if damp is None:
        # Preserve the established GPTQ numerical contract unless adaptive
        # damping is explicitly requested. Adaptive damping changes both the
        # Hessian factor and sequential residual updates, so enabling it through
        # an omitted option silently changes the quantized checkpoint.
        return DampConfig()
    if isinstance(damp, (DampConfig, AdaptiveDampingConfig)):
        return damp
    if isinstance(damp, dict):
        payload = dict(damp)
        # Backward-compatible aliases.
        if "min_percdamp" in payload and "min" not in payload:
            payload["min"] = payload.pop("min_percdamp")
        if "max_percdamp" in payload and "max" not in payload:
            payload["max"] = payload.pop("max_percdamp")
        if "estimation_method" in payload and "method" not in payload:
            payload["method"] = payload.pop("estimation_method")
        # Flatten nested v3 config groups.
        if "spectral" in payload and isinstance(payload["spectral"], dict):
            spectral = payload.pop("spectral")
            payload.setdefault("method", spectral.get("estimator", "power_iteration"))
            payload.setdefault("eigen_iterations", spectral.get("iterations", 10))
            payload.setdefault("spectral_alpha", spectral.get("alpha", 0.25))
        if "clamp" in payload and isinstance(payload["clamp"], dict):
            clamp = payload.pop("clamp")
            payload.setdefault("min", clamp.get("min"))
            payload.setdefault("max", clamp.get("max"))
        if "module_prior" in payload and isinstance(payload["module_prior"], dict):
            mp = payload.pop("module_prior")
            payload.setdefault("module_prior_enabled", mp.get("enabled", True))
        if "group" in payload and isinstance(payload["group"], dict):
            g = payload.pop("group")
            payload.setdefault("group_error_enabled", g.get("error_enabled", g.get("enabled", False)))
            payload.setdefault("group_error_gamma", g.get("error_gamma", g.get("gamma", 0.1)))
            payload.setdefault("group_size_prior_enabled", g.get("size_prior_enabled", False))
            payload.setdefault("group_size_prior_beta", g.get("size_prior_beta", 0.25))
        # Deprecated keys no longer have a field; drop them to avoid dataclass errors.
        payload.pop("target_condition", None)
        adaptive_keys = {
            "enabled",
            "base_percdamp",
            "module_prior_enabled",
            "module_factors",
            "method",
            "eigen_iterations",
            "spectral_alpha",
            "group_error_enabled",
            "group_error_gamma",
            "group_error_ema_decay",
            "group_error_factor_min",
            "group_error_factor_max",
            "online_feedback_enabled",
            "group_error_scale_min",
            "group_error_scale_max",
            "group_error_use_hessian_weighting",
            "group_error_measure_raw_residual",
            "group_size_prior_enabled",
            "group_size_prior_beta",
            "group_size_prior_reference",
        }
        if any(k in payload for k in adaptive_keys):
            return AdaptiveDampingConfig(**payload)
        return DampConfig(**payload)
    raise ValueError("QuantizeConfig: `damp` must be a DampConfig, AdaptiveDampingConfig, dict, or None.")


def _normalize_adaptive_damping(
    adaptive_damping: Optional[Union[AdaptiveDampingConfig, Dict[str, Any]]]
) -> Union[DampConfig, AdaptiveDampingConfig]:
    """Backward-compatible alias for :func:`_normalize_damp`."""
    return _normalize_damp(adaptive_damping)


def _normalize_adaptive_clipping(
    adaptive_clipping: Optional[Union["AdaptiveClippingConfig", Dict[str, Any]]]
) -> Optional["AdaptiveClippingConfig"]:
    if adaptive_clipping is None:
        return None
    if isinstance(adaptive_clipping, AdaptiveClippingConfig):
        return adaptive_clipping
    if isinstance(adaptive_clipping, dict):
        return AdaptiveClippingConfig(**adaptive_clipping)
    raise ValueError("QuantizeConfig: `adaptive_clipping` must be an AdaptiveClippingConfig, dict, or None.")


def _normalize_fused_forward_config(
    fused_forward: Optional[Union[FusedForwardConfig, Dict[str, Any]]],
) -> Optional[FusedForwardConfig]:
    if fused_forward is None:
        return None
    if isinstance(fused_forward, dict):
        return FusedForwardConfig(**fused_forward)
    if isinstance(fused_forward, bool):
        return FusedForwardConfig() if fused_forward else None
    if not isinstance(fused_forward, FusedForwardConfig):
        raise ValueError(
            "QuantizeConfig: `fused_forward` must be a FusedForwardConfig, dict, bool, or None."
        )
    return fused_forward


def _normalize_dense_vram_strategy(value: Union[str, VramStrategy]) -> VramStrategy:
    """Validate one user-supplied dense-pool placement strategy value."""

    if isinstance(value, str):
        try:
            return VramStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `dense_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            ) from exc
    if not isinstance(value, VramStrategy):
        raise ValueError(
            f"QuantizeConfig: `dense_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
        )
    return value


def _normalize_moe_vram_strategy(value: Union[str, VramStrategy]) -> VramStrategy:
    """Validate one user-supplied MoE expert-pool placement strategy value."""

    if isinstance(value, str):
        try:
            return VramStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `moe_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
            ) from exc
    if not isinstance(value, VramStrategy):
        raise ValueError(
            f"QuantizeConfig: `moe_vram_strategy` must be one of {[v.value for v in VramStrategy]}."
        )
    return value


def _normalize_strategy_devices(
    value: Optional[List[Union[str, torch.device]]],
    *,
    field_name: str,
) -> Optional[List[str]]:
    """Normalize one user-facing strategy device pool to stable device strings."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"QuantizeConfig: `{field_name}` must be a list of device strings or torch.device values.")
    if not value:
        raise ValueError(f"QuantizeConfig: `{field_name}` must not be empty when provided.")

    # Import lazily to keep config parsing light and avoid depending on looper
    # modules unless the caller actually configures explicit device pools.
    from ..utils.looper_helpers import normalize_device_like

    normalized_devices: List[str] = []
    seen = set()
    for raw_device in value:
        normalized = normalize_device_like(raw_device)
        if normalized is None:
            raise ValueError(f"QuantizeConfig: `{field_name}` contains an unsupported device value: {raw_device!r}.")
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        normalized_devices.append(key)
    return normalized_devices


def _normalize_gc_mode(value: Union[str, GcMode]) -> GcMode:
    if isinstance(value, str):
        try:
            return GcMode(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
            ) from exc
    if not isinstance(value, GcMode):
        raise ValueError(
            f"QuantizeConfig: `gc_mode` must be one of {[v.value for v in GcMode]}."
        )
    return value


def _normalize_shard_strategy(value: Optional[Union[str, ShardStrategy]]) -> Optional[ShardStrategy]:
    if value is None:
        return None
    if isinstance(value, ShardStrategy):
        return value
    if isinstance(value, str):
        try:
            return ShardStrategy(value.lower())
        except ValueError as exc:
            raise ValueError(
                f"QuantizeConfig: `shard_strategy` must be one of {[v.value for v in ShardStrategy]}."
            ) from exc
    raise ValueError(
        "QuantizeConfig: `shard_strategy` must be a ShardStrategy, str, or None."
    )


def _normalize_moe_config(value: Optional[Union[MoEConfig, Dict[str, Any]]]) -> Optional[MoEConfig]:
    if value is None:
        return None
    if isinstance(value, MoEConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("QuantizeConfig: `moe` must be a MoEConfig, dict, or None.")

    routing = value.get("routing")
    execution = value.get("execution")
    if execution is None:
        execution_obj = MoEExecutionConfig()
    elif isinstance(execution, MoEExecutionConfig):
        execution_obj = execution
    elif isinstance(execution, dict):
        execution_obj = MoEExecutionConfig(
            batch_size=execution.get("batch_size"),
            parallel_input_capture=execution.get("parallel_input_capture", True),
            parallel_input_capture_streams=execution.get("parallel_input_capture_streams", 2),
            parallel_output_replay=execution.get("parallel_output_replay", True),
        )
    else:
        raise ValueError("QuantizeConfig: `moe.execution` must be a MoEExecutionConfig, dict, or None.")

    if isinstance(routing, MoERoutingConfig):
        return MoEConfig(routing=routing, execution=execution_obj)
    if not isinstance(routing, dict):
        raise ValueError("QuantizeConfig: `moe.routing` must be a MoERoutingConfig, dict, or None.")

    routing_class = routing.get("class")
    if routing_class == MoERoutingConfig.__name__:
        routing_obj = MoERoutingConfig()
    elif routing_class == ExpertsRoutingOverride.__name__:
        routing_obj = ExpertsRoutingOverride(
            num_experts_per_tok=routing.get("num_experts_per_tok", MOE_ALL_EXPERTS)
        )
    elif routing_class == ExpertsRoutingBypass.__name__:
        routing_obj = ExpertsRoutingBypass()
    else:
        raise ValueError(f"QuantizeConfig: Unknown `moe.routing.class`: `{routing_class}`.")

    return MoEConfig(routing=routing_obj, execution=execution_obj)


def _resolve_dynamic_group_size_error() -> str:
    return "QuantizeConfig: `group_size` must be one of `[-1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]`."


def _default_damp_percent(method: METHOD) -> float:
    return 0.005 if method == METHOD.QQQ else GPTQ_DEFAULT_DAMP_PERCENT


def _default_damp_auto_increment(method: METHOD) -> float:
    return 0.001 if method == METHOD.QQQ else GPTQ_DEFAULT_DAMP_AUTO_INCREMENT


def _peek_weight_only_method(payload: Any) -> Optional[WeightOnlyMethod]:
    if payload is None:
        return None
    if isinstance(payload, WeightOnlyConfig):
        return payload.method
    if isinstance(payload, str):
        try:
            return WeightOnlyMethod(payload.lower())
        except ValueError:
            return None
    if isinstance(payload, dict):
        method = payload.get("method", WeightOnlyMethod.RTN)
        try:
            return WeightOnlyMethod(str(method).lower())
        except ValueError:
            return None
    return None


def _extract_weight_only_smooth(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, WeightOnlyConfig):
        return payload.smooth
    if isinstance(payload, dict):
        smooth = payload.get("smooth")
        if smooth is None:
            smooth = payload.get("smooth_method")
        return smooth
    if isinstance(payload, str):
        return None
    raise ValueError("QuantizeConfig: `weight_only` must be a WeightOnlyConfig, dict, string, or None.")


def _extract_weight_only_legacy_gguf_bits(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, WeightOnlyConfig):
        return getattr(payload, "gguf_qtype", None)
    if isinstance(payload, dict):
        return payload.get("gguf_qtype")
    if isinstance(payload, str):
        return None
    raise ValueError("QuantizeConfig: `weight_only` must be a WeightOnlyConfig, dict, string, or None.")


def _normalize_rtn_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    legacy_gguf_bits = normalized.pop("gguf_qtype", None)
    weight_only = normalized.pop("weight_only", None)
    weight_only_method = _peek_weight_only_method(weight_only)

    # `weight_only.method="gguf"` is a backward-compatible shorthand for the direct GGUF weight-only lifecycle.
    if weight_only_method == WeightOnlyMethod.GGUF and FORMAT_FIELD_CODE not in normalized:
        normalized[FORMAT_FIELD_CODE] = FORMAT.GGUF

    if "smooth" not in normalized:
        normalized["smooth"] = _extract_weight_only_smooth(weight_only)
    if legacy_gguf_bits is None:
        legacy_gguf_bits = _extract_weight_only_legacy_gguf_bits(weight_only)
    if legacy_gguf_bits is not None and BITS_FIELD_CODE not in normalized:
        normalized[BITS_FIELD_CODE] = legacy_gguf_bits
    return normalized


def _normalize_gguf_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    legacy_gguf_bits = normalized.pop("gguf_qtype", None)
    weight_only = normalized.pop("weight_only", None)

    if "smoother" not in normalized and "smooth" not in normalized:
        normalized["smoother"] = _extract_weight_only_smooth(weight_only)
    if legacy_gguf_bits is None:
        legacy_gguf_bits = _extract_weight_only_legacy_gguf_bits(weight_only)
    if legacy_gguf_bits is not None and BITS_FIELD_CODE not in normalized:
        normalized[BITS_FIELD_CODE] = legacy_gguf_bits
    normalized[BITS_FIELD_CODE], normalized[FORMAT_FIELD_CODE], _ = _normalize_gguf_config_spec(
        normalized.get(BITS_FIELD_CODE, 4),
        normalized.get(FORMAT_FIELD_CODE),
    )
    return normalized


def _normalize_fp8_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    weight_only = normalized.pop("weight_only", None)
    legacy_fmt = normalized.pop("fmt", None)

    if "smoother" not in normalized and "smooth" not in normalized:
        normalized["smoother"] = _extract_weight_only_smooth(weight_only)

    normalized[FORMAT_FIELD_CODE] = _normalize_fp8_fmt(
        normalized.get(FORMAT_FIELD_CODE, legacy_fmt)
    )

    weight_block_size = _normalize_fp8_weight_block_size(normalized.get("weight_block_size"))
    normalized["weight_block_size"] = list(weight_block_size) if weight_block_size is not None else None

    normalized["weight_scale_method"] = _normalize_fp8_weight_scale_method(
        normalized.get("weight_scale_method"),
        weight_block_size=weight_block_size,
    )
    return normalized


def _normalize_bitsandbytes_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    weight_only = normalized.pop("weight_only", None)

    if "smoother" not in normalized and "smooth" not in normalized:
        normalized["smoother"] = _extract_weight_only_smooth(weight_only)

    legacy_format = normalized.pop("bnb_quant_type", None)
    legacy_block_size = normalized.pop("bnb_block_size", None)
    legacy_compress_statistics = normalized.pop("bnb_compress_statistics", None)

    normalized[FORMAT_FIELD_CODE] = _normalize_bitsandbytes_format(
        normalized.get(FORMAT_FIELD_CODE, legacy_format),
        bits=int(normalized.get(BITS_FIELD_CODE, 4)),
    )
    normalized["block_size"] = _normalize_bitsandbytes_block_size(
        normalized.get("block_size", legacy_block_size)
    )
    normalized["compress_statistics"] = bool(
        normalized.get("compress_statistics", legacy_compress_statistics if legacy_compress_statistics is not None else True)
    )
    return normalized


def _resolve_export_quant_method(format_value: FORMAT, fallback_method: Optional[METHOD] = None) -> METHOD:
    if format_value == FORMAT.MARLIN:
        if fallback_method is None:
            raise ValueError("QuantizeConfig: FORMAT.MARLIN requires an explicit quantization method family.")
        return fallback_method

    method = _UNAMBIGUOUS_EXPORT_METHOD_BY_FORMAT.get(format_value)
    if method is None:
        if fallback_method is not None:
            return fallback_method
        raise ValueError(f"QuantizeConfig: Unable to resolve export method for format `{format_value}`.")
    return method


def _normalize_quantize_config_payload_for_target_cls(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)

    if target_cls is AWQConfig:
        expected_method = METHOD.AWQ
    elif target_cls is FP8Config:
        expected_method = METHOD.FP8
    elif target_cls is BitsAndBytesConfig:
        expected_method = METHOD.BITSANDBYTES
    elif target_cls is EXL3Config:
        expected_method = METHOD.EXL3
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format is not None and normalized_format != FORMAT.EXL3:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.EXL3}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.EXL3
    elif target_cls is QVQConfig:
        expected_method = METHOD.QVQ
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format not in {
            FORMAT.QVQ,
            FORMAT.QVQ_V4,
            FORMAT.QVQ_V4_L18,
            FORMAT.QVQ_DUAL_V2,
            FORMAT.QVQ_V2B4_P64,
            FORMAT.QVQ_V2B2_P32,
        }:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.QVQ}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.QVQ
    elif target_cls is ParoConfig:
        expected_method = METHOD.PARO
        # ParoQuant does not implement GPTQ activation ordering. Accept legacy
        # payloads that serialized the inherited field, but do not expose it to
        # ParoConfig's constructor.
        normalized.pop("desc_act", None)
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format is not None and normalized_format != FORMAT.PAROQUANT:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.PAROQUANT}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.PAROQUANT
    elif target_cls is GGUFConfig:
        expected_method = METHOD.GGUF
    elif target_cls is QQQConfig:
        expected_method = METHOD.QQQ
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format is not None and normalized_format != FORMAT.QQQ:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.QQQ}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.QQQ
    elif target_cls is MXFP4Config:
        expected_method = METHOD.MXFP4
        format_value = normalized.get(FORMAT_FIELD_CODE)
        normalized_format = None
        if format_value is not None:
            try:
                normalized_format = _normalize_format(format_value)
                normalized[FORMAT_FIELD_CODE] = normalized_format
            except ValueError:
                normalized_format = None
        if normalized_format is not None and normalized_format != FORMAT.MXFP4:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.MXFP4}`")
            normalized[FORMAT_FIELD_CODE] = FORMAT.MXFP4
    else:
        expected_method = METHOD.GPTQ

    method = normalized.get(METHOD_FIELD_CODE)
    normalized_method = None
    if method is not None:
        try:
            normalized_method = _normalize_quant_method(method)
            normalized[METHOD_FIELD_CODE] = normalized_method
        except ValueError:
            normalized_method = None

    if normalized_method is not None and normalized_method != expected_method:
        if target_cls is GGUFConfig and normalized_method == METHOD.GPTQ:
            pass
        else:
            log.warn(
                f"QuantizeConfig: `{METHOD_FIELD_CODE}`=`{normalized_method}` is incompatible with `{target_cls.__name__}`. "
                f"Auto-fix method to `{expected_method}`."
            )
        normalized[METHOD_FIELD_CODE] = expected_method

    return normalized


def _filter_quantize_config_payload_for_target_cls(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    target_field_names = {field.name for field in fields(target_cls) if field.init}
    return {key: value for key, value in payload.items() if key in target_field_names}


def _prepare_target_quantize_config_kwargs(target_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, payload)
    target_field_names = {field.name for field in fields(target_cls) if field.init}
    if normalized.get("adjacent_model") is not None and "adjacent_model" not in target_field_names:
        raise ValueError(
            "QuantizeConfig: `adjacent_model` currently supports GPTQ and AWQ quantization only."
        )
    if target_cls is RTNConfig:
        normalized = _normalize_rtn_kwargs(normalized)
    elif target_cls is GGUFConfig:
        normalized = _normalize_gguf_kwargs(normalized)
    elif target_cls is FP8Config:
        normalized = _normalize_fp8_kwargs(normalized)
    elif target_cls is BitsAndBytesConfig:
        normalized = _normalize_bitsandbytes_kwargs(normalized)
    return _filter_quantize_config_payload_for_target_cls(target_cls, normalized)


class QuantizeConfigMeta(type):
    def __instancecheck__(cls, instance):
        if cls is QuantizeConfig:
            return isinstance(instance, BaseQuantizeConfig)
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if cls is QuantizeConfig:
            try:
                return issubclass(subclass, BaseQuantizeConfig)
            except TypeError:
                return False
        return super().__subclasscheck__(subclass)

    def __call__(cls, *args, **kwargs):
        kwargs = _normalize_quantize_config_constructor_kwargs(kwargs)
        if cls is QuantizeConfig:
            target_cls = _resolve_quantize_config_class(kwargs)
            target_kwargs = _prepare_target_quantize_config_kwargs(target_cls, kwargs)
            return type.__call__(target_cls, *args, **target_kwargs)
        return super().__call__(*args, **kwargs)


def _normalize_quantize_config_constructor_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return kwargs

    normalized = dict(kwargs)
    if "moe_parallel_input_capture" in normalized:
        raise ValueError(
            "QuantizeConfig: `moe_parallel_input_capture` has been removed; "
            "use `moe.execution.parallel_input_capture`."
        )
    for legacy_name in ("_adjacent_model_config", "adjacent_config"):
        if legacy_name in normalized:
            raise ValueError(f"QuantizeConfig: `{legacy_name}` was renamed to `adjacent_model`.")
    if FORMAT_FIELD_COMPAT_MARLIN in normalized:
        raise ValueError(
            "QuantizeConfig: `is_marlin_format` has been removed. Use `format=\"marlin\"` only for legacy checkpoint inspection, "
            "or `format=\"gptq\"` for new GPTQ quantization."
        )
    if METHOD_FIELD_CODE not in normalized and QUANT_METHOD_FIELD in normalized:
        normalized[METHOD_FIELD_CODE] = normalized[QUANT_METHOD_FIELD]
    normalized.pop(QUANT_METHOD_FIELD, None)

    if FORMAT_FIELD_CODE not in normalized and FORMAT_FIELD_CHECKPOINT in normalized:
        normalized[FORMAT_FIELD_CODE] = normalized[FORMAT_FIELD_CHECKPOINT]
    normalized.pop(FORMAT_FIELD_CHECKPOINT, None)
    return normalized


@dataclass
class BaseQuantizeConfig(metaclass=QuantizeConfigMeta):
    bits: Union[int, str, GGUFBits] = field(default=4, metadata={"choices": [2, 3, 4, 5, 6, 7, 8]})

    # allow dynamic bitsize per layer, if None or some layer not set, use bits
    dynamic: Optional[Dict[str, Dict[str, Union[int, str, bool, GGUFBits]]]] = field(default=None)

    # 128 offers a good balance between inference speed, VRAM usage, and quality.
    group_size: int = field(default=128)

    desc_act: Optional[bool] = field(default=None)

    # symmetric quantization toggle (True=symmetric, False=asymmetric).
    sym: bool = field(default=True)

    true_sequential: bool = field(default=True)

    fused_forward: Optional[FusedForwardConfig] = field(
        default=None,
        metadata={
            "help": (
                "Optional configuration for fusing same-input module forward passes "
                "(e.g. q/k/v, gate/up) during calibration. None disables fusion; "
                "FusedForwardConfig() enables it with splice='view'."
            )
        },
    )

    lm_head: bool = field(default=False)

    method: METHOD = field(default=METHOD.GPTQ)

    # Serialized/exported checkpoint layout. This is the authoritative post-quantization format.
    format: FORMAT = field(default=FORMAT.GPTQ)

    # properties that do not directly contribute to quantization or inference should be placed in meta
    meta: Optional[Dict] = field(default=None)

    # normalized to DEVICE after passing to load()
    device: Optional[Union[str, torch.device]] = field(default=None)

    # Maximum host-pinned memory (GB) that LazyTurtle will keep registered across layers.
    # None or <=0 disables the cap and keeps all touched shards pinned (fastest, more RAM).
    lazy_turtle_max_pinned_gb: Optional[float] = field(
        default=4.0,
        metadata={"help": "Maximum host-pinned checkpoint shard memory (GB) for LazyTurtle. None or <=0 = unlimited."},
    )

    # gptq was originally designed to pack quantized weights inside INT32 dtypes
    # allowing using different dtypes used for packing quantized weights
    # affects [`qweights`, `qzeros`]
    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    # packing implementation hint (`original` = legacy CPU pack, `gpu` enables CUDA pack, `cpu` forces block CPU pack).
    pack_impl: str = field(default="cpu")

    # Number of CPU threads to use for packing. None or 0 lets the extension auto-select.
    pack_threads: Optional[int] = field(
        default=None,
        metadata={"help": "Number of CPU threads to use for packing. None or 0 = auto."},
    )

    adapter: Optional[Union[Dict[str, Any], Lora]] = field(default=None)

    # controls cpu memory saving by offloading layers/modules to disk in the slow quantization process
    offload_to_disk: bool = field(
        default=True,
        metadata={"help": "Offload completed module memory to disk during quantization loop"},
    )
    offload_to_disk_path: str = field(
        default=None,
        metadata={"help": "Offload disk path. Only applicable if Offload to disk is enabled"},
    )
    _offload_temp_dir: Optional[_SharedTemporaryDirectory] = field(default=None, init=False, repr=False, compare=False)

    rotation: Optional[str] = field(default=None, metadata={"choices": ["hadamard", "random"]})

    # if calibration is insufficient, fallback to a simple quantization strategy
    fallback: Optional[Fallback] = field(default_factory=Fallback)

    # Callback function to filter devices for compute-intensive stages (quantization and forwarding)
    compute_device_filter: Optional[callable] = field(
        default=None,
        metadata={"help": "Callback function to filter devices for compute-intensive stages. Function signature: fn(devices: List) -> List. "
                  "Example to exclude device 0: compute_device_filter=lambda devices: [d for d in devices if d.index != 0]"}
    )

    # Device for storing calibration data during input capture
    calibration_data_device: Optional[Union[str, torch.device]] = field(
        default=None,
        metadata={"help": "Device for storing calibration data. 'balanced' = round-robin across GPUs, or specify device like 'cuda:1'."}
    )

    auto_forward_data_parallel: bool = field(
        default=True,
        metadata={"help": "When multi-gpu is detected, we may data clone modules to each gpu for data parallelism "
        "to speed up quantization forwarding. This causes extra time spent (especially for MoE layers) and vram pressure, "
        "leading in some cases to slower forwarding or vram OOM"}
    )

    # CPU worker count for weight-only quantization. None keeps the automatic
    # policy: RTN CPU quantization uses multiple workers only when Python's GIL
    # is disabled, or when the runtime environment explicitly overrides it.
    weight_only_quant_threads: Optional[int] = field(
        default=None,
        metadata={"help": "CPU worker count for weight-only quantization. None = auto."},
    )

    # User-facing dense-pool strategy. The dense pool owns the serial path:
    # qkv, z, out_proj, norms, router, shared expert, and dense MLP modules.
    dense_vram_strategy: VramStrategy = field(
        default=VramStrategy.EXCLUSIVE,
        metadata={"help": "Dense pool placement strategy. The dense pool owns qkv, z, out_proj, norms, router, shared expert, and dense MLP modules."},
    )
    # Optional dense-pool device list, relative to CUDA_VISIBLE_DEVICES. In
    # BALANCED mode, model-tree calculation groups stay together, so qkv is not split.
    dense_vram_strategy_devices: Optional[List[Union[str, torch.device]]] = field(
        default=None,
        metadata={"help": "Explicit device pool for dense modules. In dense BALANCED mode, modules are assigned by calculation groups, so qkv stays co-located."},
    )
    # User-facing expert-pool strategy. Expert families are placed as whole
    # units so gate/up/down for one expert stay on the same device.
    moe_vram_strategy: VramStrategy = field(
        default=VramStrategy.EXCLUSIVE,
        metadata={"help": "MoE expert-pool placement strategy. Expert families stay co-located and can be balanced across this pool."},
    )
    # Optional expert-pool device list, relative to CUDA_VISIBLE_DEVICES.
    moe_vram_strategy_devices: Optional[List[Union[str, torch.device]]] = field(
        default=None,
        metadata={"help": "Explicit device pool for MoE expert modules. Each expert family (gate/up/down) stays on one device."},
    )

    # Checkpoint sharding strategy used when saving quantized models.
    # ``per_layer`` writes one safetensors file per transformer layer and a
    # separate file for non-layer tensors (embeddings, norm, lm_head).
    shard_strategy: Optional[ShardStrategy] = field(default=ShardStrategy.PER_LAYER)

    gc_mode: GcMode = field(
        default=GcMode.INTERVAL,
        metadata={"help": "Garbage collection mode: 'interval' for regular GC or 'on_stage_end' for GC after stage end (after forward pass, quantize, layer finilization)."}
    )

    wait_for_submodule_finalizers: bool = field(
        default=False,
        metadata={"help": "Wait for all layer finalization tasks (packing, offloading to disk, etc) to complete before proceeding to next layer. May reduce vram pressure for some env."}
    )

    native_kernel_replay: bool = field(
        default=False,
        metadata={"help": "If True, post-quantization per-layer replay runs through a packed native kernel (e.g. Marlin on CUDA) instead of the dense reconstructed weight. The final checkpoint is still packed independently so the replay module can be discarded."}
    )

    moe: Optional[MoEConfig] = field(
        default=None,
        metadata={"help": "Mixture-of-Experts (MoE) routing and execution configuration. "
                  "Requires import: from gptqmodel.quantization.config import MoEConfig, MoEExecutionConfig, ExpertsRoutingBypass, ExpertsRoutingOverride. "
                  "Example with bypass routing (forward all data to each expert): "
                  "moe=MoEConfig(routing=ExpertsRoutingBypass()) - processes all experts in one batch (default). "
                  "moe=MoEConfig(routing=ExpertsRoutingBypass(), execution=MoEExecutionConfig(batch_size=4)) "
                  "- processes 4 modules at a time to reduce VRAM pressure. "
                  "Example with routing override (limit experts per token): "
                  "moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok=2)). "
                  "Example to forward to all experts: "
                  "moe=MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok='all'))"}
    )

    @property
    def quant_method(self) -> METHOD:
        return self.method

    @quant_method.setter
    def quant_method(self, value: Union[str, METHOD]) -> None:
        self.method = value

    @property
    def checkpoint_format(self):
        return self.format

    @checkpoint_format.setter
    def checkpoint_format(self, value) -> None:
        self.format = value

    @property
    def runtime_bits(self):
        return self.bits

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_format(self.format)
        return self.format

    def _normalize_bits_field(self, bits_value, checkpoint_format: FORMAT):
        return _normalize_quant_bits(bits_value, format_value=checkpoint_format)

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        for key, value in layer_dict.items():
            if key == "bits":
                normalized_bits = self._normalize_bits_field(value, checkpoint_format=checkpoint_format)
                layer_dict[key] = normalized_bits
                if quant_bits_width(normalized_bits) not in valid_bit_widths:
                    raise ValueError(
                        f"QuantizeConfig: Layer `{layer_name}` only support quantization of `{valid_bit_widths}` bits."
                    )
            if key == "group_size" and value != -1 and value <= 0:
                raise ValueError(_resolve_dynamic_group_size_error())

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return tuple(METHOD)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        valid_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.method, None)
        if valid_formats is None:
            raise ValueError(f"QuantizeConfig: Unsupported `method`: {self.method}")
        return tuple(valid_formats)

    def export_quant_method(self) -> METHOD:
        return _resolve_export_quant_method(resolve_quant_format(self.format, self.method), fallback_method=self.method)

    def default_desc_act(self) -> bool:
        return True

    def _bits_in_choices(self, valid_bits: List[Union[int, float]]) -> bool:
        """Return whether the normalized bit field belongs to this format's choices."""

        return quant_bits_width(self.bits) in valid_bits

    def _ensure_offload_temp_dir(self) -> None:
        if self.offload_to_disk and not self.offload_to_disk_path:
            self._offload_temp_dir = _create_temp_offload_dir()
            self.offload_to_disk_path = self._offload_temp_dir.name
            log.info(f"QuantizeConfig: offload_to_disk_path auto set to temporary dir `{self.offload_to_disk_path}`")

    def __post_init__(self):
        fields_info = fields(self)

        self.method = _normalize_quant_method(self.method)
        format_family = self._resolve_checkpoint_format()
        self.pack_dtype = _normalize_pack_dtype(self.pack_dtype)
        self.bits = self._normalize_bits_field(self.bits, checkpoint_format=format_family)

        allowed_methods = self.allowed_quant_methods()
        if allowed_methods and self.method not in allowed_methods:
            raise ValueError(
                f"{self.__class__.__name__}: `method` must be one of {[v.value for v in allowed_methods]}."
            )

        # TODO FIXME awq compat which didn't have checkpoint_format before merging to gptqmodel
        if self.quant_method == METHOD.AWQ and self.format not in [
            FORMAT.MARLIN,
            FORMAT.GEMV,
            FORMAT.GEMV_FAST,
            FORMAT.GEMM,
            FORMAT.BITBLAS,
            FORMAT.LLM_AWQ,
        ]:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.GEMM}`")
            self.format = FORMAT.GEMM
            format_family = self._resolve_checkpoint_format()

        valid_formats = self.supported_export_formats()
        if format_family not in valid_formats:
            raise ValueError(f"{self.__class__.__name__}: unsupported export `format` `{format_family}`.")

        self.fallback = _normalize_fallback(self.fallback)

        valid_bit_widths = fields_info[0].metadata["choices"]
        if not self._bits_in_choices(valid_bit_widths):
            raise ValueError(f"QuantizeConfig: `bits` must be in the set of `{fields_info[0].metadata['choices']}`.")

        # 5/6/7-bit GPTQ only exists in the planar (split-plane) layout, which is
        # a distinct checkpoint format from the continuous gptq/gptq_v2 layouts.
        # Per-layer dynamic bit overrides count too: one 5/6/7-bit layer makes
        # the checkpoint planar.
        planar_only_bits = False
        if self.method == METHOD.GPTQ:
            planar_only_bits = quant_bits_width(self.bits) in (5, 6, 7)
            if not planar_only_bits and self.dynamic is not None:
                planar_only_bits = any(
                    isinstance(layer_dict, dict) and quant_bits_width(layer_dict.get("bits", self.bits)) in (5, 6, 7)
                    for layer, layer_dict in self.dynamic.items()
                    if not layer.startswith("-")
                )
        if format_family in (FORMAT.GPTQ, FORMAT.GPTQ_V2) and planar_only_bits:
            log.info(
                f"QuantizeConfig: 5/6/7-bit layers use the planar layout; auto fix `format` to `{FORMAT.GPTQ_P}`."
            )
            self.format = FORMAT.GPTQ_P
            format_family = self._resolve_checkpoint_format()

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith("-")},
                **{k: v for k, v in self.dynamic.items() if not k.startswith("-")},
            }

            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=valid_bit_widths,
                    checkpoint_format=format_family,
                )

        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError(_resolve_dynamic_group_size_error())

        if self.desc_act is None:
            self.desc_act = self.default_desc_act()
        elif not isinstance(self.desc_act, bool):
            self.desc_act = bool(self.desc_act)

        if self.meta is not None:
            if not isinstance(self.meta, dict):
                raise ValueError("QuantizeConfig: `meta` must be a dictionary")
            for key in self.meta:
                if not isinstance(key, str):
                    raise ValueError("QuantizeConfig: `meta` keys must be strings")
        else:
            self.meta = {}

        self.fused_forward = _normalize_fused_forward_config(self.fused_forward)
        self.adapter = normalize_adapter(self.adapter)

        # Rotation fuses orthogonal transforms into the weights and requires
        # materialized tensors; meta-device/shell loading cannot be used.
        if self.rotation and self.offload_to_disk:
            log.warn(
                f"{self.__class__.__name__}: `rotation` is incompatible with `offload_to_disk`; disabling disk offload."
            )
            self.offload_to_disk = False

        self._ensure_offload_temp_dir()

        self.dense_vram_strategy = _normalize_dense_vram_strategy(self.dense_vram_strategy)
        self.dense_vram_strategy_devices = _normalize_strategy_devices(
            self.dense_vram_strategy_devices,
            field_name="dense_vram_strategy_devices",
        )
        self.moe_vram_strategy = _normalize_moe_vram_strategy(self.moe_vram_strategy)
        self.moe_vram_strategy_devices = _normalize_strategy_devices(
            self.moe_vram_strategy_devices,
            field_name="moe_vram_strategy_devices",
        )
        self.gc_mode = _normalize_gc_mode(self.gc_mode)
        self.moe = _normalize_moe_config(self.moe)
        self.shard_strategy = _normalize_shard_strategy(self.shard_strategy)
        if self.weight_only_quant_threads is not None:
            try:
                self.weight_only_quant_threads = int(self.weight_only_quant_threads)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "QuantizeConfig: `weight_only_quant_threads` must be a positive integer or None."
                ) from exc
            if self.weight_only_quant_threads < 1:
                raise ValueError("QuantizeConfig: `weight_only_quant_threads` must be a positive integer or None.")

        # Normalize calibration_data_device to canonical form if it's a specific device (not "balanced")
        if self.calibration_data_device is not None:
            if isinstance(self.calibration_data_device, str):
                if self.calibration_data_device.lower() == "balanced":
                    self.calibration_data_device = "balanced"
                else:
                    # Import here to avoid circular import
                    from ..utils.looper_helpers import _canonical_device

                    self.calibration_data_device = _canonical_device(torch.device(self.calibration_data_device))
            elif isinstance(self.calibration_data_device, torch.device):
                # Also normalize when passed as torch.device object
                from ..utils.looper_helpers import _canonical_device

                self.calibration_data_device = _canonical_device(self.calibration_data_device)

    def __deepcopy__(self, memo):
        # Share the immutable `dynamic` dict and skip per-instance dynamic lookup
        # caches.  Global caches keyed by `id(dynamic)` handle the resolution, so
        # cloning only needs the fields the clone will actually mutate.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("_dynamic_patterns", "_dynamic_override_cache", "dynamic"):
                setattr(result, k, v)
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def extension_set(self, key: str, value: Any):
        if self.adapter is None:
            self.adapter = {}
        self.adapter[key.lower()] = value

    def extension_get(self, key: str) -> Any:
        return self.adapter.get(key.lower()) if self.adapter else None

    def meta_set(self, key: str, value: Any):
        self.meta[key] = value

    def meta_get(self, key: str) -> Any:
        return self.meta.get(key)

    def dynamic_get(
        self,
        layer_name: str,
        key: str = None,
        default: Union[int, bool, float] = None,
        sub_key: str = None,
    ) -> Union[Dict, int, bool, float]:
        if self.dynamic is None:
            return default

        overrides = _resolve_dynamic_override(self.dynamic, layer_name)
        if overrides is False:
            return False
        if overrides is None:
            return default

        if key is None:
            return overrides

        if key in overrides:
            sub_value = overrides[key]
        elif key in DYNAMIC_FIELD_SYNONYMS:
            sub_value = None
            for legacy_key in DYNAMIC_FIELD_SYNONYMS[key]:
                if legacy_key in overrides:
                    sub_value = overrides[legacy_key]
                    break
        else:
            return default

        if sub_key:
            if isinstance(sub_value, dict):
                return sub_value.get(sub_key, default)
            log.info(
                "QuantConfig: Dynamic `sub_key`: `%s` failed extraction from `sub_value`: `%s`",
                sub_key,
                sub_value,
            )
            return default

        return sub_value

    def _invalidate_dynamic_cache(self) -> None:
        """Clear the dynamic-resolution cache keyed by the current `dynamic` dict.

        Call this before replacing `self.dynamic` with a new dict, so a recycled
        object id does not return stale compiled patterns or override lookups.
        """
        if self.dynamic is None:
            return
        global _DYNAMIC_CACHE_OVERRIDE_COUNT
        cache_key = id(self.dynamic)
        with _DYNAMIC_CACHE_LOCK:
            cached = _DYNAMIC_CACHE.get(cache_key)
            if cached is not None and cached.dynamic is self.dynamic:
                _DYNAMIC_CACHE.pop(cache_key)
                _DYNAMIC_CACHE_OVERRIDE_COUNT -= len(cached.override_cache)

    def meta_set_versionable(self, key: str, value: List[str]):
        self.meta_set(key, value)

    def meta_get_versionable(self, key: str) -> List[Tuple[str, str]]:
        values = self.meta_get(key)
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        result = []
        for val in values:
            parts = val.split(":")
            if len(parts) >= 2:
                result.append((parts[0].lower(), parts[1].lower()))
        return result

    def is_quantized_by_gptaq(self) -> bool:
        result = self.meta_get_versionable(META_FIELD_QUANTIZER)
        if len(result) > 0:
            for producer, _version in result:
                if producer == META_QUANTIZER_GPTQMODEL:
                    return version.parse(_version) >= version.parse(MIN_VERSION_WITH_V2)
        return False

    def is_quantized_by_foem(self) -> bool:
        result = self.meta_get_versionable(META_FIELD_QUANTIZER)
        if len(result) > 0:
            for producer, _version in result:
                if producer == META_QUANTIZER_GPTQMODEL:
                    return version.parse(_version) >= version.parse(MIN_VERSION_WITH_V2)
        return False

    def extract_adapter_rank_patterns(self) -> Optional[Dict[str, int]]:
        adapter_rank_patterns = {}
        if not self.dynamic or not self.adapter:
            return adapter_rank_patterns

        for k, v in self.dynamic.items():
            if not isinstance(v, dict):
                # Skip negative/boolean dynamic entries (e.g. layer-scope exclusions).
                continue
            adapter_override = v.get("adapter", None)
            if adapter_override and isinstance(adapter_override, Dict):
                rank = adapter_override.get("rank", None)
                if rank and isinstance(rank, int):
                    adapter_rank_patterns[k.lstrip("+:")] = rank

        return adapter_rank_patterns

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, QUANT_CONFIG_FILENAME), "w", encoding="utf-8") as f:
            payload = self.to_dict()
            json_str = json.dumps(payload, indent=2)
            log.info(f"Saved Quantize Config: \n{json_str}")
            f.write(json_str)

    @classmethod
    def from_quant_config(cls, quantize_cfg, format: str = None):
        if "activation" in quantize_cfg and "activation_quantization" in quantize_cfg:
            raise ValueError(
                "QuantizeConfig: cannot contain both `activation` and legacy `activation_quantization`."
            )
        valid_formats = set(FORMAT)
        format_auto_inferred = False
        checkpoint_format_hint = quantize_cfg.get(FORMAT_FIELD_CHECKPOINT) if isinstance(quantize_cfg, dict) else None
        serialized_format = quantize_cfg.get(FORMAT_FIELD_CODE) if isinstance(quantize_cfg, dict) else None
        if format:
            if _looks_like_fp8_fmt(format):
                format = _normalize_fp8_fmt(format)
            elif _looks_like_bitsandbytes_format(format):
                format = _normalize_bitsandbytes_format(format)
            else:
                format = _normalize_format(format)
                if format not in valid_formats:
                    raise ValueError(f"QuantizeConfig: Unknown quantization checkpoint format: {format}.")
            if checkpoint_format_hint is not None or serialized_format is not None:
                raise ValueError(
                    "QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config."
                )
        elif checkpoint_format_hint is None and serialized_format is None:
            format_auto_inferred = True

        field_names = _known_quantize_config_field_names()

        normalized = {
            METHOD_FIELD_CODE: METHOD.GPTQ,
            FORMAT_FIELD_CODE: format if format else FORMAT.GPTQ,
        }
        format_field_present = format is not None
        legacy_checkpoint_format = None

        for key, val in quantize_cfg.items():
            key = key.lower()

            if key == FORMAT_FIELD_COMPAT_MARLIN:
                raise ValueError(
                    "QuantizeConfig: `is_marlin_format` is no longer supported. Replace it with an explicit `format` field."
                )

            if key == FORMAT_FIELD_CHECKPOINT:
                if _looks_like_fp8_fmt(val):
                    legacy_checkpoint_format = _normalize_fp8_fmt(val)
                elif _looks_like_bitsandbytes_format(val):
                    legacy_checkpoint_format = _normalize_bitsandbytes_format(val)
                else:
                    try:
                        legacy_checkpoint_format = _normalize_gguf_public_format(val)
                    except ValueError:
                        legacy_checkpoint_format = None
                    if legacy_checkpoint_format is None:
                        legacy_checkpoint_format = _normalize_format(val)
                if legacy_checkpoint_format is not None:
                    checkpoint_format_hint = legacy_checkpoint_format
                continue

            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]
            elif key in QUANT_CONFIG_ARG_SYNONYMS_NEGATED and QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS_NEGATED[key]
                val = not bool(val)

            if key == METHOD_FIELD_CODE:
                if isinstance(val, str) and val.lower() == FORMAT.MARLIN:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.MARLIN
                elif isinstance(val, str) and val.lower() == FORMAT.BITBLAS:
                    normalized[FORMAT_FIELD_CODE] = FORMAT.BITBLAS
                else:
                    normalized[METHOD_FIELD_CODE] = _normalize_quant_method(val)
            elif key == FORMAT_FIELD_CODE:
                format_field_present = True
                serialized_format_hint = None
                try:
                    serialized_format_hint = resolve_quant_format(
                        val,
                        normalized.get(METHOD_FIELD_CODE),
                    )
                except ValueError:
                    serialized_format_hint = None

                format_hint = format or legacy_checkpoint_format or checkpoint_format_hint
                if format_hint is not None:
                    try:
                        format_hint = resolve_quant_format(
                            format_hint,
                            normalized.get(METHOD_FIELD_CODE),
                        )
                    except ValueError:
                        format_hint = None
                if serialized_format_hint in {
                    FORMAT.GGUF,
                    FORMAT.FP8,
                    FORMAT.BITSANDBYTES,
                } or format_hint in {
                    FORMAT.GGUF,
                    FORMAT.FP8,
                    FORMAT.BITSANDBYTES,
                }:
                    normalized[key] = val
                else:
                    normalized[key] = _normalize_format(val)
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        if not format_field_present and legacy_checkpoint_format is not None:
            normalized[FORMAT_FIELD_CODE] = legacy_checkpoint_format

        if quantize_cfg.get(AWQ_PACKING_BACKEND_FIELD) == "llm-awq":
            normalized[METHOD_FIELD_CODE] = METHOD.AWQ
            normalized[FORMAT_FIELD_CODE] = FORMAT.LLM_AWQ
            normalized[PACK_DTYPE_FIELD] = torch.int16
            log.info("Detected llm-awq quantization format; FORMAT automatically set to FORMAT.LLM_AWQ.")

        meta_payload = normalized.get(META_FIELD)
        meta_field_map = {
            "fallback": "fallback",
            "hessian": "hessian",
            "gptaq": "gptaq",
            "foem": "foem",
            "weight_only": "weight_only",
            "preprocessors": "preprocessors",
            "gc_mode": "gc_mode",
            "wait_for_submodule_finalizers": "wait_for_submodule_finalizers",
            "auto_forward_data_parallel": "auto_forward_data_parallel",
            "weight_only_quant_threads": "weight_only_quant_threads",
            "dense_vram_strategy": "dense_vram_strategy",
            "dense_vram_strategy_devices": "dense_vram_strategy_devices",
            "moe_vram_strategy": "moe_vram_strategy",
            "moe_vram_strategy_devices": "moe_vram_strategy_devices",
            "moe": "moe",
            "offload_to_disk": "offload_to_disk",
            "offload_to_disk_path": "offload_to_disk_path",
            "pack_impl": "pack_impl",
            "shard_strategy": "shard_strategy",
            "mse": "mse",
            "scale_search": "scale_search",
            "mock_quantization": "mock_quantization",
            "act_group_aware": "act_group_aware",
            "adjacent_model": "adjacent_model",
            "true_sequential": "true_sequential",
            "damp_percent": "damp_percent",
            "damp_auto_increment": "damp_auto_increment",
            "opt_rotation_epochs": "opt_rotation_epochs",
            "opt_finetune_epochs": "opt_finetune_epochs",
            "opt_train_samples": "opt_train_samples",
            "opt_validation_samples": "opt_validation_samples",
            "opt_batch_size": "opt_batch_size",
            "opt_rotation_lr": "opt_rotation_lr",
            "opt_weight_lr": "opt_weight_lr",
            "opt_quantizer_lr": "opt_quantizer_lr",
            "opt_pair_ratio": "opt_pair_ratio",
            "opt_seed": "opt_seed",
            "opt_optimizer": "opt_optimizer",
            "opt_weight_decay": "opt_weight_decay",
            "opt_betas": "opt_betas",
            "opt_eps": "opt_eps",
            "opt_amsgrad": "opt_amsgrad",
            "opt_sgd_momentum": "opt_sgd_momentum",
            "opt_sgd_dampening": "opt_sgd_dampening",
            "opt_sgd_nesterov": "opt_sgd_nesterov",
            "opt_fused_rotation": "opt_fused_rotation",
            "opt_gradient_checkpointing": "opt_gradient_checkpointing",
            "opt_stage_cudagraph": "opt_stage_cudagraph",
            "opt_best_state_dtype": "opt_best_state_dtype",
            "opt_train_on_noisy_inputs": "opt_train_on_noisy_inputs",
            "opt_scope": "opt_scope",
            "opt_stage_impl": "opt_stage_impl",
            "opt_pair_impl": "opt_pair_impl",
            "opt_quantizer_impl": "opt_quantizer_impl",
            "opt_channel_scale_clamp_min": "opt_channel_scale_clamp_min",
            "opt_channel_scale_clamp_max": "opt_channel_scale_clamp_max",
            "scale_search_chunked_activations": "scale_search_chunked_activations",
            "scale_search_gpu_weight_restore": "scale_search_gpu_weight_restore",
            "scale_search_refine_steps": "scale_search_refine_steps",
            "enable_shared_hessian_cache": "enable_shared_hessian_cache",
            "enable_activation_x_mean_cache": "enable_activation_x_mean_cache",
            "quantization_diagnostics": "quantization_diagnostics",
            "fused_forward": "fused_forward",
            "native_kernel_replay": "native_kernel_replay",
            "adaptive_damping": "adaptive_damping",
            "adaptive_clipping": "adaptive_clipping",
        }
        if isinstance(meta_payload, dict):
            for normalized_key, meta_key in meta_field_map.items():
                if normalized_key not in normalized and meta_key in meta_payload:
                    normalized[normalized_key] = meta_payload.get(meta_key)

        target_cls = (
            cls if cls not in {BaseQuantizeConfig, QuantizeConfig} else _resolve_quantize_config_class(normalized)
        )
        target_field_names = {config_field.name for config_field in fields(target_cls) if config_field.init}
        if normalized.get("adjacent_model") is not None and "adjacent_model" not in target_field_names:
            raise ValueError("QuantizeConfig: `adjacent_model` currently supports GPTQ and AWQ quantization only.")
        normalized = _normalize_quantize_config_payload_for_target_cls(target_cls, normalized)
        if target_cls is RTNConfig:
            normalized = _normalize_rtn_kwargs(normalized)
        elif target_cls is GGUFConfig:
            normalized = _normalize_gguf_kwargs(normalized)
        elif target_cls is FP8Config:
            normalized = _normalize_fp8_kwargs(normalized)
        elif target_cls is BitsAndBytesConfig:
            normalized = _normalize_bitsandbytes_kwargs(normalized)

        if format_auto_inferred:
            log.info(
                f"QuantizeConfig: `{FORMAT_FIELD_CODE}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}"
            )

        resolved_format_family = resolve_quant_format(
            normalized[FORMAT_FIELD_CODE],
            normalized.get(METHOD_FIELD_CODE),
        )
        if resolved_format_family in {FORMAT.BITBLAS, FORMAT.BITSANDBYTES}:
            normalized["desc_act"] = False

        if "sym" not in normalized and target_cls not in {
            GGUFConfig,
            FP8Config,
            BitsAndBytesConfig,
            EXL3Config,
            QVQConfig,
        }:
            log.warn(
                "QuantizeConfig: config does not contain `sym` (symmetric quantization). This may result in silent errors. Defaulting to `sym=True`."
            )
        return target_cls(**_filter_quantize_config_payload_for_target_cls(target_cls, normalized))

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        format = kwargs.pop("format", None)

        transformers_config = False
        resolved_config_file = None
        for quantize_config_filename in QUANT_CONFIG_FILENAME_COMPAT:
            resolved_config_file = join(save_dir, quantize_config_filename)
            if os.path.exists(resolved_config_file):
                if quantize_config_filename == "config.json":
                    transformers_config = True
                break

        if resolved_config_file is None:
            raise ValueError(
                "QuantizeConfig: No quantize_config.json, quant_config.json or config.json file was found in the model repository."
            )

        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)
            if transformers_config:
                args_from_json = args_from_json["quantization_config"]
            return cls.from_quant_config(args_from_json, format)

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        if self.fused_forward is None:
            meta_payload["fused_forward"] = None
        else:
            meta_payload["fused_forward"] = self.fused_forward.to_dict()

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        return None

    def to_dict(self):
        smooth = _serialize_smooth_method(self.fallback.smooth if self.fallback is not None else None)

        # Serialization normalizes nested values (for example scale_dtype) in
        # place below, so it must operate on an independent payload.  Keeping
        # aliases here would silently rewrite the live config and invalidate
        # identity-based dynamic-resolution cache entries.
        meta_payload = copy.deepcopy(self.meta) if self.meta else {}
        if self.moe:
            meta_payload["moe"] = self.moe.to_dict()

        if self.fallback is None:
            meta_payload["fallback"] = None
        else:
            meta_payload["fallback"] = {
                "strategy": (
                    self.fallback.strategy.value
                    if isinstance(self.fallback.strategy, FallbackStrategy)
                    else self.fallback.strategy
                ),
                "threshold": self.fallback.threshold,
                "smooth": smooth,
            }

        meta_payload["offload_to_disk"] = self.offload_to_disk
        meta_payload["offload_to_disk_path"] = self.offload_to_disk_path
        meta_payload["pack_impl"] = self.pack_impl
        meta_payload["gc_mode"] = self.gc_mode.value if isinstance(self.gc_mode, GcMode) else self.gc_mode
        meta_payload["shard_strategy"] = self.shard_strategy.value if self.shard_strategy is not None else None
        meta_payload["wait_for_submodule_finalizers"] = self.wait_for_submodule_finalizers
        meta_payload["auto_forward_data_parallel"] = self.auto_forward_data_parallel
        meta_payload["weight_only_quant_threads"] = self.weight_only_quant_threads
        meta_payload["dense_vram_strategy"] = (
            self.dense_vram_strategy.value
            if isinstance(self.dense_vram_strategy, VramStrategy)
            else self.dense_vram_strategy
        )
        meta_payload["dense_vram_strategy_devices"] = self.dense_vram_strategy_devices
        meta_payload["moe_vram_strategy"] = (
            self.moe_vram_strategy.value
            if isinstance(self.moe_vram_strategy, VramStrategy)
            else self.moe_vram_strategy
        )
        meta_payload["moe_vram_strategy_devices"] = self.moe_vram_strategy_devices
        self._update_meta_payload(meta_payload)
        meta_payload["native_kernel_replay"] = self.native_kernel_replay

        dynamic_payload = None
        if self.dynamic is not None:
            dynamic_payload = {}
            for pattern, layer_config in self.dynamic.items():
                if isinstance(layer_config, dict):
                    # Per-layer adapter overrides are runtime-only. Exclude
                    # them before cloning so serialization does not copy a
                    # potentially heavyweight adapter payload merely to drop it.
                    serializable_layer_config = {
                        key: value for key, value in layer_config.items() if key != "adapter"
                    }
                    dynamic_payload[pattern] = copy.deepcopy(serializable_layer_config)
                else:
                    dynamic_payload[pattern] = copy.deepcopy(layer_config)

        out = {
            "bits": serialize_quant_bits(self.bits),
            "dynamic": dynamic_payload,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "lm_head": self.lm_head,
            METHOD_FIELD_CODE: self.method,
            QUANT_METHOD_FIELD: self.method,
            FORMAT_FIELD_CODE: self.format,
            FORMAT_FIELD_CHECKPOINT: self.format,
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            "rotation": self.rotation,
            META_FIELD: meta_payload,
        }
        self._update_output_payload(out)

        dynamic = out["dynamic"]
        if dynamic:
            for _, v in dynamic.items():
                if not isinstance(v, dict):
                    continue
                if "bits" in v:
                    v["bits"] = serialize_quant_bits(v["bits"])

        out = {k: v for k, v in out.items() if v is not None and (v not in [None, {}])}
        dict_scale_dtype_to_str(out)
        return out

    def calculate_bits_per_weight(self):
        bit_width = quant_bits_width(self.bits)
        if self.group_size != -1:
            per_group_bits = self.group_size * bit_width
            per_group_bits += 16
            per_group_bits += bit_width
            per_group_bits += 4
            bpw = per_group_bits / self.group_size
            bpw += 0.1
        else:
            bpw = bit_width
        log.info(f"Estimated Quantization BPW (bits per weight): {bpw} bpw, based on [bits: {self.bits}, group_size: {self.group_size}]")

    def moe_routing_override(self, num_experts: int) -> Union[int, None]:
        if self.moe is None:
            return None
        return self.moe.routing_override(num_experts)

    def moe_routing_bypass(self) -> bool:
        if self.moe is None:
            return False
        return self.moe.routing_bypass()

    def uses_weight_only_lifecycle(self) -> bool:
        return False

    def requires_calibration_dataset(self) -> bool:
        return not self.uses_weight_only_lifecycle()

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {}


@dataclass
class PreProcessorConfig(BaseQuantizeConfig):
    preprocessors: Optional[List[Union[BasePreProcessorConfig, Dict[str, Any], str]]] = field(default_factory=list)
    smoother: Optional[Union[SmootherConfig, SmoothMethod, Dict[str, Any], str]] = field(default=None)
    # Backward-compatible alias. New code should use `smoother`.
    smooth: Optional[Union[SmoothMethod, Dict[str, Any], str]] = field(default=None, repr=False)

    def _normalize_preprocessor_state(self) -> None:
        self.preprocessors = _normalize_preprocessors(self.preprocessors)

        smoother_payload = self.smoother if self.smoother is not None else self.smooth
        self.smoother = _normalize_smoother_config(smoother_payload)

        if self.smoother is None:
            for preprocessor in self.preprocessors:
                if isinstance(preprocessor, SmootherConfig):
                    self.smoother = preprocessor
                    break

        non_smoother_preprocessors = [
            preprocessor for preprocessor in self.preprocessors if not isinstance(preprocessor, SmootherConfig)
        ]
        if self.smoother is not None:
            non_smoother_preprocessors.append(self.smoother)
        self.preprocessors = non_smoother_preprocessors
        _validate_unique_preprocessors(self.preprocessors)
        self.smooth = self.resolve_smooth_method()

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

    def resolve_smooth_method(self) -> Optional[SmoothMethod]:
        if self.smoother is None:
            return None
        return self.smoother.smooth

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        if self.preprocessors:
            meta_payload["preprocessors"] = [preprocessor.to_dict() for preprocessor in self.preprocessors]


@dataclass
class QuantizeConfig(BaseQuantizeConfig, metaclass=QuantizeConfigMeta):
    """Backward-compatible quantization config factory.

    Direct construction dispatches to a concrete method-specific config class.
    """


@dataclass
class GPTQConfig(PreProcessorConfig):
    damp_percent: Optional[float] = field(default=None)
    damp_auto_increment: Optional[float] = field(default=None)
    act_group_aware: Optional[bool] = field(default=None)
    static_groups: bool = field(default=False)
    mse: float = field(default=0.0)
    scale_search: Optional[ScaleSearchConfig] = field(
        default=_UNSET_SCALE_SEARCH,  # type: ignore[arg-type]
        metadata={
            "help": (
                "Scale-search objective. Defaults to activation; pass None to disable. "
                "Choices: mse, activation-diagonal, group-local Hessian, hybrid shrinkage error, "
                "marlin (Hessian through packed Marlin kernel), marlin_mse, or marlin_activation."
            )
        },
    )
    scale_search_candidate_chunk_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional exact scale-search candidate chunk size. Larger values reduce launch overhead at the "
                "cost of temporary accelerator memory; None uses the bounded automatic policy."
            )
        },
    )
    gptaq: Optional[GPTAQConfig] = field(default=None)
    foem: Optional[FOEMConfig] = field(default=None)
    mock_quantization: bool = field(
        default=False,
        metadata={"help": "Skip heavy computations for fast model loading validation"},
    )
    hessian: Optional[HessianConfig] = field(default_factory=HessianConfig)
    adaptive_damping: Optional[Union[DampConfig, AdaptiveDampingConfig, Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help": (
                "Hessian damping configuration. Defaults to static damping; pass an enabled "
                "AdaptiveDampingConfig to opt in to adaptive behavior."
            )
        },
    )
    adaptive_clipping: Optional[Union[AdaptiveClippingConfig, Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help": (
                "Optional adaptive weight-clipping search. Omit it to preserve the configured mse/scale_search "
                "path; opting in defaults to the calibration-aware GPTQ correction objective."
            )
        },
    )
    # Experimental quantization-time AdjacentExact policy. Checkpoints retain
    # the policy for reproducibility and optional future requantization.
    adjacent_model: Optional[Any] = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"help": "Optional AdjacentModelConfig used during GPTQ quantization; disabled by default."},
    )
    enable_shared_hessian_cache: bool = field(
        default=True,
        metadata={
            "help": "Share same-input GPTQ Hessian accumulation and inverse/Cholesky cache within one processor subset."
        },
    )
    quantization_diagnostics: QuantizationDiagnosticsMode = field(
        default=QuantizationDiagnosticsMode.AUTO,
        metadata={
            "choices": [mode.value for mode in QuantizationDiagnosticsMode],
            "help": (
                "Quantization anomaly diagnostics: off disables them, auto adds a cheap module-loss summary, "
                "and channel additionally measures W-to-Wq reconstruction error, scale/group anomalies, and sampled "
                "pre-pack/post-pack logical-code parity. Saved models include JSON and Markdown reports."
            ),
        },
    )

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.GPTQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return GPTQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        desc_act_user_value = self.desc_act
        act_group_aware_user_value = self.act_group_aware
        adaptive_damping_user_value = self.adaptive_damping
        super().__post_init__()

        # Preserve the user's explicit choice so quantization-time safeguards can
        # distinguish "defaulted to True" from "explicitly requested True".
        self._act_group_aware_user_value = act_group_aware_user_value
        self._adaptive_damping_user_value = adaptive_damping_user_value

        self.adjacent_model = _normalize_adjacent_model(self.adjacent_model)
        if self.scale_search_candidate_chunk_size is not None:
            if isinstance(self.scale_search_candidate_chunk_size, bool) or self.scale_search_candidate_chunk_size < 1:
                raise ValueError("QuantizeConfig: `scale_search_candidate_chunk_size` must be a positive integer or None.")
            self.scale_search_candidate_chunk_size = int(self.scale_search_candidate_chunk_size)
        if self.adjacent_model is not None and self.method != METHOD.GPTQ:
            raise ValueError(
                "QuantizeConfig: `adjacent_model` currently supports `method=\"gptq\"` only."
            )
        raw_damp_percent = self.damp_percent
        raw_damp_auto_increment = self.damp_auto_increment
        if self.damp_percent is None:
            self.damp_percent = _default_damp_percent(self.method)
        if self.damp_auto_increment is None:
            self.damp_auto_increment = _default_damp_auto_increment(self.method)
        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")
        if (
            isinstance(self.damp_auto_increment, bool)
            or not isinstance(self.damp_auto_increment, (int, float))
            or not math.isfinite(self.damp_auto_increment)
            or not (0 <= self.damp_auto_increment < 1)
        ):
            raise ValueError("QuantizeConfig: `damp_auto_increment` must be a finite number in [0, 1).")

        # Store the original fixed damping values before they are overwritten by the
        # adaptive damp config. These are the empirical floor used when adaptive
        # damping is enabled and the static value when it is disabled.
        self._damp_percent_user_value = self.damp_percent
        self._damp_auto_increment_user_value = self.damp_auto_increment

        self.hessian = _normalize_hessian(self.hessian)
        self.adaptive_damping = _normalize_adaptive_damping(self.adaptive_damping)
        self.adaptive_clipping = _normalize_adaptive_clipping(self.adaptive_clipping)

        # Synchronize the adaptive baseline and step with the legacy scalar values
        # when the user did not explicitly override them in `adaptive_damping`.
        if isinstance(self.adaptive_damping, AdaptiveDampingConfig):
            user_adaptive = self._adaptive_damping_user_value
            user_set_base = isinstance(user_adaptive, dict) and "base_percdamp" in user_adaptive
            user_set_base = user_set_base or isinstance(user_adaptive, AdaptiveDampingConfig)
            if not user_set_base:
                self.adaptive_damping.base_percdamp = (
                    raw_damp_percent if raw_damp_percent is not None else self._damp_percent_user_value
                )

            user_set_step = isinstance(user_adaptive, dict) and "step" in user_adaptive
            user_set_step = user_set_step or isinstance(user_adaptive, AdaptiveDampingConfig)
            if not user_set_step:
                self.adaptive_damping.step = (
                    raw_damp_auto_increment
                    if raw_damp_auto_increment is not None
                    else self._damp_auto_increment_user_value
                )
            if not self.adaptive_damping.enabled and raw_damp_percent is None:
                # A disabled adaptive config without a separate legacy damp_percent
                # field (including older serialized configs) has only base_percdamp
                # available for the fixed damping path.
                self._damp_percent_user_value = self.adaptive_damping.base_percdamp
        else:
            # Legacy scalar options remain authoritative for the default static
            # damping path. Keep the normalized config synchronized so callers
            # that still pass these fields retain their exact prior behavior.
            if raw_damp_percent is not None:
                self.adaptive_damping.min = raw_damp_percent
                self.adaptive_damping.max = raw_damp_percent
            if raw_damp_auto_increment is not None:
                self.adaptive_damping.step = raw_damp_auto_increment

        # Keep legacy scalar fields in sync with the canonical damp config.
        if isinstance(self.adaptive_damping, AdaptiveDampingConfig) and self.adaptive_damping.enabled:
            self.damp_percent = self.adaptive_damping.base_percdamp
            self.damp_auto_increment = self.adaptive_damping.step
        else:
            self.damp_percent = self.damp.min
            self.damp_auto_increment = self.damp.step
        self.gptaq = _normalize_gptaq(self.gptaq)
        self.foem = _normalize_foem(self.foem)
        if (
            isinstance(self.adaptive_clipping, AdaptiveClippingConfig)
            and self.adaptive_clipping.enabled
            and self.adaptive_clipping.metric == AdaptiveClippingMetric.GPTQ_ERROR.value
        ):
            if self.static_groups:
                raise ValueError(
                    "QuantizeConfig: `adaptive_clipping.metric='gptq_error'` is incompatible with "
                    "`static_groups=True` because static scales are selected before the sequential GPTQ "
                    "inverse-Hessian correction is available. Use `hessian_diag`/`mse`, or disable static groups."
                )
            if self.gptaq is not None or self.foem is not None:
                raise ValueError(
                    "QuantizeConfig: `adaptive_clipping.metric='gptq_error'` currently supports canonical GPTQ only; "
                    "GPTAQ/FOEM use different correction recursions. Select `hessian_diag`/`mse` explicitly."
                )
        self.quantization_diagnostics = normalize_quantization_diagnostics_mode(
            self.quantization_diagnostics
        )
        self._normalize_scale_search()

        if act_group_aware_user_value is None:
            self.act_group_aware = self.method == METHOD.GPTQ
        elif not isinstance(act_group_aware_user_value, bool):
            self.act_group_aware = bool(act_group_aware_user_value)

        self._resolve_activation_ordering(desc_act_user_value, act_group_aware_user_value)
        if self.act_group_aware and self.desc_act:
            raise ValueError("QuantizeConfig:: `act_group_aware` == `True` requires `desc_act` == `False`.")

    @property
    def damp(self) -> Union[DampConfig, AdaptiveDampingConfig]:
        """Return the resolved damping configuration.

        When an enabled :class:`AdaptiveDampingConfig` is configured it is
        returned directly. When a disabled :class:`AdaptiveDampingConfig` or no
        explicit config is set, a static :class:`DampConfig` is derived from the
        legacy ``damp_percent``/``damp_auto_increment`` fields.
        """
        cfg = self.adaptive_damping
        if isinstance(cfg, AdaptiveDampingConfig):
            if cfg.enabled:
                return cfg
            # Disabled adaptive -> static damping at the user-configured fixed percdamp.
            fixed_damp = getattr(self, "_damp_percent_user_value", self.damp_percent)
            return DampConfig(min=fixed_damp, max=fixed_damp, step=cfg.step)
        if isinstance(cfg, DampConfig):
            return cfg
        fixed_damp = getattr(self, "_damp_percent_user_value", self.damp_percent)
        return DampConfig(min=fixed_damp, max=fixed_damp, step=self.damp_auto_increment)

    def _act_group_aware_is_default(self) -> bool:
        """Return True when `act_group_aware` was not explicitly set by the user."""
        return getattr(self, "_act_group_aware_user_value", None) is None

    def _should_disable_act_group_aware_for_small_groups(self) -> bool:
        """Return True when GAR should be disabled because the group size is small."""
        return (
            self._act_group_aware_is_default()
            and self.act_group_aware
            and self.group_size is not None
            and 0 < self.group_size <= 32
        )

    def _normalize_act_group_aware_for_small_groups(self) -> bool:
        """Disable GAR for small positive group sizes when not explicitly requested.

        Returns True if the value was changed.
        """
        if self._should_disable_act_group_aware_for_small_groups():
            self.act_group_aware = False
            return True
        return False

    def _normalize_scale_search(self) -> None:
        """Resolve the new strategy selector and the legacy MSE exponent together."""

        selector_was_omitted = self.scale_search is _UNSET_SCALE_SEARCH
        try:
            self.mse = float(self.mse or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("QuantizeConfig: `mse` must be a non-negative number.") from exc
        if self.mse < 0:
            raise ValueError("QuantizeConfig: `mse` must be a non-negative number.")

        if selector_was_omitted:
            # Preserve the legacy ``mse=<positive>`` API while making an
            # otherwise unconfigured GPTQ quantization use activation search.
            self.scale_search = ScaleSearchConfig.MSE if self.mse > 0 else GPTQ_DEFAULT_SCALE_SEARCH

        self.scale_search = normalize_scale_search(self.scale_search)
        if self.scale_search is None:
            # Backward compatibility: historically any positive `mse` enabled
            # the uniform weight-error range search.
            if self.mse > 0:
                self.scale_search = ScaleSearchConfig.MSE
            return

        if self.mse == 0:
            self.mse = 2.0

        if self.scale_search in {
            ScaleSearchConfig.ACTIVATION,
            ScaleSearchConfig.HESSIAN,
            ScaleSearchConfig.HYBRID,
            ScaleSearchConfig.MARLIN,
            ScaleSearchConfig.MARLIN_MSE,
            ScaleSearchConfig.MARLIN_ACTIVATION,
        } and self.mse != 2.0:
            raise ValueError(
                "QuantizeConfig: activation, hessian, hybrid, marlin, marlin_mse, and marlin_activation scale search require `mse=2.0`."
            )

    def scale_search_cli_summary(self) -> str:
        """Render the resolved global objective and scale-search-only dynamic overrides."""

        global_method = self.scale_search.value if self.scale_search is not None else "disabled"
        overrides = []
        for pattern, layer_config in (self.dynamic or {}).items():
            if not isinstance(layer_config, dict):
                continue
            if "scale_search" in layer_config:
                method = normalize_scale_search(layer_config["scale_search"])
                method_label = method.value if method is not None else "disabled"
            elif "mse" in layer_config:
                try:
                    method_label = "mse" if float(layer_config["mse"] or 0.0) > 0 else "disabled"
                except (TypeError, ValueError):
                    method_label = f"mse({layer_config['mse']!r})"
            else:
                continue
            overrides.append(f"{pattern} -> {method_label}")

        override_summary = "none" if not overrides else f"{len(overrides)} [{'; '.join(overrides)}]"
        return f"global={global_method}; dynamic_overrides={override_summary}"

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        super()._normalize_dynamic_layer_config(
            layer_name,
            layer_dict,
            valid_bit_widths=valid_bit_widths,
            checkpoint_format=checkpoint_format,
        )
        if "scale_search" in layer_dict:
            layer_dict["scale_search"] = normalize_scale_search(layer_dict["scale_search"])

    def _resolve_activation_ordering(
        self,
        desc_act_user_value: Optional[bool],
        act_group_aware_user_value: Optional[bool],
    ) -> None:
        desc_act_enabled_by_user = bool(desc_act_user_value) if desc_act_user_value is not None else False
        act_group_aware_enabled_by_user = (
            bool(act_group_aware_user_value) if act_group_aware_user_value is not None else False
        )

        if desc_act_enabled_by_user and act_group_aware_user_value is not None and act_group_aware_enabled_by_user:
            raise ValueError(
                "QuantizeConfig:: `act_group_aware` == `True` requires `desc_act` == `False` when both are explicitly set."
            )

        if desc_act_enabled_by_user and act_group_aware_user_value is None and self.act_group_aware:
            log.warn(
                "QuantizeConfig: `desc_act=True` automatically disables `act_group_aware`. "
                "Set `act_group_aware=False` explicitly to silence this warning."
            )
            self.act_group_aware = False

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        super()._update_meta_payload(meta_payload)

        if self.gptaq is None:
            meta_payload["gptaq"] = None
        elif self.foem is None:
            device = self.gptaq.device
            meta_payload["gptaq"] = {
                "alpha": self.gptaq.alpha,
                "device": device if isinstance(device, str) else str(device),
            }
        else:
            device = self.foem.device
            meta_payload["foem"] = {
                "alpha": self.foem.alpha,
                "beta": self.foem.beta,
                "device": device if isinstance(device, str) else str(device),
            }

        meta_payload["mse"] = self.mse
        meta_payload["scale_search"] = self.scale_search.value if self.scale_search is not None else None
        meta_payload["mock_quantization"] = self.mock_quantization
        meta_payload["act_group_aware"] = self.act_group_aware
        meta_payload["damp_percent"] = self.damp_percent
        meta_payload["damp_auto_increment"] = self.damp_auto_increment
        if self.adjacent_model is None:
            meta_payload.pop("adjacent_model", None)
        else:
            meta_payload["adjacent_model"] = _serialize_adjacent_model(self.adjacent_model)
        meta_payload["enable_shared_hessian_cache"] = self.enable_shared_hessian_cache
        meta_payload["quantization_diagnostics"] = self.quantization_diagnostics.value
        meta_payload["hessian"] = self.hessian.to_dict()
        if self.adaptive_damping is not None:
            meta_payload["adaptive_damping"] = self.adaptive_damping.to_dict()
        if self.adaptive_clipping is not None:
            meta_payload["adaptive_clipping"] = self.adaptive_clipping.to_dict()

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["sym"] = self.sym
        out[FORMAT_FIELD_CODE] = self.format


@dataclass
class AWQConfig(PreProcessorConfig):
    method: METHOD = field(default=METHOD.AWQ)
    format: FORMAT = field(default=FORMAT.GEMM)
    # Experimental quantization-time AdjacentExact policy. Checkpoints retain
    # the policy for reproducibility and optional future requantization.
    adjacent_model: Optional[Any] = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"help": "Optional AdjacentModelConfig used during AWQ quantization; disabled by default."},
    )
    scale_search_chunked_activations: bool = field(
        default=True,
        metadata={
            "help": "Stream and chunk AWQ scale-search activations to reduce peak memory during reference and reconstruction forwards."
        },
    )
    enable_activation_x_mean_cache: bool = field(
        default=True,
        metadata={
            "help": "Share same-input AWQ activation CPU captures and chunked activation x-mean reductions within one processor subset."
        },
    )
    scale_search_gpu_weight_restore: bool = field(
        default=True,
        metadata={
            "help": "Keep AWQ scale-search restore weights on GPU when there is enough free device memory; otherwise fall back to CPU restore."
        },
    )
    scale_search_refine_steps: int = field(
        default=0,
        metadata={
            "help": "Subdivisions per interval for two-stage AWQ ratio refinement. The stable default 0 retains only the canonical coarse grid. Use dynamic module patterns for per-group overrides."
        },
    )

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.AWQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return AWQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        # AWQ runtimes do not use GPTQ-style activation reordering unless the
        # checkpoint explicitly asks for it.
        return False

    def __post_init__(self):
        self.method = _normalize_quant_method(self.method)
        self.format = _normalize_format(self.format)
        self.adjacent_model = _normalize_adjacent_model(self.adjacent_model)
        if self.format not in self.supported_export_formats():
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.GEMM}`")
            self.format = FORMAT.GEMM
        value = self.scale_search_refine_steps
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value == 1:
            raise ValueError(
                "AWQConfig: `scale_search_refine_steps` must be 0 or an integer greater than 1."
            )
        for pattern, overrides in (self.dynamic or {}).items():
            if "scale_search_refine_steps" not in overrides:
                continue
            value = overrides["scale_search_refine_steps"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value == 1:
                raise ValueError(
                    "AWQConfig: dynamic `scale_search_refine_steps` for pattern "
                    f"`{pattern}` must be 0 or an integer greater than 1."
                )
        super().__post_init__()

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["zero_point"] = not self.sym
        out["version"] = self.format
        out[FORMAT_FIELD_CODE] = self.format

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        super()._update_meta_payload(meta_payload)
        if self.adjacent_model is None:
            meta_payload.pop("adjacent_model", None)
        else:
            meta_payload["adjacent_model"] = _serialize_adjacent_model(self.adjacent_model)
        meta_payload["scale_search_chunked_activations"] = self.scale_search_chunked_activations
        meta_payload["enable_activation_x_mean_cache"] = self.enable_activation_x_mean_cache
        meta_payload["scale_search_gpu_weight_restore"] = self.scale_search_gpu_weight_restore
        meta_payload["scale_search_refine_steps"] = self.scale_search_refine_steps


@dataclass
class ParoConfig(PreProcessorConfig):
    method: METHOD = field(default=METHOD.PARO)
    format: FORMAT = field(default=FORMAT.PAROQUANT)
    # Generic quantized-linear plumbing still reads this internal sentinel, but
    # GPTQ activation ordering is not part of ParoQuant's algorithm or format.
    desc_act: bool = field(default=False, init=False, repr=False, compare=False)
    krot: int = field(default=8)
    opt_rotation_epochs: int = field(default=10)
    opt_finetune_epochs: int = field(default=10)
    opt_train_samples: int = field(default=2048)
    opt_validation_samples: int = field(default=64)
    opt_batch_size: int = field(default=64)
    opt_rotation_lr: float = field(default=0.05)
    opt_weight_lr: float = field(default=1e-5)
    opt_quantizer_lr: float = field(default=1e-6)
    opt_pair_ratio: float = field(default=0.5)
    opt_seed: int = field(default=0)
    opt_optimizer: str = field(default="adamw")
    opt_weight_decay: float = field(default=0.01)
    opt_betas: Tuple[float, float] = field(default=(0.9, 0.95))
    opt_eps: float = field(default=1e-10)
    opt_amsgrad: bool = field(default=False)
    opt_sgd_momentum: float = field(default=0.0)
    opt_sgd_dampening: float = field(default=0.0)
    opt_sgd_nesterov: bool = field(default=False)
    opt_fused_rotation: bool = field(default=True)
    opt_gradient_checkpointing: Optional[bool] = field(default=None)
    opt_stage_cudagraph: bool = field(default=True)
    opt_best_state_dtype: Union[str, torch.dtype] = field(default="fp32")
    opt_train_on_noisy_inputs: bool = field(default=False)
    opt_scope: str = field(default="module")
    opt_stage_impl: str = field(default="fast")
    opt_pair_impl: str = field(default="fast")
    opt_quantizer_impl: str = field(default="reference")
    opt_channel_scale_clamp_min: float = field(default=PAROQUANT_OPT_SCALE_CLAMP_MIN_DEFAULT)
    opt_channel_scale_clamp_max: float = field(default=PAROQUANT_OPT_SCALE_CLAMP_MAX_DEFAULT)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.PARO,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return PAROQUANT_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        # Old generic configs may carry per-layer activation-order overrides.
        # They have never affected ParoQuant weights, so discard them.
        layer_dict.pop("desc_act", None)
        super()._normalize_dynamic_layer_config(
            layer_name,
            layer_dict,
            valid_bit_widths=valid_bit_widths,
            checkpoint_format=checkpoint_format,
        )

    @staticmethod
    def default_opt_gradient_checkpointing_for_scope(opt_scope: str) -> bool:
        """Enable activation checkpointing by default only for whole-layer optimization."""
        return str(opt_scope).strip().lower() == "layer"

    def __post_init__(self):
        self.method = _normalize_quant_method(self.method)
        self.format = _normalize_format(self.format)
        if self.format != FORMAT.PAROQUANT:
            log.info(f"QuantizeConfig: Auto fix `format` to `{FORMAT.PAROQUANT}`")
            self.format = FORMAT.PAROQUANT
        super().__post_init__()
        self.krot = int(self.krot)
        if self.krot <= 0:
            raise ValueError("ParoConfig: `krot` must be a positive integer.")
        self.opt_rotation_epochs = int(self.opt_rotation_epochs)
        self.opt_finetune_epochs = int(self.opt_finetune_epochs)
        self.opt_train_samples = int(self.opt_train_samples)
        self.opt_validation_samples = int(self.opt_validation_samples)
        self.opt_batch_size = int(self.opt_batch_size)
        self.opt_rotation_lr = float(self.opt_rotation_lr)
        self.opt_weight_lr = float(self.opt_weight_lr)
        self.opt_quantizer_lr = float(self.opt_quantizer_lr)
        self.opt_pair_ratio = float(self.opt_pair_ratio)
        self.opt_seed = int(self.opt_seed)
        self.opt_optimizer = str(self.opt_optimizer).strip().lower()
        self.opt_weight_decay = float(self.opt_weight_decay)
        if not isinstance(self.opt_betas, (list, tuple)) or len(self.opt_betas) != 2:
            raise ValueError("ParoConfig: `opt_betas` must be a 2-tuple/list of floats.")
        self.opt_betas = (float(self.opt_betas[0]), float(self.opt_betas[1]))
        self.opt_eps = float(self.opt_eps)
        self.opt_amsgrad = bool(self.opt_amsgrad)
        self.opt_sgd_momentum = float(self.opt_sgd_momentum)
        self.opt_sgd_dampening = float(self.opt_sgd_dampening)
        self.opt_sgd_nesterov = bool(self.opt_sgd_nesterov)
        self.opt_fused_rotation = bool(self.opt_fused_rotation)
        self.opt_scope = str(self.opt_scope).strip().lower()
        checkpointing = self.opt_gradient_checkpointing
        if isinstance(checkpointing, str):
            normalized_checkpointing = checkpointing.strip().lower()
            if normalized_checkpointing in {"1", "true", "yes", "on", "y", "t"}:
                checkpointing = True
            elif normalized_checkpointing in {"0", "false", "no", "off", "n", "f"}:
                checkpointing = False
            else:
                raise ValueError(
                    "ParoConfig: `opt_gradient_checkpointing` string values must be one of "
                    "{'1','0','true','false','yes','no','on','off','y','n','t','f'}."
                )
        if checkpointing is None:
            checkpointing = self.default_opt_gradient_checkpointing_for_scope(self.opt_scope)
        self.opt_gradient_checkpointing = bool(checkpointing)
        self.opt_stage_cudagraph = bool(self.opt_stage_cudagraph)
        self.opt_best_state_dtype = _normalize_paroquant_best_state_dtype(self.opt_best_state_dtype)
        self.opt_train_on_noisy_inputs = bool(self.opt_train_on_noisy_inputs)
        self.opt_stage_impl = str(self.opt_stage_impl).strip().lower()
        self.opt_pair_impl = str(self.opt_pair_impl).strip().lower()
        self.opt_quantizer_impl = str(self.opt_quantizer_impl).strip().lower()
        self.opt_channel_scale_clamp_min = float(self.opt_channel_scale_clamp_min)
        self.opt_channel_scale_clamp_max = float(self.opt_channel_scale_clamp_max)
        if self.opt_rotation_epochs < 0 or self.opt_finetune_epochs < 0:
            raise ValueError("ParoConfig: optimization epochs must be non-negative.")
        if self.opt_train_samples <= 0 or self.opt_validation_samples <= 0:
            raise ValueError("ParoConfig: optimization sample counts must be positive.")
        if self.opt_batch_size <= 0:
            raise ValueError("ParoConfig: `opt_batch_size` must be positive.")
        if self.opt_rotation_lr <= 0 or self.opt_weight_lr <= 0 or self.opt_quantizer_lr <= 0:
            raise ValueError("ParoConfig: optimization learning rates must be positive.")
        if not (0.0 < self.opt_pair_ratio <= 0.5):
            raise ValueError("ParoConfig: `opt_pair_ratio` must be in the interval (0, 0.5].")
        if self.opt_optimizer not in {"adamw", "adam", "sgd"}:
            raise ValueError("ParoConfig: `opt_optimizer` must be one of {'adamw', 'adam', 'sgd'}.")
        if self.opt_weight_decay < 0:
            raise ValueError("ParoConfig: `opt_weight_decay` must be non-negative.")
        if self.opt_eps <= 0:
            raise ValueError("ParoConfig: `opt_eps` must be positive.")
        if not all(0.0 <= beta < 1.0 for beta in self.opt_betas):
            raise ValueError("ParoConfig: `opt_betas` values must be in the interval [0, 1).")
        if self.opt_sgd_momentum < 0:
            raise ValueError("ParoConfig: `opt_sgd_momentum` must be non-negative.")
        if self.opt_sgd_dampening < 0:
            raise ValueError("ParoConfig: `opt_sgd_dampening` must be non-negative.")
        if self.opt_sgd_nesterov and self.opt_sgd_momentum <= 0:
            raise ValueError("ParoConfig: `opt_sgd_nesterov=True` requires `opt_sgd_momentum > 0`.")
        if self.opt_sgd_nesterov and self.opt_sgd_dampening != 0:
            raise ValueError("ParoConfig: `opt_sgd_nesterov=True` requires `opt_sgd_dampening == 0`.")
        if self.opt_scope not in {"module", "compute_block", "layer"}:
            raise ValueError("ParoConfig: `opt_scope` must be one of {'module', 'compute_block', 'layer'}.")
        if self.opt_stage_impl not in {"fast", "reference"}:
            raise ValueError("ParoConfig: `opt_stage_impl` must be one of {'fast', 'reference'}.")
        if self.opt_pair_impl not in {"fast", "reference"}:
            raise ValueError("ParoConfig: `opt_pair_impl` must be one of {'fast', 'reference'}.")
        if self.opt_quantizer_impl not in {"fast", "reference"}:
            raise ValueError("ParoConfig: `opt_quantizer_impl` must be one of {'fast', 'reference'}.")
        if self.opt_channel_scale_clamp_min <= 0 or self.opt_channel_scale_clamp_max <= 0:
            raise ValueError("ParoConfig: scale clamp bounds must be positive.")
        if self.opt_channel_scale_clamp_min >= self.opt_channel_scale_clamp_max:
            raise ValueError(
                "ParoConfig: `opt_channel_scale_clamp_min` must be smaller than "
                "`opt_channel_scale_clamp_max`."
            )

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {
            "krot": self.krot,
        }

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        meta_payload["opt_rotation_epochs"] = self.opt_rotation_epochs
        meta_payload["opt_finetune_epochs"] = self.opt_finetune_epochs
        meta_payload["opt_train_samples"] = self.opt_train_samples
        meta_payload["opt_validation_samples"] = self.opt_validation_samples
        meta_payload["opt_batch_size"] = self.opt_batch_size
        meta_payload["opt_rotation_lr"] = self.opt_rotation_lr
        meta_payload["opt_weight_lr"] = self.opt_weight_lr
        meta_payload["opt_quantizer_lr"] = self.opt_quantizer_lr
        meta_payload["opt_pair_ratio"] = self.opt_pair_ratio
        meta_payload["opt_seed"] = self.opt_seed
        meta_payload["opt_optimizer"] = self.opt_optimizer
        meta_payload["opt_weight_decay"] = self.opt_weight_decay
        meta_payload["opt_betas"] = list(self.opt_betas)
        meta_payload["opt_eps"] = self.opt_eps
        meta_payload["opt_amsgrad"] = self.opt_amsgrad
        meta_payload["opt_sgd_momentum"] = self.opt_sgd_momentum
        meta_payload["opt_sgd_dampening"] = self.opt_sgd_dampening
        meta_payload["opt_sgd_nesterov"] = self.opt_sgd_nesterov
        meta_payload["opt_fused_rotation"] = self.opt_fused_rotation
        meta_payload["opt_gradient_checkpointing"] = self.opt_gradient_checkpointing
        meta_payload["opt_stage_cudagraph"] = self.opt_stage_cudagraph
        meta_payload["opt_best_state_dtype"] = self.opt_best_state_dtype
        meta_payload["opt_train_on_noisy_inputs"] = self.opt_train_on_noisy_inputs
        meta_payload["opt_scope"] = self.opt_scope
        meta_payload["opt_stage_impl"] = self.opt_stage_impl
        meta_payload["opt_pair_impl"] = self.opt_pair_impl
        meta_payload["opt_quantizer_impl"] = self.opt_quantizer_impl
        meta_payload["opt_channel_scale_clamp_min"] = self.opt_channel_scale_clamp_min
        meta_payload["opt_channel_scale_clamp_max"] = self.opt_channel_scale_clamp_max

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out.pop("desc_act", None)
        out["zero_point"] = not self.sym
        out["krot"] = self.krot
        out[FORMAT_FIELD_CODE] = self.format


@dataclass
class QQQConfig(GPTQConfig):
    method: METHOD = field(default=METHOD.QQQ)
    format: FORMAT = field(default=FORMAT.QQQ)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.QQQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return QQQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return True

    def __post_init__(self):
        super().__post_init__()

        # QQQ's tuned default damp_percent is much lower than GPTQ's; use a
        # static DampConfig unless the user explicitly requested adaptive damping.
        if getattr(self, "_adaptive_damping_user_value", None) is None:
            fixed_damp = getattr(self, "_damp_percent_user_value", self.damp_percent)
            fixed_step = getattr(self, "_damp_auto_increment_user_value", self.damp_auto_increment)
            self.adaptive_damping = DampConfig(
                min=fixed_damp,
                max=fixed_damp,
                step=fixed_step,
            )


@dataclass
class FP8Config(PreProcessorConfig):
    bits: int = field(default=8, metadata={"choices": [8]})
    method: METHOD = field(default=METHOD.FP8)
    format: Optional[str] = field(default="float8_e4m3fn")
    group_size: int = field(default=-1)
    desc_act: Optional[bool] = field(default=False)
    sym: bool = field(default=True)
    weight_scale_method: str = field(default="row")
    weight_block_size: Optional[Union[List[int], Tuple[int, int]]] = field(default=None)
    weight_scale_semantics: str = field(default="inverse")

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_fp8_fmt(self.format)
        return FORMAT.FP8

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.FP8,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return FP8_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

        if self.bits != 8:
            raise ValueError("FP8Config: `bits` must be `8`.")

        if self.method != METHOD.FP8:
            raise ValueError("FP8Config: `method` must be `fp8`.")

        self.group_size = -1
        self.desc_act = False
        self.sym = True

        self.format = _normalize_fp8_fmt(self.format)
        block_size = _normalize_fp8_weight_block_size(self.weight_block_size)
        self.weight_scale_method = _normalize_fp8_weight_scale_method(
            self.weight_scale_method,
            weight_block_size=block_size,
        )
        self.weight_block_size = list(block_size) if block_size is not None else None
        self.weight_scale_semantics = _normalize_fp8_scale_semantics(self.weight_scale_semantics)

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }
            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=[8],
                    checkpoint_format=FORMAT.FP8,
                )

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        del valid_bit_widths, checkpoint_format
        if "bits" in layer_dict and int(layer_dict["bits"]) != 8:
            raise ValueError(f"FP8Config: layer `{layer_name}` only supports 8-bit FP8 weights.")
        if "group_size" in layer_dict and layer_dict["group_size"] not in (-1, None):
            raise ValueError("FP8Config: `group_size` is not used; keep it at `-1`.")

        block_size = _normalize_fp8_weight_block_size(layer_dict.get("weight_block_size"))
        raw_format = layer_dict.get(FORMAT_FIELD_CODE, layer_dict.get("fmt"))
        if raw_format is not None:
            layer_dict[FORMAT_FIELD_CODE] = _normalize_fp8_fmt(raw_format)
        layer_dict.pop("fmt", None)
        if "weight_scale_method" in layer_dict or block_size is not None:
            layer_dict["weight_scale_method"] = _normalize_fp8_weight_scale_method(
                layer_dict.get("weight_scale_method"),
                weight_block_size=block_size,
            )
        if "weight_scale_semantics" in layer_dict:
            layer_dict["weight_scale_semantics"] = _normalize_fp8_scale_semantics(
                layer_dict["weight_scale_semantics"]
            )
        if "weight_block_size" in layer_dict:
            layer_dict["weight_block_size"] = list(block_size) if block_size is not None else None

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "weight_scale_method": self.weight_scale_method,
            "weight_block_size": self.weight_block_size,
            "weight_scale_semantics": self.weight_scale_semantics,
        }

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format
        out["weight_scale_method"] = self.weight_scale_method
        out["weight_block_size"] = self.weight_block_size
        out["weight_scale_semantics"] = self.weight_scale_semantics

    def uses_weight_only_lifecycle(self) -> bool:
        return True


@dataclass
class BitsAndBytesConfig(PreProcessorConfig):
    bits: int = field(default=4, metadata={"choices": [4, 8]})
    method: METHOD = field(default=METHOD.BITSANDBYTES)
    format: Optional[str] = field(default=None)
    group_size: int = field(default=-1)
    desc_act: Optional[bool] = field(default=False)
    sym: bool = field(default=True)
    block_size: int = field(default=64)
    compress_statistics: bool = field(default=True)

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_bitsandbytes_format(self.format, bits=int(self.bits))
        return FORMAT.BITSANDBYTES

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.BITSANDBYTES,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return BITSANDBYTES_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

        if self.bits not in {4, 8}:
            raise ValueError("BitsAndBytesConfig: `bits` must be `4` or `8`.")
        if self.method != METHOD.BITSANDBYTES:
            raise ValueError("BitsAndBytesConfig: `method` must be `bitsandbytes`.")

        self.group_size = -1
        self.desc_act = False
        self.sym = True

        self.format = _normalize_bitsandbytes_format(self.format, bits=int(self.bits))
        self.block_size = _normalize_bitsandbytes_block_size(self.block_size)
        self.compress_statistics = bool(self.compress_statistics)

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }
            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=[4, 8],
                    checkpoint_format=FORMAT.BITSANDBYTES,
                )

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        del valid_bit_widths, checkpoint_format
        if "bits" in layer_dict and int(layer_dict["bits"]) not in {4, 8}:
            raise ValueError(f"BitsAndBytesConfig: layer `{layer_name}` only supports 4-bit or 8-bit weights.")
        if "group_size" in layer_dict and layer_dict["group_size"] not in (-1, None):
            raise ValueError("BitsAndBytesConfig: `group_size` is not used; keep it at `-1`.")
        if "desc_act" in layer_dict and bool(layer_dict["desc_act"]):
            raise ValueError("BitsAndBytesConfig: `desc_act` is not supported.")
        if "sym" in layer_dict and layer_dict["sym"] is not True:
            raise ValueError("BitsAndBytesConfig: `sym` must stay `True`.")
        dynamic_bits = int(layer_dict.get("bits", self.bits))
        raw_format = layer_dict.get(FORMAT_FIELD_CODE, layer_dict.get("bnb_quant_type"))
        if raw_format is not None:
            layer_dict[FORMAT_FIELD_CODE] = _normalize_bitsandbytes_format(raw_format, bits=dynamic_bits)
        if "block_size" in layer_dict or "bnb_block_size" in layer_dict:
            layer_dict["block_size"] = _normalize_bitsandbytes_block_size(
                layer_dict.get("block_size", layer_dict.get("bnb_block_size"))
            )
        layer_dict.pop("bnb_block_size", None)
        if "compress_statistics" in layer_dict or "bnb_compress_statistics" in layer_dict:
            layer_dict["compress_statistics"] = bool(
                layer_dict.get("compress_statistics", layer_dict.get("bnb_compress_statistics"))
            )
        layer_dict.pop("bnb_compress_statistics", None)

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "block_size": self.block_size,
            "compress_statistics": self.compress_statistics,
        }

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format
        out["block_size"] = self.block_size
        out["compress_statistics"] = self.compress_statistics

    def uses_weight_only_lifecycle(self) -> bool:
        return True

    @property
    def bnb_quant_type(self) -> str:
        return self.format

    @bnb_quant_type.setter
    def bnb_quant_type(self, value: str) -> None:
        self.format = _normalize_bitsandbytes_format(value, bits=int(self.bits))

    @property
    def bnb_block_size(self) -> int:
        return self.block_size

    @bnb_block_size.setter
    def bnb_block_size(self, value: int) -> None:
        self.block_size = _normalize_bitsandbytes_block_size(value)

    @property
    def bnb_compress_statistics(self) -> bool:
        return self.compress_statistics

    @bnb_compress_statistics.setter
    def bnb_compress_statistics(self, value: bool) -> None:
        self.compress_statistics = bool(value)


@dataclass
class EXL3Config(BaseQuantizeConfig):
    bits: float = field(default=3.0)
    method: METHOD = field(default=METHOD.EXL3)
    format: FORMAT = field(default=FORMAT.EXL3)
    group_size: int = field(default=-1)
    desc_act: Optional[bool] = field(default=False)
    sym: bool = field(default=True)
    head_bits: Optional[float] = field(default=None)
    out_scales: Optional[str] = field(default="auto")
    codebook: str = field(default="mcg")
    tensor_storage: Optional[Dict[str, Any]] = field(default=None)
    calibration: Optional[Dict[str, int]] = field(default=None)

    @property
    def runtime_bits(self) -> int:
        return quant_bits_width(self.bits)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.EXL3,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return EXL3_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def _normalize_bits_field(self, bits_value, checkpoint_format: FORMAT):
        return _normalize_exl3_bits(bits_value)

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        del valid_bit_widths, checkpoint_format
        for key, value in layer_dict.items():
            if key == "bits":
                layer_dict[key] = _normalize_exl3_bits(value)
            elif key == "head_bits":
                layer_dict[key] = None if value is None else _normalize_exl3_bits(value)
            elif key == "group_size" and value not in (-1, None):
                raise ValueError("EXL3Config: `group_size` is not used; keep it at `-1`.")

    def __post_init__(self):
        self.method = _normalize_quant_method(self.method)
        self.format = _normalize_format(self.format)
        self.pack_dtype = _normalize_pack_dtype(self.pack_dtype)
        self.bits = _normalize_exl3_bits(self.bits)
        self.head_bits = None if self.head_bits is None else _normalize_exl3_bits(self.head_bits)

        if self.method != METHOD.EXL3:
            raise ValueError("EXL3Config: `method` must be `exl3`.")
        if self.format != FORMAT.EXL3:
            raise ValueError("EXL3Config: `format` must be `exl3`.")

        self.group_size = -1
        self.desc_act = False
        self.sym = True

        self.fallback = _normalize_fallback(self.fallback)

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')},
            }
            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=[],
                    checkpoint_format=FORMAT.EXL3,
                )

        if self.out_scales is not None:
            normalized_out_scales = str(self.out_scales).strip().lower()
            out_scale_aliases = {
                "always": "always",
                "true": "always",
                "never": "never",
                "false": "never",
                "auto": "auto",
                "none": "auto",
            }
            if normalized_out_scales not in out_scale_aliases:
                raise ValueError("EXL3Config: `out_scales` must be one of `always`, `never`, or `auto`.")
            self.out_scales = out_scale_aliases[normalized_out_scales]

        self.codebook = str(self.codebook).strip().lower()
        if self.codebook not in {"mcg", "mul1", "3inst"}:
            raise ValueError("EXL3Config: `codebook` must be one of `mcg`, `mul1`, or `3inst`.")

        if self.tensor_storage is not None and not isinstance(self.tensor_storage, dict):
            raise ValueError("EXL3Config: `tensor_storage` must be a dictionary when provided.")
        if self.calibration is not None:
            if not isinstance(self.calibration, dict):
                raise ValueError("EXL3Config: `calibration` must be a dictionary when provided.")
            self.calibration = {
                str(key): int(value)
                for key, value in self.calibration.items()
            }

        if self.meta is not None:
            if not isinstance(self.meta, dict):
                raise ValueError("QuantizeConfig: `meta` must be a dictionary")
            for key in self.meta:
                if not isinstance(key, str):
                    raise ValueError("QuantizeConfig: `meta` keys must be strings")
        else:
            self.meta = {}

        self.fused_forward = _normalize_fused_forward_config(self.fused_forward)
        self.adapter = normalize_adapter(self.adapter)

        # Rotation fuses orthogonal transforms into the weights and requires
        # materialized tensors; meta-device/shell loading cannot be used.
        if self.rotation and self.offload_to_disk:
            log.warn(f"{self.__class__.__name__}: `rotation` is incompatible with `offload_to_disk`; disabling disk offload.")
            self.offload_to_disk = False

        self._ensure_offload_temp_dir()

        self.dense_vram_strategy = _normalize_dense_vram_strategy(self.dense_vram_strategy)
        self.dense_vram_strategy_devices = _normalize_strategy_devices(
            self.dense_vram_strategy_devices,
            field_name="dense_vram_strategy_devices",
        )
        self.moe_vram_strategy = _normalize_moe_vram_strategy(self.moe_vram_strategy)
        self.moe_vram_strategy_devices = _normalize_strategy_devices(
            self.moe_vram_strategy_devices,
            field_name="moe_vram_strategy_devices",
        )
        self.gc_mode = _normalize_gc_mode(self.gc_mode)
        self.moe = _normalize_moe_config(self.moe)

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["bits"] = float(self.bits)
        out["head_bits"] = None if self.head_bits is None else float(self.head_bits)
        out["out_scales"] = self.out_scales
        out["codebook"] = self.codebook
        out["tensor_storage"] = self.tensor_storage
        out["calibration"] = self.calibration

    def calculate_bits_per_weight(self):
        head_bits = self.head_bits if self.head_bits is not None else self.bits
        log.info(
            "Estimated Quantization BPW (bits per weight): %s bpw, based on [bits: %s, head_bits: %s]",
            self.bits,
            self.bits,
            head_bits,
        )


@dataclass
class QVQActivationConfig:
    """Optional FP8 activation target for the V2B2-P32 weight codec.

    Dynamic per-token scaling is shared by calibration and inference, so the
    Hessian sees the same A8 values consumed by the runtime without retaining
    a calibration-sized activation cache or introducing order-dependent
    observer state. Enabling A8 also requires FP8 E4M3 K/V-cache storage for
    every cache-enabled decoder forward; callers cannot substitute a dense
    Transformers cache.
    """

    bits: int = 8
    format: str = QVQ_FP8_ACTIVATION_FORMAT
    scale_method: str = QVQ_FP8_ACTIVATION_SCALE_METHOD
    target: str = "p32_operand"
    kernel_mode: str = "auto"
    replay_passes: int = 1
    replay_max_rows: int = 2048
    replay_validation_fraction: float = 0.125

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits != 8:
            raise ValueError("QVQActivationConfig: `bits` must be 8.")
        self.format = normalize_qvq_fp8_activation_format(self.format)
        self.scale_method = normalize_qvq_fp8_activation_scale_method(self.scale_method)
        if not isinstance(self.target, str):
            raise TypeError("QVQActivationConfig: `target` must be a string.")
        self.target = self.target.strip().lower().replace("-", "_")
        if self.target not in {"p32_operand", "linear_input"}:
            raise ValueError("QVQActivationConfig: `target` must be `p32_operand` or `linear_input`.")
        if not isinstance(self.kernel_mode, str):
            raise TypeError("QVQActivationConfig: `kernel_mode` must be a string.")
        self.kernel_mode = self.kernel_mode.strip().lower().replace("-", "_")
        if self.kernel_mode not in {"auto", "require", "disable"}:
            raise ValueError("QVQActivationConfig: `kernel_mode` must be `auto`, `require`, or `disable`.")
        if isinstance(self.replay_passes, bool) or not isinstance(self.replay_passes, int):
            raise TypeError("QVQActivationConfig: `replay_passes` must be an integer.")
        if self.replay_passes not in {0, 1}:
            raise ValueError("QVQActivationConfig: `replay_passes` must be 0 or 1.")
        if (
            isinstance(self.replay_max_rows, bool)
            or not isinstance(self.replay_max_rows, int)
            or self.replay_max_rows < 16
        ):
            raise ValueError("QVQActivationConfig: `replay_max_rows` must be an integer >= 16.")
        if isinstance(self.replay_validation_fraction, bool) or not isinstance(
            self.replay_validation_fraction, (int, float)
        ):
            raise TypeError("QVQActivationConfig: `replay_validation_fraction` must be a real scalar.")
        self.replay_validation_fraction = float(self.replay_validation_fraction)
        if not 0.0 < self.replay_validation_fraction < 0.5:
            raise ValueError(
                "QVQActivationConfig: `replay_validation_fraction` must be in (0, 0.5)."
            )


def _normalize_qvq_activation_config(
    value: Optional[Union[QVQActivationConfig, Dict[str, Any], bool]],
) -> Optional[QVQActivationConfig]:
    if value is None or value is False:
        return None
    if value is True:
        return QVQActivationConfig()
    if isinstance(value, QVQActivationConfig):
        value.__post_init__()
        return value
    if isinstance(value, dict):
        return QVQActivationConfig(**value)
    raise TypeError(
        "QVQConfig: `activation` must be a QVQActivationConfig, dictionary, boolean, or None."
    )


@dataclass
class OutputAlignConfig:
    """Offline decoder-layer output alignment for fixed QVQ trellises.

    This stage is opt-in. Pass ``OutputAlignConfig(...)`` through
    :class:`QVQConfig` to enable it; the default ``output_alignment=None``
    retains the unaligned QVQ baseline.
    """

    learning_rate: float = 1e-5
    epochs: int = 1
    optimizer: str = "adam"
    weight_decay: float = 0.0
    maximum_train_batches: int = 32
    maximum_validation_batches: int = 16
    validation_fraction: float = 0.2
    minimum_relative_improvement: float = 0.0
    # QTIP derives every input Hessian from the untouched dense model before
    # committing blockwise corrections. Keep that contract inside the
    # Keep QTIP's pristine dense-Hessian contract instead of mixing later
    # layers with activations already perturbed by earlier QVQ layers.
    pristine_hessian: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.pristine_hessian, bool):
            raise TypeError("QVQ output alignment `pristine_hessian` must be a boolean.")

        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise TypeError("QVQ output alignment `learning_rate` must be a real scalar.")
        self.learning_rate = float(self.learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("QVQ output alignment `learning_rate` must be finite and positive.")

        if not isinstance(self.optimizer, str):
            raise TypeError("QVQ output alignment `optimizer` must be a string.")
        self.optimizer = self.optimizer.strip().lower()
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("QVQ output alignment `optimizer` must be `adam` or `adamw`.")
        if isinstance(self.weight_decay, bool) or not isinstance(self.weight_decay, (int, float)):
            raise TypeError("QVQ output alignment `weight_decay` must be a real scalar.")
        self.weight_decay = float(self.weight_decay)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("QVQ output alignment `weight_decay` must be finite and nonnegative.")

        for field_name in ("epochs", "maximum_train_batches", "maximum_validation_batches"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"QVQ output alignment `{field_name}` must be a positive integer.")

        for field_name, upper_bound in (
            ("validation_fraction", 1.0),
            ("minimum_relative_improvement", None),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"QVQ output alignment `{field_name}` must be a real scalar.")
            value = float(value)
            if not math.isfinite(value) or value < 0 or (upper_bound is not None and (value <= 0 or value >= upper_bound)):
                bound = " in `(0, 1)`" if upper_bound is not None else " finite and nonnegative"
                raise ValueError(f"QVQ output alignment `{field_name}` must be{bound}.")
            setattr(self, field_name, value)


def _normalize_qvq_output_alignment_config(
    value: Optional[Union[OutputAlignConfig, Dict[str, Any]]],
) -> Optional[OutputAlignConfig]:
    if value is None:
        return None
    if isinstance(value, OutputAlignConfig):
        value.__post_init__()
        return value
    if isinstance(value, dict):
        return OutputAlignConfig(**value)
    raise TypeError("QVQConfig: `output_alignment` must be an OutputAlignConfig, dictionary, or None.")


def _normalize_module_granular_replay_config(
    value: Optional[Union[ModuleGranularReplayConfig, Dict[str, Any], bool]],
) -> Optional[ModuleGranularReplayConfig]:
    if value is None or value is False:
        return None
    if value is True:
        return ModuleGranularReplayConfig()
    if isinstance(value, ModuleGranularReplayConfig):
        value.__post_init__()
        return value
    if isinstance(value, dict):
        return ModuleGranularReplayConfig(**value)
    raise TypeError(
        "QVQConfig: `module_granular_replay` must be a ModuleGranularReplayConfig, dictionary, boolean, or None."
    )


@dataclass
class QVQConfig(BaseQuantizeConfig):
    """QVQ trellis-code configuration.

    The planar-only contract uses GPT-QModel's versioned PGC16 state decoder.
    It is deliberately separate from GPTQ's affine groupwise tensor layout:
    ``bits`` is the trellis rate K, not an integer weight code.
    """

    bits: float = field(default=2, metadata={"choices": list(QVQ_BITS)})
    method: METHOD = field(default=METHOD.QVQ)
    format: FORMAT = field(default=FORMAT.QVQ)
    group_size: int = field(default=-1)
    # Remove GPTQ activation ordering from QVQ's dataclass/config schema. Some
    # generic loader paths still read the class-level fixed-false capability.
    desc_act: ClassVar[bool] = False
    sym: bool = field(default=True)
    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    codebook: str = field(default=PGC16_CODEBOOK_VERSION)
    trellis_window: int = field(default=16)
    vector_size: int = field(default=2)
    # Opt-in V4 bank count. Four rate-keyed banks add a per-tile selector to
    # the serialized payload; the default remains the single canonical bank.
    bank_count: int = field(default=1)
    # Explicit held-out propagation gate. Callers must provide disjoint
    # module inputs/targets through the lifecycle attachment; never inferred
    # from calibration or benchmark rows.
    # None means use the processor's default automatic gate for banked V4
    # Block-LDLQ. False is an explicit opt-out for A/B comparisons; True is an
    # explicit request and still requires a supplied/derived gate.
    propagated_bank_selection: Optional[bool] = field(default=None)
    # These fields describe YAQA/Hessian processing geometry.
    tile_rows: int = field(default=16)
    tile_cols: int = field(default=16)
    # YAQA is the production QVQ lifecycle default; callers that need the
    # local-only baseline must request rounding="block_ldlq" explicitly.
    rounding: str = field(default="yaqa")
    yaqa: YaqaConfig = field(default_factory=YaqaConfig)
    # Exact Viterbi survivor-pruning policy threaded to the CUDA V2 segmented
    # grid dispatch. A missing key deserializes to `auto`, which reproduces
    # the historical automatic behavior exactly.
    viterbi_pruning: ViterbiPruningConfig = field(default_factory=ViterbiPruningConfig)
    incoherence: str = field(default="rht")
    module_scale_search: bool = field(default=False)
    output_channel_scale_optimization: bool = field(default=False)
    viterbi_objective: str = field(default="euclidean")
    tail_biting_candidates: int = field(default=1)
    viterbi_minimum_proxy_improvement: float = field(default=0.0)
    output_alignment: Optional[OutputAlignConfig] = field(default=None)
    module_granular_replay: Optional[ModuleGranularReplayConfig] = field(default=None)
    smooth_swiglu: Optional[SmoothSwiGLUConfig] = field(default=None)
    # Opt-in W2--W3.5/A8 calibration and inference. None preserves the exact
    # historical dense-activation QVQ contract.
    activation: Optional[QVQActivationConfig] = field(default=None)
    tensor_storage: Optional[Dict[str, Any]] = field(default=None)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.QVQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return QVQ_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def _bits_in_choices(self, valid_bits: List[Union[int, float]]) -> bool:
        """Validate the exact planar rate instead of its floored storage width."""

        return self.bits in valid_bits

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        del valid_bit_widths
        unsupported = set(layer_dict) - {"bits", "format", "yaqa_regularization"}
        if unsupported:
            raise ValueError(
                f"QVQConfig: layer `{layer_name}` only supports `bits`, `format`, and `yaqa_regularization` overrides; "
                f"got {sorted(unsupported)}."
            )
        layer_format = checkpoint_format
        if "format" in layer_dict:
            raw_format = layer_dict["format"]
            try:
                layer_format = _normalize_format(raw_format)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"QVQConfig: layer `{layer_name}` has unsupported dynamic format `{raw_format}`."
                ) from exc
            if layer_format not in QVQ_EXPORT_FORMATS:
                raise ValueError(f"QVQConfig: layer `{layer_name}` cannot use format `{layer_format.value}`.")
            layer_dict["format"] = layer_format.value
        if "bits" in layer_dict:
            layer_bits = _normalize_quant_bits(layer_dict["bits"], format_value=layer_format)
            if layer_bits not in QVQ_BITS:
                raise ValueError(
                    f"QVQConfig: layer `{layer_name}` only supports integer or half-integer rates from 1 through 8."
                )
            if layer_format == FORMAT.QVQ_V4 and layer_bits > 4:
                raise ValueError(
                    f"QVQConfig: layer `{layer_name}` with `format=qvq_v4` only supports rates W1 through W4."
                )
            if layer_format == FORMAT.QVQ_V4_L18 and layer_bits > 2.5:
                raise ValueError(
                    f"QVQConfig: layer `{layer_name}` with `format=qvq_v4_l18` only supports rates W1 through W2.5."
                )
            if layer_format == FORMAT.QVQ_V2B4_P64 and layer_bits > 3.5:
                raise ValueError(
                    f"QVQConfig: layer `{layer_name}` with `format=qvq_v2b4_p64` only supports W1 through W3.5."
                )
            if layer_format == FORMAT.QVQ_V2B2_P32 and layer_bits > 3.5:
                raise ValueError(
                    f"QVQConfig: layer `{layer_name}` with `format=qvq_v2b2_p32` only supports W1 through W3.5."
                )
            layer_dict["bits"] = layer_bits
        if "yaqa_regularization" in layer_dict:
            value = layer_dict["yaqa_regularization"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    f"QVQConfig: layer `{layer_name}` `yaqa_regularization` must be finite and nonnegative."
                )
            layer_dict["yaqa_regularization"] = float(value)

    def __post_init__(self):
        requested_group_size = self.group_size
        requested_sym = self.sym
        requested_pack_dtype = _normalize_pack_dtype(self.pack_dtype)
        super().__post_init__()

        # Apple exposes one process-wide MPS command stream. Data-parallel
        # calibration forwards can overlap encoders on that stream and trigger
        # Metal's "command encoder is already encoding" assertion. Keep the
        # numerical path unchanged but force the serial executor on MPS.
        if self.device is not None and torch.device(self.device).type == "mps":
            self.auto_forward_data_parallel = False

        if self.bits not in QVQ_BITS:
            raise ValueError(
                "QVQConfig: `bits` must be an integer or half-integer in `[1, 8]` for the PGC16 bitshift trellis."
            )
        if requested_group_size != -1:
            raise ValueError("QVQConfig: `group_size` is not part of the trellis format; keep it at `-1`.")
        if requested_sym is not True:
            raise ValueError("QVQConfig: affine asymmetric quantization is not part of the QVQ format.")
        if requested_pack_dtype != torch.int32:
            raise ValueError("QVQConfig: the planar trellis stream requires `pack_dtype=torch.int32`.")
        if isinstance(self.bank_count, bool) or not isinstance(self.bank_count, int) or self.bank_count not in (1, 2, 4):
            raise ValueError("QVQConfig: `bank_count` must be 1, 2, or 4.")

        if self.format == FORMAT.QVQ_V4:
            if self.bits > 4:
                raise ValueError("QVQConfig: `format=qvq_v4` supports only rates W1 through W4.")
            self.vector_size = 4
            self.trellis_window = 16
        elif self.format == FORMAT.QVQ_V4_L18:
            if self.bits > 2.5:
                raise ValueError("QVQConfig: `format=qvq_v4_l18` supports only rates W1 through W2.5.")
            self.vector_size = 4
            self.trellis_window = 18
            if self.bank_count != 1:
                raise ValueError("QVQConfig: `format=qvq_v4_l18` uses implicit history banks and requires bank_count=1.")
        elif self.format == FORMAT.QVQ_DUAL_V2:
            self.vector_size = 2
            self.trellis_window = 16
            if self.bank_count != 1:
                raise ValueError("QVQConfig: `format=qvq_dual_v2` requires bank_count=1.")
        elif self.format == FORMAT.QVQ_V2B4_P64:
            if self.bits > 3.5:
                raise ValueError("QVQConfig: `format=qvq_v2b4_p64` supports only rates W1 through W3.5.")
            if self.bank_count not in (1, 4):
                raise ValueError("QVQConfig: `format=qvq_v2b4_p64` requires bank_count=4.")
            self.vector_size = 2
            self.trellis_window = 16
            self.bank_count = 4
        elif self.format == FORMAT.QVQ_V2B2_P32:
            if self.bits > 3.5:
                raise ValueError("QVQConfig: `format=qvq_v2b2_p32` supports only rates W1 through W3.5.")
            if self.bank_count not in (1, 2):
                raise ValueError("QVQConfig: `format=qvq_v2b2_p32` requires bank_count=2.")
            self.vector_size = 2
            self.trellis_window = 16
            self.bank_count = 2

        self.activation = _normalize_qvq_activation_config(self.activation)
        if self.activation is not None:
            if self.format != FORMAT.QVQ_V2B2_P32:
                raise ValueError(
                    "QVQConfig: FP8 `activation` requires `format=qvq_v2b2_p32` "
                    "(`qvq_v2b2_g32` is accepted as an input alias)."
                )
            if self.bits not in (2, 2.5, 3, 3.5):
                raise ValueError("QVQConfig: FP8 activation quantization supports P32 rates W2 through W3.5.")

        self.codebook = str(self.codebook).strip().lower()
        pgc16_levels_for_version(self.codebook)
        if not isinstance(self.rounding, str):
            raise TypeError("QVQConfig: `rounding` must be a string.")
        self.rounding = self.rounding.strip().lower()
        if self.rounding not in {"block_ldlq", "yaqa"}:
            raise ValueError("QVQConfig: `rounding` must be `block_ldlq` or `yaqa`.")
        if (
            self.rounding == "yaqa"
            and self.activation is not None
            and self.activation.target == "p32_operand"
        ):
            raise ValueError(
                "QVQConfig: YAQA activation-aware calibration currently requires "
                "activation.target=`linear_input`; `p32_operand` needs a "
                "post-SU/Hadamard Sketch-B collector."
            )
        if isinstance(self.yaqa, dict):
            self.yaqa = YaqaConfig(**self.yaqa)
        elif isinstance(self.yaqa, YaqaConfig):
            self.yaqa.__post_init__()
        else:
            raise TypeError("QVQConfig: `yaqa` must be a YaqaConfig or dictionary.")
        if self.viterbi_pruning is None:
            self.viterbi_pruning = ViterbiPruningConfig()
        elif isinstance(self.viterbi_pruning, dict):
            self.viterbi_pruning = ViterbiPruningConfig(**self.viterbi_pruning)
        elif isinstance(self.viterbi_pruning, ViterbiPruningConfig):
            self.viterbi_pruning.__post_init__()
        else:
            raise TypeError("QVQConfig: `viterbi_pruning` must be a ViterbiPruningConfig or dictionary.")
        if (self.yaqa.spectral_refinement or self.yaqa.spectral_push or self.yaqa.spectral_localized) and (
            self.rounding != "yaqa" or self.format != FORMAT.QVQ_V2B2_P32
        ):
            raise ValueError(
                "QVQConfig: YAQA spectral experiment requires `format=qvq_v2b2_p32` "
                "with YAQA rounding."
            )
        if self.yaqa.sample_strategy != "full" and (
            self.rounding != "yaqa" or self.format != FORMAT.QVQ_V2B2_P32
        ):
            raise ValueError(
                "QVQConfig: sampled YAQA family selection requires `format=qvq_v2b2_p32` with YAQA rounding."
            )
        self.incoherence = str(self.incoherence).strip().lower()
        if not isinstance(self.module_scale_search, bool):
            raise TypeError("QVQConfig: `module_scale_search` must be boolean.")
        if not isinstance(self.output_channel_scale_optimization, bool):
            raise TypeError("QVQConfig: `output_channel_scale_optimization` must be boolean.")
        if not isinstance(self.viterbi_objective, str):
            raise TypeError("QVQConfig: `viterbi_objective` must be a string.")
        self.viterbi_objective = self.viterbi_objective.strip().lower()
        if self.viterbi_objective not in {"euclidean", "hessian_diagonal"}:
            raise ValueError("QVQConfig: `viterbi_objective` must be `euclidean` or `hessian_diagonal`.")
        if (
            isinstance(self.tail_biting_candidates, bool)
            or not isinstance(self.tail_biting_candidates, int)
            or self.tail_biting_candidates < 1
        ):
            raise ValueError("QVQConfig: `tail_biting_candidates` must be a positive integer.")
        if isinstance(self.viterbi_minimum_proxy_improvement, bool) or not isinstance(
            self.viterbi_minimum_proxy_improvement, (int, float)
        ):
            raise TypeError("QVQConfig: `viterbi_minimum_proxy_improvement` must be a real scalar.")
        self.viterbi_minimum_proxy_improvement = float(self.viterbi_minimum_proxy_improvement)
        if not math.isfinite(self.viterbi_minimum_proxy_improvement) or self.viterbi_minimum_proxy_improvement < 0:
            raise ValueError("QVQConfig: `viterbi_minimum_proxy_improvement` must be finite and nonnegative.")
        if self.viterbi_minimum_proxy_improvement > 0 and self.viterbi_objective != "hessian_diagonal":
            raise ValueError("QVQConfig: `viterbi_minimum_proxy_improvement` requires `hessian_diagonal` objective.")
        self.output_alignment = _normalize_qvq_output_alignment_config(self.output_alignment)
        if self.activation is not None and self.output_alignment is not None:
            raise ValueError("QVQConfig: FP8 activation quantization does not yet support output alignment.")
        if self.output_alignment is not None and self.lm_head:
            raise ValueError(
                "QVQ output alignment currently supports decoder layers, not language-model head (`lm_head`) quantization."
            )
        self.module_granular_replay = _normalize_module_granular_replay_config(self.module_granular_replay)
        if self.smooth_swiglu is not None:
            if isinstance(self.smooth_swiglu, dict):
                self.smooth_swiglu = SmoothSwiGLUConfig(**self.smooth_swiglu)
            elif isinstance(self.smooth_swiglu, SmoothSwiGLUConfig):
                self.smooth_swiglu.__post_init__()
            else:
                raise TypeError("QVQConfig: `smooth_swiglu` must be a SmoothSwiGLUConfig, dictionary, or None.")
        if self.module_granular_replay is not None:
            if self.format != FORMAT.QVQ_V2B2_P32 or self.rounding != "yaqa":
                raise ValueError(
                    "QVQConfig: module-granular replay requires `format=qvq_v2b2_p32` with YAQA rounding."
                )
            if self.yaqa.sample_strategy != "full":
                raise ValueError("QVQConfig: module-granular replay requires exact `yaqa.sample_strategy='full'`.")
            if self.yaqa.spectral_refinement or self.yaqa.spectral_push or self.yaqa.spectral_localized:
                raise ValueError(
                    "QVQConfig: module-granular replay enumerates fixed complete bank arms and cannot combine "
                    "with YAQA spectral candidate refinement."
                )
            if self.propagated_bank_selection is True:
                raise ValueError(
                    "QVQConfig: module-granular replay and localized propagated bank selection are mutually exclusive."
                )
            if (
                self.module_granular_replay.strategy == "atomic_swiglu"
                and self.yaqa.v2b2_family_mode != "reselect"
            ):
                raise ValueError(
                    "QVQConfig: atomic_swiglu requires YAQA `v2b2_family_mode='reselect'` "
                    "so canonical candidate zero remains the normal reselect result."
                )
        if self.rounding == "yaqa" and self.output_channel_scale_optimization:
            raise ValueError("QVQConfig: YAQA does not support independent output-channel scale optimization.")
        if self.rounding == "yaqa" and self.module_scale_search:
            raise ValueError("QVQConfig: YAQA does not support input-Hessian-only module-scale search.")
        if self.rounding == "yaqa" and self.viterbi_objective != "euclidean":
            raise ValueError("QVQConfig: YAQA requires `viterbi_objective='euclidean'`.")
        if self.rounding == "yaqa" and self.lm_head:
            raise ValueError("QVQConfig: YAQA does not support quantizing the language-model head.")
        if self.format == FORMAT.QVQ_V2B4_P64 and self.tail_biting_candidates != 1:
            raise ValueError("QVQConfig: `format=qvq_v2b4_p64` initially requires one tail-biting candidate.")
        if self.format == FORMAT.QVQ_V2B2_P32 and self.tail_biting_candidates != 1:
            raise ValueError("QVQConfig: V2B2-P32 formats initially require one tail-biting candidate.")
        canonical_fields = {
            "tile_rows": (self.tile_rows, 16),
            "tile_cols": (self.tile_cols, 16),
            "incoherence": (self.incoherence, "rht"),
        }
        for field_name, (actual, expected) in canonical_fields.items():
            if actual != expected:
                raise ValueError(
                    f"QVQConfig: QVQ integration requires `{field_name}={expected!r}`, got `{actual!r}`."
                )
        if self.vector_size not in (2, 4) or (
            self.vector_size == 4 and self.format not in (FORMAT.QVQ_V4, FORMAT.QVQ_V4_L18)
        ):
            raise ValueError("QVQConfig: `vector_size=4` requires `format=qvq_v4` or `format=qvq_v4_l18`.")
        expected_window = 18 if self.format == FORMAT.QVQ_V4_L18 else 16
        if self.trellis_window != expected_window:
            raise ValueError(
                f"QVQConfig: format `{self.format.value}` requires `trellis_window={expected_window}`."
            )
        if self.vector_size == 4 and self.bits > 4:
            raise ValueError("QVQConfig: `vector_size=4` supports only rates W1 through W4.")
        if isinstance(self.bank_count, bool) or not isinstance(self.bank_count, int) or self.bank_count not in (1, 2, 4):
            raise ValueError("QVQConfig: `bank_count` must be 1, 2, or 4.")
        if self.bank_count == 2 and self.format != FORMAT.QVQ_V2B2_P32:
            raise ValueError("QVQConfig: `bank_count=2` requires a V2B2-P32 format.")
        if self.bank_count == 4 and self.format not in (FORMAT.QVQ_V4, FORMAT.QVQ_V2B4_P64):
            raise ValueError("QVQConfig: `bank_count=4` requires `format=qvq_v4` or `format=qvq_v2b4_p64`.")
        if self.bank_count == 4 and (self.module_scale_search or self.output_channel_scale_optimization):
            raise ValueError("QVQConfig: four-bank selection currently excludes scale-search controls.")
        if self.propagated_bank_selection is not None and not isinstance(self.propagated_bank_selection, bool):
            raise TypeError("QVQConfig: `propagated_bank_selection` must be boolean or None.")
        localized_v2b2_propagation = (
            self.format == FORMAT.QVQ_V2B2_P32
            and self.rounding == "yaqa"
            and self.yaqa.spectral_localized
        )
        if (
            self.format == FORMAT.QVQ_V2B2_P32
            and self.propagated_bank_selection is True
            and not localized_v2b2_propagation
        ):
            raise ValueError(
                "QVQConfig: V2B2-P32 propagation replay requires YAQA localized spectral refinement."
            )
        if self.propagated_bank_selection is True and not localized_v2b2_propagation:
            if self.bank_count != 4:
                raise ValueError("QVQConfig: propagated bank selection requires `bank_count=4`.")
            if self.rounding != "block_ldlq":
                raise ValueError("QVQConfig: propagated bank selection requires `rounding='block_ldlq'`.")
        if self.format == FORMAT.QVQ_V2B4_P64 and self.propagated_bank_selection is True:
            raise ValueError("QVQConfig: V2B4-P64 propagation replay is not enabled in the initial reference slice.")

        if self.tensor_storage is not None:
            if not isinstance(self.tensor_storage, dict):
                raise ValueError("QVQConfig: `tensor_storage` must be a dictionary when provided.")
            allowed_tensors = {"trellis", "SU", "SV", "bias", "bank_ids", "bank_alt_id"}
            for module_name, tensors in self.tensor_storage.items():
                if not isinstance(tensors, dict):
                    raise ValueError(f"QVQConfig: tensor storage for `{module_name}` must be a dictionary.")
                unexpected = set(tensors) - allowed_tensors
                if unexpected:
                    raise ValueError(
                        f"QVQConfig: tensor storage for `{module_name}` has unexpected tensors: {sorted(unexpected)}."
                    )
                has_selector = "bank_ids" in tensors
                has_alt_id = "bank_alt_id" in tensors
                if self.bank_count in (2, 4) and not has_selector:
                    raise ValueError(f"QVQConfig: banked module `{module_name}` is missing `bank_ids` selectors.")
                if self.bank_count == 2 and not has_alt_id:
                    raise ValueError(f"QVQConfig: V2B2-P32 module `{module_name}` is missing `bank_alt_id`.")
                if self.bank_count != 2 and has_alt_id:
                    raise ValueError(f"QVQConfig: only V2B2-P32 module `{module_name}` may contain `bank_alt_id`.")
                if self.bank_count == 1 and has_selector:
                    raise ValueError(f"QVQConfig: canonical module `{module_name}` cannot contain `bank_ids` selectors.")

        self.group_size = -1
        self.sym = True
        self.pack_dtype = torch.int32

    def calculate_bits_per_weight(self):
        banked_v2 = self.format in (FORMAT.QVQ_V2B4_P64, FORMAT.QVQ_V2B2_P32)
        effective_bpw = self.bits + (2 / 64 if banked_v2 else 0)
        description = " including the segmented-bank selector payload" if banked_v2 else ""
        log.info(
            "Estimated Quantization BPW (bits per weight): %s bpw%s",
            effective_bpw,
            description,
        )

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out.pop("desc_act", None)
        out["sym"] = True
        out["codebook"] = self.codebook
        out["trellis_window"] = self.trellis_window
        out["vector_size"] = self.vector_size
        out["bank_count"] = self.bank_count
        out["propagated_bank_selection"] = self.propagated_bank_selection
        out["tile_rows"] = self.tile_rows
        out["tile_cols"] = self.tile_cols
        out["rounding"] = self.rounding
        out["yaqa"] = None if self.yaqa is None else asdict(self.yaqa)
        out["viterbi_pruning"] = asdict(self.viterbi_pruning)
        out["incoherence"] = self.incoherence
        out["module_scale_search"] = self.module_scale_search
        out["output_channel_scale_optimization"] = self.output_channel_scale_optimization
        out["viterbi_objective"] = self.viterbi_objective
        out["tail_biting_candidates"] = self.tail_biting_candidates
        out["viterbi_minimum_proxy_improvement"] = self.viterbi_minimum_proxy_improvement
        out["output_alignment"] = None if self.output_alignment is None else asdict(self.output_alignment)
        out["module_granular_replay"] = (
            None if self.module_granular_replay is None else asdict(self.module_granular_replay)
        )
        out["smooth_swiglu"] = None if self.smooth_swiglu is None else asdict(self.smooth_swiglu)
        out["activation"] = None if self.activation is None else asdict(self.activation)
        out["tensor_storage"] = self.tensor_storage

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {
            "codebook_version": self.codebook,
            "vector_size": self.vector_size,
            "trellis_window": self.trellis_window,
            "bank_count": self.bank_count,
            "dual_v2": self.format == FORMAT.QVQ_DUAL_V2,
            "v2b4_p64": self.format == FORMAT.QVQ_V2B4_P64,
            "v2b2_p32": self.format == FORMAT.QVQ_V2B2_P32,
            "activation": None if self.activation is None else asdict(self.activation),
        }


@dataclass
class RTNConfig(PreProcessorConfig):
    method: METHOD = field(default=METHOD.GPTQ)
    format: FORMAT = field(default=FORMAT.GPTQ)

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.GPTQ,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return RTN_EXPORT_FORMATS

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        super().__post_init__()

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out["sym"] = self.sym
        out[FORMAT_FIELD_CODE] = self.format

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        super()._update_meta_payload(meta_payload)
        meta_payload["weight_only"] = {
            "smooth": _serialize_smooth_method(self.smooth),
        }

    def uses_weight_only_lifecycle(self) -> bool:
        return True


@dataclass
class GGUFConfig(PreProcessorConfig):
    bits: Union[int, str, GGUFBits] = field(default=4, metadata={"choices": [1, 2, 3, 4, 5, 6, 8]})
    format: Optional[str] = field(default=None)
    method: METHOD = field(default=METHOD.GGUF, init=False)
    group_size: int = field(default=-1, init=False, repr=False)
    desc_act: Optional[bool] = field(default=False, init=False, repr=False)
    sym: bool = field(default=True, init=False, repr=False)
    _gguf_bits: GGUFBits = field(init=False, repr=False, compare=False)

    @property
    def runtime_bits(self) -> GGUFBits:
        return self._gguf_bits

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.GGUF,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return (FORMAT.GGUF,)

    def default_desc_act(self) -> bool:
        return False

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.bits, self.format, self._gguf_bits = _normalize_gguf_config_spec(self.bits, self.format)
        return FORMAT.GGUF

    def _normalize_bits_field(self, bits_value, checkpoint_format: FORMAT):
        normalized = _normalize_quant_bits(bits_value, format_value=FORMAT.GGUF)
        return normalized.bits if isinstance(normalized, GGUFBits) else normalized

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        bits_override_present = "bits" in layer_dict
        format_override_present = FORMAT_FIELD_CODE in layer_dict

        if bits_override_present or format_override_present:
            raw_bits = layer_dict.get("bits", self.bits)
            raw_format = layer_dict.get(FORMAT_FIELD_CODE, self.format)
            normalized_bits, normalized_format, normalized_runtime_bits = _normalize_gguf_config_spec(raw_bits, raw_format)

            layer_dict["bits"] = normalized_bits

            bits_implied_format = (
                isinstance(raw_bits, GGUFBits)
                or (isinstance(raw_bits, str) and not raw_bits.strip().isdigit())
            )
            if format_override_present or bits_implied_format:
                layer_dict[FORMAT_FIELD_CODE] = normalized_format

            if quant_bits_width(normalized_runtime_bits) not in valid_bit_widths:
                raise ValueError(
                    f"QuantizeConfig: Layer `{layer_name}` only support quantization of `{valid_bit_widths}` bits."
                )

        if "group_size" in layer_dict and layer_dict["group_size"] != -1 and layer_dict["group_size"] <= 0:
            raise ValueError(_resolve_dynamic_group_size_error())

    def __post_init__(self):
        self._normalize_preprocessor_state()
        # GGUFConfig already normalized preprocessors above; skip the parent hook to
        # avoid running that normalization twice.
        BaseQuantizeConfig.__post_init__(self)
        self._gguf_bits = _gguf_bits_from_components(self.bits, self.format)

    def _update_meta_payload(self, meta_payload: Dict[str, Any]) -> None:
        super()._update_meta_payload(meta_payload)

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format

    def to_dict(self):
        out = super().to_dict()
        out.pop(GROUP_SIZE_FIELD_CODE, None)
        out.pop("desc_act", None)
        out.pop(PACK_DTYPE_FIELD, None)

        meta_payload = out.get(META_FIELD)
        if isinstance(meta_payload, dict):
            for key in (
                "fallback",
                "offload_to_disk",
                "offload_to_disk_path",
                "pack_impl",
                "gc_mode",
                "wait_for_submodule_finalizers",
                "auto_forward_data_parallel",
                "weight_only_quant_threads",
                "dense_vram_strategy",
                "dense_vram_strategy_devices",
                "moe_vram_strategy",
                "moe_vram_strategy_devices",
                "weight_only",
            ):
                meta_payload.pop(key, None)
            if not meta_payload:
                out.pop(META_FIELD, None)

        return out

    def calculate_bits_per_weight(self):
        bits_name = self.runtime_bits.to_string()
        bpw = _GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS.get(bits_name, float(quant_bits_width(self.runtime_bits)))
        log.info(
            f"Estimated Quantization BPW (bits per weight): {bpw} bpw, based on [bits: {self.bits}, format: {self.format}]"
        )

    def uses_weight_only_lifecycle(self) -> bool:
        return True


def clone_weight_only_config_for_module(
    qcfg: Union[RTNConfig, GGUFConfig, FP8Config, BitsAndBytesConfig],
    module_full_name: str,
) -> Optional[Union[RTNConfig, GGUFConfig, FP8Config, BitsAndBytesConfig]]:
    if qcfg.dynamic_get(layer_name=module_full_name) is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    if qcfg.dynamic is not None:
        smooth_override = qcfg.dynamic_get(module_full_name, "smoother", None)
        if smooth_override is None:
            smooth_override = qcfg.dynamic_get(module_full_name, "smooth", None)
        if smooth_override is not None:
            qcfg_clone.smoother = _normalize_smoother_config(smooth_override)
            qcfg_clone.smooth = qcfg_clone.resolve_smooth_method()

        if isinstance(qcfg_clone, GGUFConfig):
            dynamic_bits = qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits)
            dynamic_format = qcfg.dynamic_get(module_full_name, FORMAT_FIELD_CODE, qcfg_clone.format)
            qcfg_clone.bits, qcfg_clone.format, qcfg_clone._gguf_bits = _normalize_gguf_config_spec(
                dynamic_bits,
                dynamic_format,
            )
        elif isinstance(qcfg_clone, FP8Config):
            dynamic_format = qcfg.dynamic_get(module_full_name, FORMAT_FIELD_CODE, None)
            if dynamic_format is None:
                dynamic_format = qcfg.dynamic_get(module_full_name, "fmt", qcfg_clone.format)
            dynamic_block_size = qcfg.dynamic_get(
                module_full_name,
                "weight_block_size",
                qcfg_clone.weight_block_size,
            )
            block_size = _normalize_fp8_weight_block_size(dynamic_block_size)
            qcfg_clone.format = _normalize_fp8_fmt(dynamic_format)
            qcfg_clone.weight_scale_method = _normalize_fp8_weight_scale_method(
                qcfg.dynamic_get(
                    module_full_name,
                    "weight_scale_method",
                    qcfg_clone.weight_scale_method,
                ),
                weight_block_size=block_size,
            )
            qcfg_clone.weight_block_size = list(block_size) if block_size is not None else None
            qcfg_clone.weight_scale_semantics = _normalize_fp8_scale_semantics(
                qcfg.dynamic_get(
                    module_full_name,
                    "weight_scale_semantics",
                    qcfg_clone.weight_scale_semantics,
                )
            )
        elif isinstance(qcfg_clone, BitsAndBytesConfig):
            qcfg_clone.bits = _normalize_quant_bits(
                qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits),
                format_value=FORMAT.BITSANDBYTES,
            )
            qcfg_clone.format = _normalize_bitsandbytes_format(
                qcfg.dynamic_get(
                    module_full_name,
                    FORMAT_FIELD_CODE,
                    qcfg.dynamic_get(
                        module_full_name,
                        "bnb_quant_type",
                        qcfg_clone.format,
                    ),
                ),
                bits=int(qcfg_clone.bits),
            )
            qcfg_clone.block_size = _normalize_bitsandbytes_block_size(
                qcfg.dynamic_get(
                    module_full_name,
                    "block_size",
                    qcfg.dynamic_get(
                        module_full_name,
                        "bnb_block_size",
                        qcfg_clone.block_size,
                    ),
                )
            )
            qcfg_clone.compress_statistics = bool(
                qcfg.dynamic_get(
                    module_full_name,
                    "compress_statistics",
                    qcfg.dynamic_get(
                        module_full_name,
                        "bnb_compress_statistics",
                        qcfg_clone.compress_statistics,
                    ),
                )
            )
        else:
            qcfg_clone.bits = _normalize_quant_bits(
                qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits),
                format_value=resolve_quant_format(qcfg_clone.format, qcfg_clone.method),
            )

        if isinstance(qcfg_clone, RTNConfig):
            qcfg_clone.sym = qcfg.dynamic_get(module_full_name, "sym", qcfg_clone.sym)
            qcfg_clone.group_size = qcfg.dynamic_get(module_full_name, "group_size", qcfg_clone.group_size)

            desc_act_override = qcfg.dynamic_get(module_full_name, "desc_act", None)
            if desc_act_override is not None:
                qcfg_clone.desc_act = desc_act_override

    return qcfg_clone


clone_rtn_config_for_module = clone_weight_only_config_for_module


@dataclass
class MXFP4Config(PreProcessorConfig):
    bits: int = field(default=4, metadata={"choices": [4]})
    method: METHOD = field(default=METHOD.MXFP4)
    format: FORMAT = field(default=FORMAT.MXFP4)
    group_size: int = field(default=-1)
    desc_act: Optional[bool] = field(default=False)
    sym: bool = field(default=True)

    def _resolve_checkpoint_format(self) -> FORMAT:
        self.format = _normalize_format(self.format)
        if self.format != FORMAT.MXFP4:
            raise ValueError(f"MXFP4Config: `format` must be `{FORMAT.MXFP4}`.")
        return FORMAT.MXFP4

    def allowed_quant_methods(self) -> Tuple[METHOD, ...]:
        return (METHOD.MXFP4,)

    def supported_export_formats(self) -> Tuple[FORMAT, ...]:
        return (FORMAT.MXFP4,)

    def default_desc_act(self) -> bool:
        return False

    def __post_init__(self):
        self._normalize_preprocessor_state()
        super().__post_init__()

        if self.bits != 4:
            raise ValueError("MXFP4Config: `bits` must be `4`.")
        if self.method != METHOD.MXFP4:
            raise ValueError("MXFP4Config: `method` must be `mxfp4`.")

        self.group_size = -1
        self.desc_act = False
        self.sym = True
        self.format = _normalize_format(self.format)

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith("-")},
                **{k: v for k, v in self.dynamic.items() if not k.startswith("-")},
            }
            for layer, layer_dict in self.dynamic.items():
                self._normalize_dynamic_layer_config(
                    layer,
                    layer_dict,
                    valid_bit_widths=[4],
                    checkpoint_format=FORMAT.MXFP4,
                )

    def _normalize_dynamic_layer_config(
        self,
        layer_name: str,
        layer_dict: Dict[str, Any],
        *,
        valid_bit_widths: List[int],
        checkpoint_format: FORMAT,
    ) -> None:
        if not isinstance(layer_dict, dict):
            return

        del valid_bit_widths, checkpoint_format
        if "bits" in layer_dict and int(layer_dict["bits"]) != 4:
            raise ValueError(f"MXFP4Config: layer `{layer_name}` only supports 4-bit MXFP4 weights.")
        if "group_size" in layer_dict and layer_dict["group_size"] not in (-1, None):
            raise ValueError("MXFP4Config: `group_size` is not used; keep it at `-1`.")
        if "desc_act" in layer_dict and bool(layer_dict["desc_act"]):
            raise ValueError("MXFP4Config: `desc_act` is not supported.")
        if "sym" in layer_dict and layer_dict["sym"] is not True:
            raise ValueError("MXFP4Config: `sym` must stay `True`.")
        raw_format = layer_dict.get(FORMAT_FIELD_CODE, layer_dict.get("fmt"))
        if raw_format is not None:
            layer_dict[FORMAT_FIELD_CODE] = _normalize_format(raw_format)
        layer_dict.pop("fmt", None)

    def quant_linear_init_kwargs(self) -> Dict[str, Any]:
        return {}

    def _update_output_payload(self, out: Dict[str, Any]) -> None:
        out[FORMAT_FIELD_CODE] = self.format.value

    def uses_weight_only_lifecycle(self) -> bool:
        return True


def _resolve_quantize_config_class(payload: Dict[str, Any]) -> type[BaseQuantizeConfig]:
    method = payload.get(METHOD_FIELD_CODE, payload.get(QUANT_METHOD_FIELD, METHOD.GPTQ))
    raw_format_value = payload.get(FORMAT_FIELD_CODE, payload.get(FORMAT_FIELD_CHECKPOINT, FORMAT.GPTQ))
    weight_only = payload.get("weight_only")
    bits = payload.get(BITS_FIELD_CODE)
    gguf_public_format = payload.get(FORMAT_FIELD_CODE)

    try:
        method = _normalize_quant_method(method)
    except Exception:
        method = METHOD.GPTQ

    if _looks_like_fp8_fmt(raw_format_value):
        format_value = FORMAT.FP8
    else:
        try:
            format_value = _normalize_format(raw_format_value)
        except Exception:
            try:
                gguf_public_format = _normalize_gguf_public_format(raw_format_value)
            except ValueError:
                gguf_public_format = payload.get(FORMAT_FIELD_CODE)
            format_value = FORMAT.GPTQ

    gguf_format_detected = False
    if gguf_public_format is not None:
        try:
            gguf_format_detected = _normalize_gguf_public_format(gguf_public_format) is not None
        except ValueError:
            gguf_format_detected = False

    weight_only_method = _peek_weight_only_method(weight_only)
    fp8_storage_fmt = payload.get(FORMAT_FIELD_CODE, payload.get("fmt"))
    if weight_only is not None and weight_only_method not in {
        None,
        WeightOnlyMethod.RTN,
        WeightOnlyMethod.GGUF,
        WeightOnlyMethod.FP8,
        WeightOnlyMethod.BITSANDBYTES,
    }:
        raise ValueError(
            "QuantizeConfig: unsupported weight-only config. Weight-only export currently supports "
            "`rtn`, `gguf`, `fp8`, and `bitsandbytes`."
        )
    if (
        format_value == FORMAT.GGUF
        or weight_only_method == WeightOnlyMethod.GGUF
        or _looks_like_gguf_bits(bits)
        or gguf_format_detected
    ):
        return GGUFConfig
    if weight_only_method == WeightOnlyMethod.FP8:
        return FP8Config
    if weight_only_method == WeightOnlyMethod.BITSANDBYTES:
        return BitsAndBytesConfig
    if weight_only_method == WeightOnlyMethod.RTN:
        return RTNConfig
    if weight_only is not None:
        return RTNConfig
    if method == METHOD.FP8 or format_value == FORMAT.FP8 or _looks_like_fp8_fmt(fp8_storage_fmt):
        return FP8Config
    if (
        method == METHOD.BITSANDBYTES
        or format_value == FORMAT.BITSANDBYTES
        or _looks_like_bitsandbytes_format(raw_format_value)
    ):
        return BitsAndBytesConfig
    if method == METHOD.EXL3 or format_value == FORMAT.EXL3:
        return EXL3Config
    if method == METHOD.QVQ or format_value in (
        FORMAT.QVQ,
        FORMAT.QVQ_V4,
        FORMAT.QVQ_V4_L18,
        FORMAT.QVQ_DUAL_V2,
        FORMAT.QVQ_V2B4_P64,
        FORMAT.QVQ_V2B2_P32,
    ):
        return QVQConfig
    if method == METHOD.PARO or format_value == FORMAT.PAROQUANT:
        return ParoConfig
    if method == METHOD.QQQ or format_value == FORMAT.QQQ:
        return QQQConfig
    if method == METHOD.AWQ:
        return AWQConfig
    if format_value in {FORMAT.GEMM, FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ}:
        return AWQConfig
    if format_value == FORMAT.MARLIN:
        return AWQConfig if method == METHOD.AWQ else GPTQConfig
    if method == METHOD.MXFP4 or format_value == FORMAT.MXFP4:
        return MXFP4Config
    return GPTQConfig


def _known_quantize_config_field_names() -> set[str]:
    field_names: set[str] = set()
    for cls in (
        BaseQuantizeConfig,
        PreProcessorConfig,
        QuantizeConfig,
        GPTQConfig,
        AWQConfig,
        ParoConfig,
        QQQConfig,
        FP8Config,
        BitsAndBytesConfig,
        EXL3Config,
        QVQConfig,
        RTNConfig,
        GGUFConfig,
        MXFP4Config,
    ):
        field_names.update(field.name for field in fields(cls))
    return field_names
