# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from device_smi import Device
from torch import Tensor
from torch.nn import Module

from .. import DEVICE_THREAD_POOL
from ..looper.input_cache import InputCache
from ..looper.named_module import NamedModule
from ..models.base import (
    MODULE_TREE_FLAG_DOWN,
    MODULE_TREE_FLAG_ROUTED,
    BaseQModel,
    module_tree_flags_are_expert,
    module_tree_flags_are_moe,
)
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..quantization.config import ExpertsRoutingBypass, QuantizeConfig
from ..utils.attn_mask import attention_mask_sequence_lengths, input_id_sequence_lengths
from ..utils.colors import ANSIColor, color_text
from ..utils.logger import setup_logger
from ..utils.random_str import get_random_string
from ..utils.torch import CPU, DEVICE_0, DEVICE_1, HAS_NPU


log = setup_logger()

_PROJECT_LOG_DIR = Path("logs")

MODULE_FEATURE_COLUMN = "feat: in, out"
DTYPE_SIZE_COLUMN = "dtype: size"

DEFAULT_LOG_COLUMNS: List[str] = [
    PROCESS_LOG_NAME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    MODULE_FEATURE_COLUMN,
    DTYPE_SIZE_COLUMN,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
    QUANT_LOG_DAMP,
    PROCESS_LOG_TIME,
    PROCESS_LOG_FWD_TIME,
    PROCESS_USED_MEMORY,
    "dynamic",
]


class _SafeDict(dict):
    """Thread-safe dict subclass that protects reads, writes, and iteration.

    ``LoopProcessor`` exposes its ``tasks`` mapping to worker threads (e.g.
    ``pre_process_fwd_hook`` closures running inside ``DEVICE_THREAD_POOL``).
    This wrapper lets us keep the normal ``self.tasks[name]`` / ``for task in
    self.tasks.values()`` API while guaranteeing the dict is never in an
    inconsistent state under free-threading ``GIL=0``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def __iter__(self):
        with self._lock:
            return iter(list(super().__iter__()))

    def __len__(self):
        with self._lock:
            return super().__len__()

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def setdefault(self, key, default=None):
        with self._lock:
            return super().setdefault(key, default)

    def pop(self, key, *args):
        with self._lock:
            return super().pop(key, *args)

    def popitem(self):
        with self._lock:
            return super().popitem()

    def clear(self):
        with self._lock:
            super().clear()

    def update(self, *args, **kwargs):
        with self._lock:
            super().update(*args, **kwargs)

    def keys(self):
        with self._lock:
            return list(super().keys())

    def values(self):
        with self._lock:
            return list(super().values())

    def items(self):
        with self._lock:
            return list(super().items())


class _ThreadSafeInputCache:
    """Thread-safe proxy around an ``InputCache`` instance.

    ``LoopProcessor.inputs_cache`` is read and written by the main
    orchestration thread and by worker threads during stage hand-offs. This
    proxy makes every attribute access a short critical section while still
    allowing the rest of the code to use normal attribute syntax.
    """

    def __init__(self, cache: InputCache):
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_lock", threading.RLock())

    def set_cache(self, cache: InputCache) -> None:
        """Replace the wrapped cache with ``cache`` (unwraps another proxy)."""
        if isinstance(cache, _ThreadSafeInputCache):
            cache = cache.unwrap()
        with self._lock:
            object.__setattr__(self, "_cache", cache)

    def unwrap(self) -> InputCache:
        """Return the underlying ``InputCache`` instance."""
        with self._lock:
            return object.__getattribute__(self, "_cache")

    def __getattr__(self, name: str):
        with self._lock:
            return getattr(object.__getattribute__(self, "_cache"), name)

    def __setattr__(self, name: str, value):
        if name in ("_cache", "_lock"):
            object.__setattr__(self, name, value)
        else:
            with self._lock:
                setattr(
                    object.__getattribute__(self, "_cache"),
                    name,
                    value,
                )

    def __delattr__(self, name: str):
        with self._lock:
            delattr(object.__getattribute__(self, "_cache"), name)


@dataclass
class ExecutionConfig:
    """Describe how a processor participates in forward replay and activation capture.

    Processor defaults:

    +--------------+------+------------------------+---------------------------+----------------+--------------------+
    | Processor     | fwd? | replay_after_process? | single_pass_all_modules?  | early_stop?    | activation_capture?|
    +--------------+------+------------------------+---------------------------+----------------+--------------------+
    | GPTQ          | yes  | yes                    | no                        | yes            | no                 |
    | AWQ           | yes  | yes                    | no                        | yes            | yes                |
    | ParoQuant     | yes  | yes                    | no                        | yes            | yes                |
    | Native        | yes  | no                     | yes                       | no             | no                 |
    | WeightOnly    | no   | no                     | no/default                | no/default     | no                 |
    +--------------+------+------------------------+---------------------------+----------------+--------------------+
    """

    # Whether the processor needs forward replay at all.
    require_fwd: bool = True
    # Whether the layer outputs become authoritative only after a replay that
    # runs after process(), rather than from the subset forward itself.
    fwd_replay_after_process: bool = True
    # Whether layer modules are replayed in one combined pass instead of per subset.
    fwd_all_modules_in_single_pass: bool = False
    # Whether a subset forward can stop as soon as the last subset hook fires.
    subset_forward_early_stop: bool = False
    # Whether capture-only modules/hooks (for example ':?') should be enabled.
    enable_activation_capture: bool = False
    # Whether the processor needs the original layer IO from the pre-process
    # forward, even when the authoritative next-layer cache is produced by a
    # later replay-after-process step.
    capture_layer_forward_context: bool = False


# LoopProcessor is a singleton(), not per module instance
class LoopProcessor:
    """Base lifecycle coordinator shared by all quantization processors."""

    def __init__(
            self,
            tokenizer, qcfg: QuantizeConfig,
            calibration,
            prepare_dataset_func: Optional[Callable] = None,
            calibration_concat_size: Optional[int] = None,
            calibration_sort: Optional[str] = None,
            calibration_concat_separator: Optional[str] = None,
            batch_size: int = 1,
            execution_config: Optional[ExecutionConfig] = None,
    ):
        """Initializes shared processor state, logging, and calibration bookkeeping."""

        # process level lock
        self.lock = threading.Lock()

        # result is total collection of all module results mapped by module.full_name
        self._results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._fwd_time_lock = threading.Lock()
        self._device_smi_lock = threading.RLock()
        self._input_cache_lock = threading.RLock()

        # Locks for the remaining shared state that workers touch concurrently.
        self._pb_lock = threading.Lock()
        self._fwd_time_lock = threading.Lock()
        self._device_smi_lock = threading.RLock()
        self._cache_lock = threading.RLock()

        self.tokenizer = tokenizer
        self.qcfg = qcfg
        self.qcfg_dynamic = None # cloned and dynamic filtered

        # Keep lifecycle and replay policy in one object so the stages consume
        # one execution mode instead of a scattered set of booleans.
        self.execution_config = execution_config or ExecutionConfig()

        self.inputs_cache = _ThreadSafeInputCache(InputCache([], [], [], []))
        self.tasks = _SafeDict()

        self.pb = None
        self.fwd_time = None
        self.layer_count = None

        # Processor sub-classes that are data-independent must not participate in
        # calibration sample-count validation.
        self.is_weight_only = False

        # Global reference sample count taken from the first calibrated module.
        # Every later calibrated module must match it, with MoE experts being the
        # only allowed exception (they may see fewer tokens than dense modules
        # unless moe.routing is set to bypass).
        self._global_reference_nsamples: Optional[int] = None
        self._global_sample_count_lock = threading.Lock()


        self.gpu_memorys = []
        self.cpu_memorys = []
        self.durations = []
        self.module_names = []

        # logging
        self.log = []
        self.log_call_count = 0
        self._log_column_labels: List[str] = []
        self._log_columns = None
        self._log_header_interval = 20
        current_time = datetime.now().strftime("%m_%d_%Y_%Hh_%Mm_%Ss")
        self.log_tmp_log_file_name = str(self._build_log_file_path(self.name(), current_time))
        self._device_smi_handles = self._init_device_smi_handles()
        self._cpu_device_smi = self._init_cpu_device_handle()
        self._device_metric_failures: Set[str] = set()


        # prepare dataset
        if calibration is not None:
            if len(calibration) == 0:
                raise ValueError("Calibration dataset must not be empty.")

            min_calibration_dataset_size = 256
            min_calibration_dataset_input_ids_avg_length = 256
            if len(calibration) < min_calibration_dataset_size:
                log.warn(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
                               f"Current: {len(calibration)}.")

            if prepare_dataset_func is None:
                raise ValueError("prepare_dataset_func must be provided when calibration data is supplied.")

            calibration = prepare_dataset_func(
                calibration_dataset=calibration,
                calibration_dataset_concat_size=calibration_concat_size,
                calibration_dataset_sort=calibration_sort,
                batch_size=batch_size,
                calibration_concat_separator=calibration_concat_separator,
            )

            # Calculate the average length of the average input_ids
            total_input_ids_length = 0
            max_input_id_length = 0
            for row in calibration:
                input_ids = row["input_ids"]
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() <= 2:
                        input_ids_length = input_ids.shape[-1]
                    else:
                        raise ValueError(
                            "Expected a 1-dimensional tensor or 2-dimensional tensor for 'input_ids', but got a tensor with {0} dimensions.".format(
                                input_ids.dim()))
                else:
                    input_ids_length = len(input_ids)

                if input_ids_length > max_input_id_length:
                    max_input_id_length = input_ids_length
                total_input_ids_length += input_ids_length
            avg = total_input_ids_length / len(calibration)

            if avg < min_calibration_dataset_input_ids_avg_length:
                log.warn(f"The average length of input_ids of calibration_dataset should be greater than "
                               f"{min_calibration_dataset_input_ids_avg_length}: actual avg: {avg}.")

            self.num_batches = len(calibration)
            self.calibration_dataset = calibration
        else:
            self.num_batches = 0
            self.calibration_dataset = []

        self.total_calibration_tokens = self._compute_total_tokens(self.calibration_dataset)
        # Upper bound on the number of token positions that can ever be observed,
        # including padding. Modules that receive flattened (2-D) activations from
        # a model's MoE forward may legitimately count padded positions, so they
        # are allowed to be as high as this value.
        self._global_max_padded_nsamples: int = self._compute_max_padded_tokens(self.calibration_dataset)
        # Model-derived upper bound on the number of token positions actually
        # observed by calibration modules; starts from the dataset bounds and is
        # increased when a non-MoE module reports a larger count (e.g. multimodal
        # inputs where visual placeholders expand into many token embeddings).
        self._global_observed_max_nsamples: int = max(
            self.total_calibration_tokens, self._global_max_padded_nsamples
        )

        # Track the current calibration batch index on a per-thread basis so
        # processors can retrieve deterministic ordering information (e.g.
        # GPTQ's Hessian updates) even when forwards run on multiple threads.
        self._batch_tls = threading.local()

    # ------------------------------------------------------------------
    # Generic module-tree / MoE helpers used by all processors
    # ------------------------------------------------------------------
    def _is_bypass_moe_routing(self) -> bool:
        """Return True when the active config forwards every calibration token to all experts."""

        moe = getattr(self.qcfg, "moe", None)
        return moe is not None and isinstance(getattr(moe, "routing", None), ExpertsRoutingBypass)

    def _module_tree_flags(self, task: Optional[Any] = None, name: Optional[str] = None) -> frozenset:
        """Return the module-tree flags stored on the wrapped module, if any.

        Accepts either a task object that carries ``_named_module`` or a
        module ``name`` that can be resolved from ``self.tasks``.
        """

        if task is not None:
            named_module = getattr(task, "_named_module", None)
            if isinstance(named_module, NamedModule):
                return named_module.state.get("module_tree_flags", frozenset())
        if name is not None:
            task = getattr(self, "tasks", {}).get(name)
            if task is not None:
                return self._module_tree_flags(task)
        return frozenset()

    def _module_expert_isolation_key(self, name: str) -> Optional[str]:
        """Return the parsed routed-expert identity for ``name``."""

        flags = self._module_tree_flags(name=name)
        if MODULE_TREE_FLAG_ROUTED not in flags:
            return None
        task = getattr(self, "tasks", {}).get(name)
        named_module = getattr(task, "_named_module", None)
        if not isinstance(named_module, NamedModule):
            return None
        return named_module.state.get("module_tree_expert_group")

    def _module_is_expert_down_proj(self, name: str) -> bool:
        """Return True if ``name`` is the down projection of an expert."""

        flags = self._module_tree_flags(name=name)
        return MODULE_TREE_FLAG_DOWN in flags and module_tree_flags_are_expert(flags)

    def _module_is_moe_related(self, name: str) -> bool:
        """Return True if ``name`` belongs to an MoE block (routed/shared expert or gate).

        Modules inside routed expert lists, shared expert lists, or the MoE routing
        gate are treated as MoE-related because they may receive flattened
        activations that include padded token positions.

        The parsed module-tree flags stored on ``NamedModule.state`` are the
        only source of structural MoE membership.
        """

        task = getattr(self, "tasks", {}).get(name)
        if isinstance(getattr(task, "_named_module", None), NamedModule):
            flags = task._named_module.state.get("module_tree_flags", frozenset())
            return module_tree_flags_are_moe(flags)
        return False

    def _module_is_embedding(self, name: str) -> bool:
        """Return True if ``name`` is the model's input embedding module."""

        if not hasattr(self, "_input_embeddings_name"):
            gptq_model = getattr(self, "gptq_model", None)
            input_name = None
            if gptq_model is not None:
                # ``get_input_embeddings_name`` can recurse on uninitialized
                # ``nn.Module`` objects (e.g. test fixtures built with
                # ``object.__new__``), so only call it on a properly constructed
                # model instance.
                if "_modules" in getattr(gptq_model, "__dict__", {}):
                    input_name = getattr(gptq_model, "get_input_embeddings_name", lambda: None)()
            # Only cache a resolved name; a lazy/wrapped model that is not yet
            # initialized should be re-checked on the next call.
            if input_name is not None:
                self._input_embeddings_name = input_name
            else:
                return False
        return name == self._input_embeddings_name

    def _assert_calibration_sample_count(self, name: str, nsamples: int) -> None:
        """Enforce that calibration sample counts stay within allowed bounds.

        The first observed non-zero count is recorded as a diagnostic reference.
        Different module types may legitimately observe different counts:

        * Dense modules are normally keep-mask filtered (``total_calibration_tokens``)
          or flattened (``_global_max_padded_nsamples``), and may be unactivated (0).
        * Bypass routing forces every targeted module to see all valid tokens
          (``total_calibration_tokens``).
        * Input embeddings count all token positions including padding.
        * Native-routed MoE experts see a variable subset bounded by the padded total.

        Weight-only processors skip this validation because they never run a
        calibration forward pass.
        """

        if self.is_weight_only:
            return

        if not isinstance(nsamples, int):
            nsamples = int(nsamples)

        if nsamples < 0:
            raise AssertionError(
                f"Module `{name}` reported a negative sample count ({nsamples})."
            )

        is_moe = self._module_is_moe_related(name)
        is_bypass = self._is_bypass_moe_routing()
        is_embedding = self._module_is_embedding(name)

        total_unpadded = getattr(self, "total_calibration_tokens", 0)
        max_padded = getattr(self, "_global_max_padded_nsamples", 0)
        observed_max = max(
            getattr(self, "_global_observed_max_nsamples", 0),
            total_unpadded,
            max_padded,
        )

        with self._global_sample_count_lock:
            ref = self._global_reference_nsamples
            if ref is None and nsamples > 0:
                self._global_reference_nsamples = nsamples
                ref = nsamples

            if is_embedding:
                expected = max_padded if max_padded > 0 else ref
                allowed = {0, expected} if expected is not None else {0}
            elif is_bypass:
                # Bypass targets should all see the same token count; an
                # unactivated expert (0) or an expanded-position count are also
                # legitimate because the all-experts replay can fall back to the
                # model's native routed forward.
                expected = total_unpadded if total_unpadded > 0 else ref
                allowed = {0}
                if expected is not None:
                    allowed.add(expected)
                if observed_max > 0 and observed_max not in allowed:
                    allowed.add(observed_max)
            elif not is_moe:
                # Dense modules may be unactivated, keep-mask filtered, flattened,
                # or expanded by the model (multimodal visual tokens).
                expected = total_unpadded if total_unpadded > 0 else ref
                allowed = {0}
                if expected is not None:
                    allowed.add(expected)
                if max_padded > 0 and max_padded != expected:
                    allowed.add(max_padded)
                if observed_max > 0 and observed_max not in allowed:
                    allowed.add(observed_max)
            else:
                # Native-routed MoE: any subset up to the observed position count.
                if observed_max > 0 and nsamples > observed_max:
                    raise AssertionError(
                        f"MoE module `{name}` received {nsamples} samples, "
                        f"exceeding the observed token bound {observed_max}."
                    )
                return

            if nsamples not in allowed:
                # A non-embedding module reporting more positions than the dataset
                # bound has discovered an expanded sequence (e.g. multimodal visual
                # tokens). Only treat it as a new maximum when a bound is known.
                if not is_embedding and observed_max > 0 and nsamples > observed_max:
                    self._global_observed_max_nsamples = nsamples
                    return

                raise AssertionError(
                    f"Module `{name}` sample count {nsamples} is outside the allowed "
                    f"set {sorted(allowed)} for this module type."
                )

    @staticmethod
    def _build_log_file_path(processor_name: str, current_time: str) -> Path:
        """Place processor temp log files under the project-local logs directory."""

        _PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        return _PROJECT_LOG_DIR / f"{processor_name}_log_{get_random_string()}_time_{current_time}.log"

    def draw_progress(self, title: str, subtitle: str = "") -> None:
        """Best-effort progress-bar redraw for processors with an attached progress handle."""

        with self._pb_lock:
            if self.pb is None:
                return
            self.pb.title(title).subtitle(subtitle).draw()

    @staticmethod
    def _compute_total_tokens(calibration_dataset) -> int:
        """Counts total calibration tokens using masks when available."""

        if not calibration_dataset:
            return 0
        total = 0
        for row in calibration_dataset:
            if not isinstance(row, dict):
                continue
            mask = row.get("attention_mask")
            if mask is not None:
                input_lengths = input_id_sequence_lengths(row.get("input_ids"))
                seq_len = input_lengths[0] if input_lengths and len(set(input_lengths)) == 1 else None
                sequence_lengths = input_lengths if input_lengths and seq_len is None else None
                batch_size = len(input_lengths) if input_lengths else None
                total += sum(
                    attention_mask_sequence_lengths(
                        mask,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        sequence_lengths=sequence_lengths,
                    )
                )
                continue
            total += sum(input_id_sequence_lengths(row.get("input_ids")))
        return total

    @staticmethod
    def _compute_max_padded_tokens(calibration_dataset) -> int:
        """Count every calibration token position, including padding."""

        if not calibration_dataset:
            return 0
        total = 0
        for row in calibration_dataset:
            if not isinstance(row, dict):
                continue
            input_ids = row.get("input_ids")
            if input_ids is not None:
                total += sum(input_id_sequence_lengths(input_ids))
                continue
            attention_mask = row.get("attention_mask")
            if attention_mask is not None:
                mask = torch.as_tensor(attention_mask)
                if mask.ndim == 0:
                    raise ValueError("attention_mask must have at least one dimension.")
                total += int(mask.shape[-1]) * (int(mask.shape[0]) if mask.ndim > 1 else 1)
        return total

    def _set_current_batch_index(self, batch_index: Optional[int]) -> None:
        """Stores the active calibration batch index in thread-local state."""

        if batch_index is None:
            if hasattr(self._batch_tls, "index"):
                delattr(self._batch_tls, "index")
        else:
            self._batch_tls.index = int(batch_index)

    def current_batch_index(self) -> Optional[int]:
        """Returns the thread-local calibration batch index, if one is set."""

        return getattr(self._batch_tls, "index", None)

    def _async_log_writer(self, stat):
        """Appends one serialized log record to the processor temp log file."""

        with open(self.log_tmp_log_file_name, 'a') as f:
            json.dump(stat, f, indent=4)
            f.write("\n")

    def log_save_async(self, stat):
        """Schedules asynchronous log persistence on the serial CPU worker."""

        # Serialize writes on the CPU-bound worker to avoid interleaved JSON output.
        DEVICE_THREAD_POOL.submit_serial(CPU, self._async_log_writer, stat)

    def log_new_row(self, stat):
        """Prints a formatted log row and queues it for async persistence."""

        with self.lock:
            self.log_call_count += 1
            columns_rebuilt = self._ensure_log_columns(stat)

            if self._log_columns is None:
                return

            if columns_rebuilt or self.log_call_count % self._log_header_interval == 1:
                self._log_columns.info.header()

            row_values = [
                self._format_log_value(column, stat.get(column, ""), stat)
                for column in self._log_column_labels
            ]
            self._log_columns.info(*row_values)

            # Emit a plain-text summary when debugging quantization quality in test runs.
            if os.getenv("GPTQMODEL_LOG_QUANT_STATS", "0") not in ("", "0", "false", "False"):
                log.info(
                    "Quant stat: method=%s layer=%s module=%s samples=%s loss=%s time=%s",
                    stat.get(PROCESS_LOG_NAME, ""),
                    stat.get(PROCESS_LOG_LAYER, ""),
                    stat.get(PROCESS_LOG_MODULE, ""),
                    stat.get(QUANT_LOG_NSAMPLES, ""),
                    stat.get(QUANT_LOG_LOSS, ""),
                    stat.get(PROCESS_LOG_TIME, ""),
                )

        self.log_save_async(stat)

    def loss_color(self, loss_value: float) -> ANSIColor:
        """Maps a quantization loss value to a terminal highlight color."""

        if loss_value <= 0.1:
            return ANSIColor.GREEN
        elif loss_value <= 1:
            return ANSIColor.CYAN
        elif loss_value <= 5:
            return ANSIColor.YELLOW
        elif loss_value <= 20:
            return ANSIColor.ORANGE
        else:
            return ANSIColor.BRIGHT_RED

    def _ensure_log_columns(self, stat: Dict[str, Any]) -> bool:
        """Expands the CLI log table to include any new stat keys."""

        desired_labels = list(DEFAULT_LOG_COLUMNS)
        for key in stat.keys():
            if key not in desired_labels:
                desired_labels.append(key)

        if self._log_columns is not None and desired_labels == self._log_column_labels:
            return False

        self._log_column_labels = desired_labels
        column_specs = [{"label": label, "width": "fit"} for label in self._log_column_labels]
        self._log_columns = log.columns(cols=column_specs, padding=1)
        # simulate some data to snap columns on first print
        self._log_columns.info.simulate(
            "gptq", "13", "mlp.experts.110.down_proj", "(4096, 11008)",
            "int8: 11.5MB", "0.0000158448",
            "36616", "0.05000", "0.841", "7.984", "cuda:0=9.1GB","")
        return True

    def _format_log_value(self, key: str, value: Any, stat: Dict[str, Any]) -> str:
        """Formats one log cell, applying colors to loss and sample counts."""

        text = "" if value is None else str(value)

        if key == QUANT_LOG_LOSS and text:
            cleaned = text.strip().lower()
            try:
                color_code = self.loss_color(float(text))
            except (TypeError, ValueError):
                if cleaned.endswith("fallback") or cleaned.startswith("fallback("):
                    return color_text(text, ANSIColor.ORANGE)
                return text
            return color_text(text, color_code)

        if key == QUANT_LOG_NSAMPLES and text:
            try:
                samples_value = float(text)
            except (TypeError, ValueError):
                return text

            color_code = self._samples_color(samples_value, stat)
            if color_code is not None:
                return color_text(text, color_code)

        return text

    def _samples_color(self, samples_value: float, stat: Dict[str, Any]) -> Optional[ANSIColor]:
        """Colors sample counts relative to method-specific adequacy thresholds."""

        quant_method = str(stat.get(PROCESS_LOG_NAME, "")).lower()

        divisor = 10.0 if quant_method.startswith("awq") else 1.0

        threshold_green = 4096 / divisor
        threshold_dark_green = 1024 / divisor
        threshold_orange = 756 / divisor
        threshold_red = 512 / divisor

        if samples_value >= threshold_green:
            return ANSIColor.BRIGHT_GREEN
        if samples_value >= threshold_dark_green:
            return ANSIColor.GREEN
        if samples_value >= threshold_orange:
            return ANSIColor.ORANGE
        if samples_value < threshold_red:
            return ANSIColor.BRIGHT_RED
        return ANSIColor.RED

    def module_feature_summary(self, module: NamedModule) -> str:
        """Formats cached input/output feature sizes for log display."""

        in_features = module.state.get("in_features")
        out_features = module.state.get("out_features")

        if isinstance(in_features, int) and isinstance(out_features, int):
            return f"{in_features}, {out_features}"
        return ""

    def module_dtype_size_summary(self, module: NamedModule) -> str:
        """Formats dtype and total persistent tensor footprint for a module."""

        weight = getattr(module.module, "weight", None)
        dtype = getattr(weight, "dtype", None)
        total_bytes = 0

        if isinstance(weight, torch.Tensor):
            total_bytes += weight.numel() * weight.element_size()
        else:
            dtype = dtype or getattr(module, "module_dtype", None)
            in_features = module.state.get("in_features")
            out_features = module.state.get("out_features")
            if dtype is not None and isinstance(in_features, int) and isinstance(out_features, int):
                element_size = torch.empty((), dtype=dtype).element_size()
                total_bytes += in_features * out_features * element_size

        bias = getattr(module.module, "bias", None)
        if isinstance(bias, torch.Tensor):
            total_bytes += bias.numel() * bias.element_size()

        # account for persistent tensors captured in module.state (e.g., q_scales, adapters)
        total_bytes += self._state_tensor_bytes(module)

        dtype = dtype or getattr(module, "module_dtype", None)
        dtype_label = self._format_dtype(dtype)
        size_mb = total_bytes / (1024 * 1024)
        return f"{dtype_label}: {size_mb:.1f}MB"

    def _state_tensor_bytes(self, module: NamedModule) -> int:
        """Counts bytes held by tensor-like entries in `module.state`."""

        seen: Set[int] = set()
        total = 0
        for key, value in module.state.items():
            if key in {"in_features", "out_features"}:
                continue
            total += self._collect_tensor_bytes(value, seen)
        return total

    def _collect_tensor_bytes(self, obj: Any, seen: Set[int]) -> int:
        """Recursively sums tensor storage while avoiding double-counting aliases."""

        if isinstance(obj, torch.Tensor):
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            return obj.numel() * obj.element_size()

        if isinstance(obj, (list, tuple, set)):
            return sum(self._collect_tensor_bytes(item, seen) for item in obj)

        if isinstance(obj, dict):
            return sum(self._collect_tensor_bytes(item, seen) for item in obj.values())

        # handle known adapter containers without traversing entire nn.Module graphs
        if hasattr(obj, "lora_A") and hasattr(obj, "lora_B"):
            return (
                self._collect_tensor_bytes(obj.lora_A, seen)
                + self._collect_tensor_bytes(obj.lora_B, seen)
            )

        return 0

    def _format_dtype(self, dtype: Optional[torch.dtype]) -> str:
        """Shortens dtype names for compact table output."""

        if dtype is None:
            return "n/a"

        dtype_str = str(dtype)
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str.split(".", 1)[1]

        dtype_alias = {
            "bfloat16": "bf16",
            "float16": "f16",
            "float32": "f32",
        }

        return dtype_alias.get(dtype_str, dtype_str)

    def _init_device_smi_handles(self) -> Dict[str, Device]:
        """Creates Device-SMI handles for all discovered accelerator devices."""

        handles: Dict[str, Device] = {}

        for device_id in self._discover_accelerator_devices():
            try:
                handles[device_id] = Device(device_id)
            except Exception as exc:  # pragma: no cover - defensive, external tool
                log.debug(f"Device-SMI initialisation failed for `{device_id}`: {exc}")

        return handles

    def _init_cpu_device_handle(self) -> Optional[Device]:
        """Creates the optional Device-SMI handle used for CPU memory tracking."""

        try:
            return Device("cpu")
        except Exception as exc:  # pragma: no cover - defensive, external tool
            log.debug(f"Device-SMI CPU initialisation failed: {exc}")
            return None

    def _discover_accelerator_devices(self) -> List[str]:
        """Lists CUDA/ROCm/XPU device identifiers visible to the runtime."""

        devices: List[str] = []

        if hasattr(torch, "cuda"):
            try:
                if torch.cuda.is_available():
                    device_type = "rocm" if getattr(torch.version, "hip", None) else "cuda"
                    for idx in range(torch.cuda.device_count()):
                        devices.append(f"{device_type}:{idx}")
            except Exception:  # pragma: no cover - defensive, CUDA runtime differences
                pass

        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            try:
                if torch.xpu.is_available():
                    for idx in range(torch.xpu.device_count()):
                        devices.append(f"xpu:{idx}")
            except Exception:  # pragma: no cover - defensive, XPU runtime differences
                pass

        if HAS_NPU:
            try:
                for idx in range(torch.npu.device_count()):
                    devices.append(f"npu:{idx}")
            except Exception:  # pragma: no cover - defensive, NPU runtime differences
                pass

        return devices

    def _safe_query_metric(self, device_key: str, handle: Device):
        """Queries Device-SMI metrics once per device, suppressing repeated failures."""

        with self._device_smi_lock:
            try:
                return handle.metrics(fast=True)
            except Exception as exc:  # pragma: no cover - defensive, external tool
                if device_key not in self._device_metric_failures:
                    log.debug(f"Device-SMI metrics failed for `{device_key}`: {exc}")
                    self._device_metric_failures.add(device_key)
                return None

    def _snapshot_device_memory_gib(self) -> Dict[str, float]:
        """Captures current accelerator memory usage in GiB per device."""

        with self._device_smi_lock:
            snapshot: Dict[str, float] = {}
            for device_id, handle in self._device_smi_handles.items():
                metrics = self._safe_query_metric(device_id, handle)
                if metrics is None:
                    continue
                snapshot[device_id] = metrics.memory_used / (1024 ** 3)
            return snapshot

    def _snapshot_cpu_memory_gib(self) -> Optional[float]:
        """Captures current CPU memory usage in GiB when supported."""

        with self._device_smi_lock:
            if self._cpu_device_smi is None:
                return None
            metrics = self._safe_query_metric("cpu", self._cpu_device_smi)
            if metrics is None:
                return None
            return metrics.memory_used / (1024 ** 3)

    def device_memory_report(self) -> str:
        """Formats current accelerator memory usage for processor log rows."""

        snapshot = self._snapshot_device_memory_gib()
        if not snapshot:
            return "n/a"

        def _format_gib(value: float) -> str:
            """Formats a GiB value without unnecessary trailing zeros."""

            text = f"{value:.2f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            if not text:
                text = "0"
            return f"{text}G"

        grouped: Dict[str, List[Tuple[str, float, int]]] = {}
        for order, (device_id, value) in enumerate(snapshot.items()):
            family, _, index = device_id.partition(":")
            grouped.setdefault(family, []).append((index, value, order))

        segments: List[str] = []
        for family, entries in grouped.items():
            if not entries:
                continue

            def sort_key(item: Tuple[str, float, int]) -> Tuple[int, int]:
                """Sorts indexed devices numerically while preserving fallback order."""

                index, _, order = item
                try:
                    return 0, int(index)
                except (TypeError, ValueError):
                    return 1, order

            values = [_format_gib(value) for _, value, _ in sorted(entries, key=sort_key)]
            segment = f"{family} " + ", ".join(values)
            segments.append(segment)

        return " | ".join(segments)

    def _close_device_smi_handles(self) -> None:
        """Closes all Device-SMI handles owned by this processor."""

        with self._device_smi_lock:
            for handle in self._device_smi_handles.values():
                try:
                    handle.close()
                except Exception:
                    log.debug("Failed to close device telemetry handle during cleanup.", exc_info=True)
            self._device_smi_handles.clear()

            if self._cpu_device_smi is not None:
                try:
                    self._cpu_device_smi.close()
                except Exception:
                    log.debug("Failed to close CPU telemetry handle during cleanup.", exc_info=True)
                self._cpu_device_smi = None

    # Loop Procssor level scoped state data
    def result_save(self, key: str, value: Any):
        """Stores a processor-scoped result by module key."""

        with self._results_lock:
            #assert self.result_get(key) is None, f"key: {key} already exists in `self.result`"
            self._results[key] = value

    # Loop Procssor level scoped state data
    def result_get(self, key: str, default: Any = None) -> Any:
        """Fetches a processor-scoped result by key."""

        with self._results_lock:
            return self._results.get(key, default)

    # Loop Procssor level scoped state data
    def result_pop(self, key: str, default: Any = None):
        """Removes and returns a processor-scoped result by key."""

        with self._results_lock:
            return self._results.pop(key, default)

    # Loop Procssor level scoped state data
    def results(self):
        """Returns the full processor result mapping."""

        with self._results_lock:
            return dict(self._results)

    def collect_memory_info(self, layer_index: int):
        """Records current accelerator and CPU memory snapshots for diagnostics."""

        device_snapshot = self._snapshot_device_memory_gib()
        if device_snapshot:
            total_gpu_memory = sum(device_snapshot.values())
            self.gpu_memorys.append(total_gpu_memory)

        cpu_memory = self._snapshot_cpu_memory_gib()
        if cpu_memory is not None:
            self.cpu_memorys.append(cpu_memory)

    def log_plotly(self):
        """Placeholder for future Plotly-based processor log visualizations."""

        pass

    def set_calibration_dataset(self, calibration_dataset):
        """Override point for processors that need to replace calibration data."""

        pass

    def set_fwd_time(self, fwd_time: float):
        """Stores the latest forward-pass duration for logging."""

        with self._fwd_time_lock:
            self.fwd_time = fwd_time

    def formatted_fwd_time(self) -> str:
        """Returns the stored forward time as a fixed-width string."""

        with self._fwd_time_lock:
            fwd_time = self.fwd_time if self.fwd_time is not None else 0.0
            return f"{fwd_time:.3f}"

    # called first
    def preprocess(self, module: NamedModule, **kwargs):
        """Override point for per-module setup before forward capture/processing."""

        pass

    # after preproces, this process may be skipped due to dynamic override (lora adapter = None)
    def is_skipped(self, module: NamedModule) -> bool:
        """Override point for dynamic per-module skip decisions."""

        pass

    def receive_input_cache(self, input_cache: Any):
        """Injects the shared input cache for the current processor stage."""

        with self._cache_lock:
            current = getattr(self, "inputs_cache", None)
            if isinstance(current, _ThreadSafeInputCache):
                current.set_cache(input_cache)
            else:
                if isinstance(input_cache, _ThreadSafeInputCache):
                    input_cache = input_cache.unwrap()
                object.__setattr__(self, "inputs_cache", _ThreadSafeInputCache(input_cache))

    # called after every module generate
    # may be called multiple times due to batch
    def receive_layer_inputs(self, layer_inputs: List[List[Tensor]]):
        """Replaces cached layer outputs that feed the next loop stage."""

        with self._cache_lock:
            self.inputs_cache.layer_inputs = layer_inputs

    def receive_layer_forward_context(
        self,
        *,
        layer_index: int,
        layer_inputs: List[List[Tensor]],
        layer_input_kwargs: List[Dict[str, Tensor]],
        layer_outputs: List[List[Tensor]],
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Override point for processors that need original layer IO snapshots."""

        del (
            layer_index,
            layer_inputs,
            layer_input_kwargs,
            layer_outputs,
            subset_index,
            subset_total,
        )

    def prepare_subset(
        self,
        subset: Dict[str, NamedModule],
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Override point for per-subset sharing state before forward capture.

        GPTQ uses this lifecycle to prepare same-input Hessian sharing. AWQ uses
        the same generic lifecycle to prepare same-input activation sharing.
        Other quantizers can leave the hook as a no-op when they have no
        per-subset reusable state.
        """

        del subset, subset_index, subset_total

    def cleanup_subset(
        self,
        subset: Optional[Dict[str, NamedModule]] = None,
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Override point for dropping per-subset sharing state after workers finish."""

        del subset, subset_index, subset_total

    def clear_cache_data(self):
        """Drops transient task data and cached layer inputs after replay."""

        self.tasks.clear()
        with self._cache_lock:
            self.inputs_cache.layer_inputs = []

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Override point for per-module forward hooks used during capture."""

        pass

    def begin_shared_input_capture(
        self,
        model: Any,
        subset_names: List[str],
        is_lm_head_module: bool = False,
    ) -> Dict[str, str]:
        """Optionally elect one capture leader per explicit shared-input group in ``subset_names``.

        Called before capture hooks are registered for a subset pass. Returns
        ``{follower: leader}``; processors that do not dedup capture return ``{}``.
        """

        del model, subset_names, is_lm_head_module
        return {}

    def end_shared_input_capture(self, subset_names: List[str]) -> Optional[Dict[str, Any]]:
        """Propagate leader statistics and optionally return capture lifecycle telemetry."""

        del subset_names
        return None

    def register_moe_root_capture_hook(
        self,
        moe_block: Module,
        moe_block_name: str,
        handles: List[Any],
    ) -> bool:
        """Optionally register a processor-specific MoE-root capture hook.

        The generic subset executor invokes this lifecycle extension without
        knowing which processor owns the feature. Processors that do not need
        a shared MoE-root capture leave the default no-op unchanged.
        """

        del moe_block, moe_block_name, handles
        return False

    # do work and return processor.self state which will updated/merged
    def process(
            self,
            module: NamedModule,
            device: torch.device = None,
            subset: Optional[Dict[str, NamedModule]] = None,
            previous_subset: Optional[Dict[str, NamedModule]] = None,
            subset_index: Optional[int] = None,
            subset_total: Optional[int] = None,
    ):
        """Override point for the main per-module quantization or capture step."""

        pass

    # last step, after all loop processor is called
    # submodule_finalize is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Override point for per-module packing/finalization after processing."""

        pass
        #self.offload_to_disk(module=module)

    # last step, after all loop processor is called
    # finalize is called in reverse after all next sequential processes are called
    def finalize(self, model: BaseQModel, **kwargs):
        """Releases shared processor resources after the full quantization loop."""

        self._close_device_smi_handles()
        del self.inputs_cache
        del self._results

        # TODO make this file delete based on user toggle
        # cleanup temp log file
        # if os.path.exists(self.log_tmp_log_file_name):
        #     os.remove(file_path)

    def release_calibration_dataset(self):
        """Drops the retained calibration dataset to free host memory."""

        del self.calibration_dataset

    def number_batches(self) -> int:
        """Returns the number of prepared calibration batches."""

        return self.num_batches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Override point for validating or reusing calibration datasets."""

        pass

    def name(self) -> str:
        """Override point for the processor name shown in logs and reports."""

        pass

def get_max_memory() -> str:
    """Returns current CUDA memory usage for the first one or two devices."""

    stats_0 = torch.cuda.memory_stats(DEVICE_0)
    active_0 = stats_0.get("active_bytes.all.current", 0) / 1024 ** 2

    if torch.cuda.device_count() > 1:
        stats_1 = torch.cuda.memory_stats(DEVICE_1)
        active_1 = stats_1.get("active_bytes.all.current", 0) / 1024 ** 2

        max_memory = f"{active_0:.2f}MB, {active_1:.2f}MB"
    else:
        max_memory = f"{active_0:.2f}MB"
    return max_memory
