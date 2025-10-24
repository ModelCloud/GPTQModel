# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import json
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from device_smi import Device
import torch
from random_word import RandomWords
from torch import Tensor
from torch.nn import Module

from .. import DEVICE_THREAD_POOL
from ..looper.input_cache import InputCache
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_USED_MEMORY, QUANT_LOG_DAMP, QUANT_LOG_LOSS,
                             QUANT_LOG_NSAMPLES)
from ..quantization.config import QuantizeConfig
from ..utils.colors import ANSIColor, color_text
from ..utils.logger import setup_logger
from ..utils.torch import CPU, DEVICE_0, DEVICE_1

log = setup_logger()

# global level lock
PROCESSOR_GLOBAL_LOCK = threading.Lock()

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

# LoopProcessor is a singleton(), not per module instance
class LoopProcessor:
    def __init__(
            self,
            tokenizer, qcfg: QuantizeConfig,
            calibration,
            prepare_dataset_func: Optional[Callable] = None,
            calibration_concat_size: Optional[int] = None,
            calibration_sort: Optional[str] = None,
            batch_size: int = 1,
            require_fwd: bool = True,
            fwd_after_process: bool = True,
            fwd_all_modules_in_single_pass: bool = False,
    ):
        # process level lock
        self.lock = threading.Lock()

        # result is total collection of all module results mapped by module.full_name
        self._results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()

        self.tokenizer = tokenizer
        self.qcfg = qcfg
        self.qcfg_dynamic = None # cloned and dynamic filtered

        # TODO FIX ME: dequantize processor sets this to False but it is nver acted on!
        # if processor require fwd generate and hooks, set this to true
        # looper should bypass generate + hooks if this is false
        self.require_fwd = require_fwd # default True

        # after process(), do we need to forward again? paried with require_fwd == True
        # if true, forward output is captured post process() and saved for next loop as input
        # if false, forward output before process() call is saved for next loop as input
        self.fwd_after_process = fwd_after_process # default True

        # native processor does not need to forward N times due to module depend segmentation
        # if true, fwd is repeated based on module dep sub-groups
        # if false, sub-module groups are merged as one and fwd happens in one pass
        self.fwd_all_modules_in_single_pass = fwd_all_modules_in_single_pass # default False

        self.inputs_cache: InputCache = InputCache(None, None, None, None)
        self.tasks = {}

        self.pb = None
        self.fwd_time = None
        self.layer_count = None


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
        self.log_tmp_log_file_name = f"{self.name()}_log_{RandomWords().get_random_word()}_time_{current_time}.log"
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

            calibration = prepare_dataset_func(calibration_dataset=calibration,
                                               calibration_dataset_concat_size=calibration_concat_size,
                                               calibration_dataset_sort=calibration_sort,
                                               batch_size=batch_size)

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

        # Track the current calibration batch index on a per-thread basis so
        # processors can retrieve deterministic ordering information (e.g.
        # GPTQ's Hessian updates) even when forwards run on multiple threads.
        self._batch_tls = threading.local()

    def _set_current_batch_index(self, batch_index: Optional[int]) -> None:
        if batch_index is None:
            if hasattr(self._batch_tls, "index"):
                delattr(self._batch_tls, "index")
        else:
            self._batch_tls.index = int(batch_index)

    def current_batch_index(self) -> Optional[int]:
        return getattr(self._batch_tls, "index", None)

    def _async_log_writer(self, stat):
        with open(self.log_tmp_log_file_name, 'a') as f:
            json.dump(stat, f, indent=4)
            f.write("\n")

    def log_save_async(self, stat):
        # Serialize writes on the CPU-bound worker to avoid interleaved JSON output.
        DEVICE_THREAD_POOL.submit_serial(CPU, self._async_log_writer, stat)

    def log_new_row(self, stat):
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

        self.log_save_async(stat)

    def loss_color(self, loss_value: float) -> ANSIColor:
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
        text = "" if value is None else str(value)

        if key == QUANT_LOG_LOSS and text:
            try:
                color_code = self.loss_color(float(text))
            except (TypeError, ValueError):
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
        in_features = module.state.get("in_features")
        out_features = module.state.get("out_features")

        if isinstance(in_features, int) and isinstance(out_features, int):
            return f"{in_features}, {out_features}"
        return ""

    def module_dtype_size_summary(self, module: NamedModule) -> str:
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
        seen: Set[int] = set()
        total = 0
        for key, value in module.state.items():
            if key in {"in_features", "out_features"}:
                continue
            total += self._collect_tensor_bytes(value, seen)
        return total

    def _collect_tensor_bytes(self, obj: Any, seen: Set[int]) -> int:
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
        handles: Dict[str, Device] = {}

        for device_id in self._discover_accelerator_devices():
            try:
                handles[device_id] = Device(device_id)
            except Exception as exc:  # pragma: no cover - defensive, external tool
                log.debug(f"Device-SMI initialisation failed for `{device_id}`: {exc}")

        return handles

    def _init_cpu_device_handle(self) -> Optional[Device]:
        try:
            return Device("cpu")
        except Exception as exc:  # pragma: no cover - defensive, external tool
            log.debug(f"Device-SMI CPU initialisation failed: {exc}")
            return None

    def _discover_accelerator_devices(self) -> List[str]:
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

        return devices

    def _safe_query_metric(self, device_key: str, handle: Device):
        try:
            return handle.metrics(fast=True)
        except Exception as exc:  # pragma: no cover - defensive, external tool
            if device_key not in self._device_metric_failures:
                log.debug(f"Device-SMI metrics failed for `{device_key}`: {exc}")
                self._device_metric_failures.add(device_key)
            return None

    def _snapshot_device_memory_gib(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for device_id, handle in self._device_smi_handles.items():
            metrics = self._safe_query_metric(device_id, handle)
            if metrics is None:
                continue
            snapshot[device_id] = metrics.memory_used / (1024 ** 3)
        return snapshot

    def _snapshot_cpu_memory_gib(self) -> Optional[float]:
        if self._cpu_device_smi is None:
            return None
        metrics = self._safe_query_metric("cpu", self._cpu_device_smi)
        if metrics is None:
            return None
        return metrics.memory_used / (1024 ** 3)

    def device_memory_report(self) -> str:
        snapshot = self._snapshot_device_memory_gib()
        if not snapshot:
            return "n/a"

        def _format_gib(value: float) -> str:
            text = f"{value:.1f}"
            if text.endswith(".0"):
                text = text[:-2]
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
        for handle in self._device_smi_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._device_smi_handles.clear()

        if self._cpu_device_smi is not None:
            try:
                self._cpu_device_smi.close()
            except Exception:
                pass
            self._cpu_device_smi = None

    # Loop Procssor level scoped state data
    def result_save(self, key: str, value: Any):
        with self._results_lock:
            #assert self.result_get(key) is None, f"key: {key} already exists in `self.result`"
            self._results[key] = value

    # Loop Procssor level scoped state data
    def result_get(self, key: str, default: Any = None) -> Any:
        with self._results_lock:
            return self._results.get(key, default)

    # Loop Procssor level scoped state data
    def result_pop(self, key: str, default: Any = None):
        with self._results_lock:
            return self._results.pop(key, default)

    # Loop Procssor level scoped state data
    def results(self):
        return self._results

    def collect_memory_info(self, layer_index: int):
        device_snapshot = self._snapshot_device_memory_gib()
        if device_snapshot:
            total_gpu_memory = sum(device_snapshot.values())
            self.gpu_memorys.append(total_gpu_memory)

        cpu_memory = self._snapshot_cpu_memory_gib()
        if cpu_memory is not None:
            self.cpu_memorys.append(cpu_memory)

    def log_plotly(self):
        pass

    def set_calibration_dataset(self, calibration_dataset):
        pass

    def set_fwd_time(self, fwd_time: float):
        self.fwd_time = fwd_time

    def formatted_fwd_time(self) -> str:
        fwd_time = self.fwd_time if self.fwd_time is not None else 0.0
        return f"{fwd_time:.3f}"

    # called first
    def preprocess(self, module: NamedModule, **kwargs):
        pass

    # after preproces, this process may be skipped due to dynamic override (lora adapter = None)
    def is_skipped(self, module: NamedModule) -> bool:
        pass

    def receive_input_cache(self, input_cache: InputCache):
        self.inputs_cache = input_cache

    # called after every module generate
    # may be called multiple times due to batch
    def receive_layer_inputs(self, layer_inputs: List[List[Tensor]]):
        self.inputs_cache.layer_inputs = layer_inputs

    def clear_cache_data(self):
        self.tasks = {}
        self.inputs_cache.layer_inputs = []

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        pass

    # do work and return processor.self state which will updated/merged
    def process(self, module: NamedModule, device: torch.device = None):
        pass

    # last step, after all loop processor is called
    # submodule_finalize is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        pass
        #self.offload_to_disk(module=module)

    # last step, after all loop processor is called
    # finalize is called in reverse after all next sequential processes are called
    def finalize(self, model: BaseQModel, **kwargs):
        self._close_device_smi_handles()
        del self.inputs_cache
        del self._results

        # TODO make this file delete based on user toggle
        # cleanup temp log file
        # if os.path.exists(self.log_tmp_log_file_name):
        #     os.remove(file_path)

    def release_calibration_dataset(self):
        del self.calibration_dataset

    def number_batches(self) -> int:
        return self.num_batches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        pass

    def name(self) -> str:
        pass

def get_max_memory() -> str:
    stats_0 = torch.cuda.memory_stats(DEVICE_0)
    active_0 = stats_0.get("active_bytes.all.current", 0) / 1024 ** 2
    peak_active_0 = stats_0.get("active_bytes.all.peak", 0) / 1024 ** 2

    if torch.cuda.device_count() > 1:
        stats_1 = torch.cuda.memory_stats(DEVICE_1)
        active_1 = stats_1.get("active_bytes.all.current", 0) / 1024 ** 2
        peak_active_1 = stats_1.get("active_bytes.all.peak", 0) / 1024 ** 2

        max_memory = f"{active_0:.2f}MB, {active_1:.2f}MB"
    else:
        max_memory = f"{active_0:.2f}MB"
    return max_memory
