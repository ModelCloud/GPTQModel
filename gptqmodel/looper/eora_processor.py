# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module

from ..adapter.adapter import Lora
from ..eora.eora import eora_compute_lora, eora_process_input, merge_eora_segments
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                             PROCESS_LOG_NAME, PROCESS_LOG_TIME, PROCESS_USED_MEMORY)
from ..quantization.config import QuantizeConfig
from ..utils.attn_mask import apply_keep_mask_bt
from ..utils.device import get_device
from ..utils.env import env_flag
from ..utils.logger import setup_logger
from ..utils.model import move_to
from ..utils.torch import CPU, DEVICE_0, DEVICE_1

log = setup_logger()

_EORA_SHARED_COVARIANCE_ENV = "GPTQMODEL_EORA_SHARED_COVARIANCE"


class EoraProcessor(LoopProcessor):
    """Builds LoRA-style error adapters from dequantization residuals and activations."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        require_fwd: bool = True,
        calibration_concat_separator: Optional[str] = None,
    ):
        """Initializes EoRA processing and per-module segment accumulation state."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=ExecutionConfig(require_fwd=require_fwd),
        )

        # Track per-module segment accumulators keyed by device so we can merge
        # contributions without repeatedly moving data through the CPU.
        self._segment_accumulators: Dict[str, Dict[torch.device, Dict[str, Any]]] = {}
        self._module_target_devices: Dict[str, torch.device] = {}
        self._enable_shared_covariance = env_flag(_EORA_SHARED_COVARIANCE_ENV, default=True)
        self._shared_covariance_batch_cache: Dict[Tuple[object, ...], Dict[str, Any]] = {}
        self._shared_covariance_module_groups: Dict[str, Tuple[object, ...]] = {}
        self._shared_covariance_group_counts: Dict[Tuple[object, ...], int] = {}
        self._shared_covariance_stats = {
            "batch_requests": 0,
            "batch_hits": 0,
            "batch_misses": 0,
        }

        # The generic hook wrapper normally applies this mask independently for
        # every module. EoRA applies it inside its hook so same-input siblings
        # retain a common source identity and can reuse one covariance GEMM.
        self.preserve_batch_keep_mask = True

        # Increase the dynamo cache size limit, default of 8 is too low
        if torch._dynamo.config.cache_size_limit < 64:
            torch._dynamo.config.cache_size_limit = 64

        # needed by eora
        # torch._dynamo.config.capture_scalar_outputs = True

        #self.eora_compute_lora = torch_compile(eora_compute_lora)
        #self.eora_process_input = torch_compile(eora_process_input)

        self.eora_compute_lora = eora_compute_lora
        self.eora_process_input = eora_process_input

    def set_calibration_dataset(self, calibration_dataset):
        """Stores the calibration dataset because EoRA depends on batch counts."""

        self.calibration_dataset = calibration_dataset
        self.num_batches = len(calibration_dataset)

    def preprocess(self, module: NamedModule, **kwargs):
        """Clones adapter config, applies rank overrides, and initializes accumulators."""

        # entire module is skipped
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            module.adapter_cfg = None # hack
            return

        adapter_cfg = copy.deepcopy(self.qcfg.adapter)

        # dynamic override of adapter.rank
        adapter_cfg.rank = self.qcfg.dynamic_get(
                module.full_name,
                key="adapter",
                sub_key="rank",
                default=adapter_cfg.rank)

        # hack store property inside module
        module.adapter_cfg = adapter_cfg

        target_device = get_device(module.module)
        if target_device.type == "meta":
            target_device = torch.device("cpu")

        self._module_target_devices[module.name] = torch.device(target_device)
        self._segment_accumulators[module.name] = {}

        return

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports whether EoRA was disabled for this module by dynamic config."""

        # dynamic override removed eora processing for this module
        return module.adapter_cfg in [None, {}]

    @staticmethod
    def _tensor_cache_fingerprint(tensor: torch.Tensor) -> Tuple[object, ...]:
        """Build a stable identity for one activation view during subset replay."""

        try:
            storage_ptr = tensor.untyped_storage().data_ptr()
        except Exception:
            storage_ptr = tensor.data_ptr()
        return (
            str(tensor.device),
            str(tensor.dtype),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
            int(storage_ptr),
        )

    def prepare_subset(
        self,
        subset: Dict[str, NamedModule],
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Prepare exact same-input covariance reuse for one EoRA subset."""

        with self.lock:
            self._shared_covariance_batch_cache.clear()
            self._shared_covariance_module_groups.clear()
            self._shared_covariance_group_counts.clear()

        if not self._enable_shared_covariance:
            return

        groups: Dict[Tuple[object, ...], List[str]] = {}
        for name, named_module in subset.items():
            if named_module.state.get("capture_only") or self.is_skipped(named_module):
                continue
            target_device = self._module_target_devices.get(name, get_device(named_module.module))
            signature = (int(named_module.weight.data.shape[1]), str(target_device))
            groups.setdefault(signature, []).append(name)

        with self.lock:
            for group_index, (signature, names) in enumerate(groups.items()):
                if len(names) < 2:
                    continue
                group_key = (
                    "eora-shared-covariance",
                    subset_index,
                    subset_total,
                    group_index,
                    tuple(names),
                    signature,
                )
                self._shared_covariance_group_counts[group_key] = len(names)
                for name in names:
                    self._shared_covariance_module_groups[name] = group_key

    def cleanup_subset(
        self,
        subset: Optional[Dict[str, NamedModule]] = None,
        *,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ) -> None:
        """Release per-subset covariance reuse entries after all workers finish."""

        del subset, subset_index, subset_total
        with self.lock:
            self._shared_covariance_batch_cache.clear()
            self._shared_covariance_module_groups.clear()
            self._shared_covariance_group_counts.clear()

    def shared_covariance_stats(self, reset: bool = False) -> Dict[str, int]:
        """Return same-input covariance reuse counters for tests and benchmarks."""

        with self.lock:
            stats = dict(self._shared_covariance_stats)
            if reset:
                for key in self._shared_covariance_stats:
                    self._shared_covariance_stats[key] = 0
        return stats

    def _shared_covariance_cache_key(
        self,
        *,
        name: str,
        source: torch.Tensor,
        keep_mask: Optional[torch.Tensor],
        batch_index: Optional[int],
    ) -> Optional[Tuple[object, ...]]:
        """Return the cache key for a compatible same-input EoRA module."""

        with self.lock:
            group_key = self._shared_covariance_module_groups.get(name)
        if group_key is None:
            return None
        mask_fingerprint = self._tensor_cache_fingerprint(keep_mask) if torch.is_tensor(keep_mask) else None
        return (
            group_key,
            batch_index,
            self._tensor_cache_fingerprint(source),
            mask_fingerprint,
        )

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Returns the forward hook that accumulates EoRA activation statistics."""

        def tmp(module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            """Processes one batch of inputs into an EoRA contribution segment."""

            batch_index = self.current_batch_index()
            source = input[0]
            keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)
            cache_key = self._shared_covariance_cache_key(
                name=name,
                source=source,
                keep_mask=keep_mask,
                batch_index=batch_index,
            )

            cached = None
            if cache_key is not None:
                with self.lock:
                    self._shared_covariance_stats["batch_requests"] += 1
                    cached = self._shared_covariance_batch_cache.get(cache_key)
                    if cached is not None:
                        self._shared_covariance_stats["batch_hits"] += 1
                        cached["remaining"] -= 1
                        if cached["remaining"] <= 0:
                            self._shared_covariance_batch_cache.pop(cache_key, None)

            if cached is not None:
                batch = int(cached["batch"])
                contribution = cached["contribution"].clone()
                scale = float(cached["scale"])
            else:
                prepared_source = source
                if (
                    torch.is_tensor(keep_mask)
                    and source.dim() >= 3
                    and keep_mask.ndim == 2
                    and keep_mask.shape[:2] == source.shape[:2]
                ):
                    prepared_source = apply_keep_mask_bt(source, keep_mask)

                batch, contribution, scale = self.eora_process_input(
                    input=(prepared_source,),
                    name=name,
                    sample_size=self.num_batches,
                    device=module.weight.data.device,
                )
                if cache_key is not None:
                    with self.lock:
                        self._shared_covariance_stats["batch_misses"] += 1
                        group_key = self._shared_covariance_module_groups.get(name)
                        remaining = max(0, self._shared_covariance_group_counts.get(group_key, 1) - 1)
                        if remaining:
                            self._shared_covariance_batch_cache[cache_key] = {
                                "batch": batch,
                                "contribution": contribution,
                                "scale": scale,
                                "remaining": remaining,
                                # Retain the source objects while this key is live so
                                # allocator pointer reuse cannot create a false hit.
                                "source": source,
                                "keep_mask": keep_mask,
                            }

            self._accumulate_eora_contribution(
                name=name,
                batch_index=batch_index,
                batch=batch,
                contribution=contribution,
                scale=scale,
            )
        return tmp

    def _accumulate_eora_contribution(
        self,
        *,
        name: str,
        batch_index: Optional[int],
        batch: int,
        contribution: torch.Tensor,
        scale: float,
    ) -> None:
        """Merges one EoRA contribution segment into the per-device accumulator."""

        if batch <= 0:
            return

        contribution = contribution.detach()
        device = torch.device(contribution.device)
        scale_value = float(scale)

        with self.lock:
            accumulators = self._segment_accumulators.setdefault(name, {})
            record = accumulators.get(device)

            index_value = int(batch_index) if batch_index is not None else 0

            if record is None:
                record = {
                    "total": contribution,
                    "scale_product": scale_value,
                    "start_index": index_value,
                    "end_index": index_value,
                    "count": 1,
                }
                accumulators[device] = record
                return

            total = record["total"]
            if total.device != contribution.device:
                total = total.to(device=contribution.device)

            total.mul_(scale_value)
            total.add_(contribution)

            record["total"] = total
            record["scale_product"] *= scale_value
            record["count"] += 1

            if batch_index is not None:
                batch_value = int(batch_index)
                if record["start_index"] is None or batch_value < record["start_index"]:
                    record["start_index"] = batch_value
                if record["end_index"] is None or batch_value > record["end_index"]:
                    record["end_index"] = batch_value
            else:
                if record.get("start_index") is None:
                    record["start_index"] = record["count"] - 1
                record["end_index"] = record["count"] - 1

            del contribution

    def _finalize_eigen_scaling_matrix(self, name: str) -> torch.Tensor:
        """Merges accumulated EoRA segments into the final scaling matrix."""

        with self.lock:
            segments = self._segment_accumulators.pop(name, {})
            target_device = self._module_target_devices.pop(name, None)

        if not segments:
            raise RuntimeError(
                f"EoRA statistics for module '{name}' were not collected before processing."
            )

        ordered_segments = sorted(
            segments.values(),
            key=lambda record: record.get("start_index", 0),
        )

        if target_device is None:
            first_total = ordered_segments[0]["total"]
            target_device = torch.device(first_total.device)

        segment_pairs = []
        for record in ordered_segments:
            total = record["total"]
            if total.device != target_device:
                total = total.to(device=target_device, dtype=torch.float32)
            segment_pairs.append((total, float(record["scale_product"])))

        return merge_eora_segments(segment_pairs)

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Computes and installs the LoRA correction for one quantized module."""

        assert isinstance(module.adapter_cfg, Lora)

        self.pb.title(f"EoRA: Processing {module.name} ({module.module_dtype}) in layer").draw()

        start = time.time()

        eigen_scaling_diag_matrix = self._finalize_eigen_scaling_matrix(module.name)

        tp_info = module.state.get("tp_pad_info")
        pad_cols = 0
        original_cols = module.weight.data.shape[1]
        if isinstance(tp_info, dict):
            pad_cols = int(tp_info.get("pad_cols", 0) or 0)
            original_cols = int(tp_info.get("original_columns", original_cols))

        target_device = module.weight.data.device

        w_wq_delta: torch.Tensor = module.state.pop("w_wq_diff").to(
            dtype=torch.float32,
            device=target_device,
        )
        if pad_cols:
            valid_cols = original_cols + pad_cols
            w_wq_delta = w_wq_delta[:, :valid_cols]

        wq: torch.Tensor = module.state["wq"]
        if pad_cols:
            wq = wq[:, :valid_cols]

        wq_device = wq.to(device=target_device, dtype=module.module_dtype)

        # print(f"types: w = `{w.dtype}`, device = `{w.device}`, wq = `{wq.dtype}`,  device = `{wq.device}`")
        assert w_wq_delta.dtype == torch.float32, f"w_wq_delta dtype: {w_wq_delta.dtype}"

        # log.info(f"EoRA: module native dtype = `{module_native_dtype}")
        A, B = self.eora_compute_lora(
            w_wq_delta=w_wq_delta,
            name=module.name,
            eigen_scaling_diag_matrix=eigen_scaling_diag_matrix,
            rank=module.adapter_cfg.rank,
            dtype=module.module_dtype,
            device=module.weight.data.device,
            use_cholesky=module.adapter_cfg.eora_cholesky,
            eora_config=module.adapter_cfg.eora_config,
        )

        del eigen_scaling_diag_matrix

        # wq with A/B applied
        computed_wq = (wq_device + (B @ A)).to(dtype=wq.dtype, device=target_device)

        if pad_cols:
            computed_wq_trim = computed_wq[:, :original_cols]
            wq_trim = wq[:, :original_cols]
        else:
            computed_wq_trim = computed_wq
            wq_trim = wq

        module.state.update({
            "wq": move_to(wq_trim, device=CPU),
        })

        assert computed_wq.dtype in (torch.float16, torch.bfloat16)

        # override module weight with computed weight with B@A delta
        module.weight.data = computed_wq_trim.to(dtype=module.weight.data.dtype, device=target_device)

        del wq_device, computed_wq

        # for assert weight
        # module.state.update({
        #     "wq_ab": move_to(computed_wq.to(dtype=module.weight.data.dtype), device=CPU),
        # })

        # lowrank_dict[f'{layer_name}.lora_A.weight'] = A.cpu().to(dtype=torch.float16)
        # lowrank_dict[f'{layer_name}.lora_B.weight'] = B.cpu().to(dtype=torch.float16)

        duration = time.time() - start
        with self.lock:
            self.durations.append(duration)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stats_0 = torch.cuda.memory_stats(DEVICE_0)
        peak_active_0 = stats_0.get("active_bytes.all.peak", 0) / 1024 ** 2

        if torch.cuda.device_count() > 1:
            stats_1 = torch.cuda.memory_stats(DEVICE_1)
            peak_active_1 = stats_1.get("active_bytes.all.peak", 0) / 1024 ** 2

            max_memory = f"{peak_active_0:.2f}MB, {peak_active_1:.2f}MB"
        else:
            max_memory = f"{peak_active_0:.2f}MB"

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: max_memory,
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.log.append(stat)

        # log.info(stat)
        self.log_new_row(stat)

        eora = Lora(
                rank=module.adapter_cfg.rank,
                lora_A=move_to(A.to(dtype=module.module_dtype), device=CPU),
                lora_B=move_to(B.to(dtype=module.module_dtype), device=CPU),
            )

        module.state.update({
            "adapter": eora
        })

        module.state.pop("tp_pad_info", None)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Stores the finalized adapter object in the processor result map."""

        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        self.result_save(module.full_name, module.state.pop("adapter"))

    def finalize(self, model: BaseQModel, **kwargs):
        """Releases accumulators and attaches the collected adapters to the model."""

        covariance_stats = self.shared_covariance_stats()
        if covariance_stats["batch_requests"]:
            log.info(
                "EoRA: same-input covariance reuse: "
                f"{covariance_stats['batch_hits']} hits, {covariance_stats['batch_misses']} misses, "
                f"{covariance_stats['batch_requests']} requests."
            )

        del self._segment_accumulators
        del self._module_target_devices
        del self._shared_covariance_batch_cache
        del self._shared_covariance_module_groups
        del self._shared_covariance_group_counts

        # hack: store loras into model until `save()` is called
        model.lora_results = self.results()

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Requires calibration on the first EoRA stage and reuses later caches thereafter."""

        if self.calibration_dataset is None:
            if processor_index == 0:
                raise ValueError("EoraProcessor's calibration_dataset must be provided.")
            else:
                return False
        return True

    def name(self) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "eora"
