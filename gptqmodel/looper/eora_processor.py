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
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                             PROCESS_LOG_NAME, PROCESS_LOG_TIME, PROCESS_USED_MEMORY)
from ..quantization.config import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.device import get_device
from ..utils.model import move_to
from ..utils.torch import CPU, DEVICE_0, DEVICE_1, torch_streamCtx, torch_sync
from ..utils.torch import HAS_CUDA, tf32_disable_guard, torch_streamCtx, torch_sync

log = setup_logger()


class EoraProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration, prepare_dataset_func,
                 calibration_concat_size: Optional[int], calibration_sort: Optional[str], batch_size: int,
                 require_fwd: bool = True
                 ):
        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration=calibration,
                         calibration_concat_size=calibration_concat_size,
                         calibration_sort=calibration_sort,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         require_fwd=require_fwd)

        # Track per-module segment accumulators keyed by device so we can merge
        # contributions without repeatedly moving data through the CPU.
        self._segment_accumulators: Dict[str, Dict[torch.device, Dict[str, Any]]] = {}
        self._module_target_devices: Dict[str, torch.device] = {}

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
        self.calibration_dataset = calibration_dataset
        self.num_batches = len(calibration_dataset)

    def preprocess(self, module: NamedModule, **kwargs):
        # entire module is skipped
        if self.qcfg.dynamic_get(layer_name=module.full_name) == False:
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
        # dynamic override removed eora processing for this module
        return module.adapter_cfg in [None, {}]

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            batch_index = self.current_batch_index()
            batch, contribution, scale = self.eora_process_input(
                input=input,
                name=name,
                sample_size=self.num_batches,
                device=module.weight.data.device,
            )

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

    def process(self, module: NamedModule):
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
        active_0 = stats_0.get("active_bytes.all.current", 0) / 1024 ** 2
        peak_active_0 = stats_0.get("active_bytes.all.peak", 0) / 1024 ** 2

        if torch.cuda.device_count() > 1:
            stats_1 = torch.cuda.memory_stats(DEVICE_1)
            active_1 = stats_1.get("active_bytes.all.current", 0) / 1024 ** 2
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
        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        self.result_save(module.full_name, module.state.pop("adapter"))

    def finalize(self, model: BaseQModel, **kwargs):
        del self._segment_accumulators
        del self._module_target_devices

        # hack: store loras into model until `save()` is called
        model.lora_results = self.results()

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        if self.calibration_dataset is None:
            if processor_index == 0:
                raise ValueError("EoraProcessor's calibration_dataset must be provided.")
            else:
                return False
        return True

    def name(self) -> str:
        return "eora"
