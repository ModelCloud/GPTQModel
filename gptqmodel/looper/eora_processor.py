# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import time
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module

from ..adapter.adapter import Lora
from ..eora.eora import eora_compute_lora, eora_process_input
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE,
                             PROCESS_LOG_NAME, PROCESS_LOG_TIME, PROCESS_USED_MEMORY)
from ..quantization.config import QuantizeConfig
from ..utils.logger import setup_logger
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

        # dict: key is module name, value is the accumulated eigen_scaling_diag_matrix
        self.eigen_scaling_diag_matrix: Dict[str, torch.Tensor] = {}
        self._pending_contributions: Dict[str, Dict[int, Tuple[int, torch.Tensor, float]]] = {}
        self._next_batch_index: Dict[str, int] = {}

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

        self.eigen_scaling_diag_matrix[module.name] = None
        self._pending_contributions[module.name] = {}
        self._next_batch_index[module.name] = 0

        return

    def is_skipped(self, module: NamedModule) -> bool:
        # dynamic override removed eora processing for this module
        return module.adapter_cfg in [None, {}]

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            with tf32_disable_guard():
                batch_index = self.current_batch_index()
                batch, contribution, scale = self.eora_process_input(
                    input=input,
                    name=name,
                    sample_size=self.num_batches,
                    device=module.weight.data.device,
                )

            self._stage_eora_contribution(
                name=name,
                batch_index=batch_index,
                batch=batch,
                contribution=contribution,
                scale=scale,
            )
        return tmp

    def _stage_eora_contribution(
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

        with self.lock:
            if name not in self._pending_contributions:
                self._pending_contributions[name] = {}
                self._next_batch_index.setdefault(name, 0)

            pending_index = batch_index if batch_index is not None else self._next_batch_index[name]
            self._pending_contributions[name][pending_index] = (batch, contribution, scale)
            self._flush_eora_pending_locked(name)

    def _flush_eora_pending_locked(self, name: str) -> None:
        pending = self._pending_contributions.get(name)
        if pending is None:
            return

        while True:
            next_index = self._next_batch_index.get(name, 0)
            update = pending.pop(next_index, None)
            if update is None:
                break

            _batch, contribution, scale = update

            current = self.eigen_scaling_diag_matrix.get(name)

            if isinstance(current, torch.Tensor):
                if current.device != torch.device("cpu"):
                    current = current.to(device=torch.device("cpu"), dtype=torch.float32)

                current.mul_(scale)
                current.add_(contribution)
                self.eigen_scaling_diag_matrix[name] = current
                del contribution
            else:
                self.eigen_scaling_diag_matrix[name] = contribution

            self._next_batch_index[name] = next_index + 1

    def process(self, module: NamedModule):
        assert isinstance(module.adapter_cfg, Lora)

        self.pb.title(f"EoRA: Processing {module.name} ({module.module_dtype}) in layer").draw()

        start = time.time()

        with self.lock:
            self._flush_eora_pending_locked(module.name)
            eigen_scaling_diag_matrix = self.eigen_scaling_diag_matrix.pop(module.name)
            self._pending_contributions.pop(module.name, None)
            self._next_batch_index.pop(module.name, None)

        if not isinstance(eigen_scaling_diag_matrix, torch.Tensor):
            raise RuntimeError(
                f"EoRA statistics for module '{module.name}' were not collected before processing."
            )

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
        with tf32_disable_guard():
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
            "wq": move_to(wq_trim, device=CPU, stream=self.stream),
        })

        assert computed_wq.dtype in (torch.float16, torch.bfloat16)

        # override module weight with computed weight with B@A delta
        module.weight.data = computed_wq_trim.to(dtype=module.weight.data.dtype, device=target_device)

        del wq_device, computed_wq

        # for assert weight
        # module.state.update({
        #     "wq_ab": move_to(computed_wq.to(dtype=module.weight.data.dtype), device=CPU, stream=self.stream),
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
                lora_A=move_to(A.to(dtype=module.module_dtype), device=CPU, stream=self.stream),
                lora_B=move_to(B.to(dtype=module.module_dtype), device=CPU, stream=self.stream),
            )

        module.state.update({
            "adapter": eora
        })

        module.state.pop("tp_pad_info", None)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        self.result_save(module.full_name, module.state.pop("adapter"))

    def finalize(self, model: BaseQModel, **kwargs):
        # block for streams
        if self.stream:
            torch_sync()

        del self.eigen_scaling_diag_matrix
        del self._pending_contributions
        del self._next_batch_index

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
