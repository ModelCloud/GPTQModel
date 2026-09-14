"""Calibration-aware FP8 capture, fitting, replay and export processor."""

import threading
import time

import torch

from .loop_processor import ExecutionConfig, LoopProcessor
from .weight_only_processor import WeightOnlyProcessor
from ..quantization.config import FP8Config, clone_weight_only_config_for_module
from ..quantization.gsq_fp8_task import FP8GSQTask
from ..quantization.gsq_scalar import gsq_enabled_for


class FP8GSQProcessor(WeightOnlyProcessor):
    """Capture module inputs, then export the fitted FP8 bytes without refitting."""

    def __init__(self, tokenizer, qcfg, calibration, prepare_dataset_func,
                 calibration_concat_size=None, calibration_sort=None, batch_size=1,
                 calibration_concat_separator=None):
        if not isinstance(qcfg, FP8Config):
            raise TypeError('FP8 GSQ processor requires FP8Config')
        LoopProcessor.__init__(
            self, tokenizer=tokenizer, qcfg=qcfg, calibration=calibration,
            prepare_dataset_func=prepare_dataset_func, calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort, calibration_concat_separator=calibration_concat_separator,
            batch_size=batch_size, execution_config=ExecutionConfig(require_fwd=True))
        self.lock = threading.Lock()
        self.is_weight_only = False
        self.preserve_batch_keep_mask = True

    def verify_calibration_dataset(self, processor_index):
        if not self.calibration_dataset:
            raise ValueError('Calibrated FP8 GSQ requires calibration data')
        return True

    def release_calibration_dataset(self):
        self._sequence_weights = [row.get('fisher_sequence_weight') for row in self.calibration_dataset]
        super().release_calibration_dataset()

    def preprocess(self, module, **kwargs):
        config = clone_weight_only_config_for_module(self.qcfg, module.full_name)
        if config is None:
            return
        if config.weight_scale_method == "tensor":
            raise ValueError("FP8 calibrated GSQ processor requires native activation replay for tensor scales")
        task = FP8GSQTask(module.weight, config) if gsq_enabled_for(config.gsq, module.full_name) else None
        self.tasks[module.name] = task

    def is_skipped(self, module):
        return module.name not in self.tasks

    def pre_process_fwd_hook(self, name):
        def capture(_module, inputs, _output):
            task = self.tasks[name]
            if task is not None:
                source = inputs[0]
                mask = getattr(getattr(self, '_mask_tls', None), 'value', None)
                if mask is not None:
                    if mask.numel() != source.numel() // source.shape[-1]:
                        raise ValueError('FP8 GSQ mask cannot be aligned to module input tokens')
                    mask = mask.to(device=source.device, dtype=torch.bool).reshape(source.shape[:-1])
                index = self.current_batch_index() if hasattr(self, '_batch_tls') else None
                batches = getattr(self, 'calibration_dataset', [])
                saved = getattr(self, '_sequence_weights', None)
                weights = None if index is None else (saved[index] if saved is not None
                                                      else batches[index].get('fisher_sequence_weight'))
                if weights is None:
                    task.add_batch(source, mask=mask)
                else:
                    weights = torch.as_tensor(weights).reshape(-1)
                    if source.ndim != 3 or len(weights) != source.shape[0]:
                        raise ValueError('FP8 GSQ sequence weights require aligned [batch,tokens,in] inputs')
                    if not torch.isfinite(weights).all() or (weights < 0).any():
                        raise ValueError('FP8 GSQ sequence weights must be finite and nonnegative')
                    for sequence, weight in enumerate(weights.tolist()):
                        task.add_batch(source[sequence], source_weight=weight,
                                       mask=None if mask is None else mask[sequence])
        return capture

    def process(self, module, device=None, **kwargs):
        started = time.perf_counter()
        task = self.tasks.pop(module.name)
        try:
            config = self.quantize_module(module, device=device)
            if task is not None:
                result = task.quantize()
            else:
                from ..nn_modules.qlinear.fp8 import TorchFP8Linear

                packed = TorchFP8Linear(
                    bits=8, group_size=-1, sym=True, desc_act=False,
                    in_features=module.weight.shape[1], out_features=module.weight.shape[0],
                    bias=module.module.bias is not None,
                    **config.quant_linear_init_kwargs())
                packed.pack_original(module.module, None, None, smooth=config.smooth)
                result = {'weight': packed.weight, 'scale_inv': packed.weight_scale_inv,
                          'diagnostics': {'status': 'skipped', 'reason': 'module_not_selected'}}
            from ..nn_modules.qlinear.fp8 import TorchFP8Linear

            module.state['fp8_reference_weight'] = module.weight.detach().float().cpu().clone()
            module.state['gsq_fp8_result'] = result
            module.state['gsq_diagnostics'] = result['diagnostics']
            from ..models.writer import PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, QUANT_LOG_NSAMPLES

            with self.lock:
                for entry in reversed(self.log):
                    if (entry.get(PROCESS_LOG_LAYER) == module.layer_index
                            and entry.get(PROCESS_LOG_MODULE) == module.name):
                        entry['lifecycle'] = 'calibrated_fp8'
                        entry['gsq'] = result['diagnostics']
                        entry[QUANT_LOG_NSAMPLES] = str(result['diagnostics'].get('tokens', 0))
                        break
            replay_decoder = TorchFP8Linear(
                bits=8, group_size=-1, sym=True, desc_act=False,
                in_features=module.weight.shape[1], out_features=module.weight.shape[0], bias=False,
                **config.quant_linear_init_kwargs())
            replay_decoder.weight = result['weight']
            replay_decoder.weight_scale_inv = result['scale_inv']
            decoded = replay_decoder.dequantize_weight(device=module.weight.device, dtype=module.weight.dtype).T
            with torch.inference_mode():
                module.weight.copy_(decoded)
            from ..models.writer import PROCESS_LOG_TIME

            with self.lock:
                for entry in reversed(self.log):
                    if (entry.get(PROCESS_LOG_LAYER) == module.layer_index
                            and entry.get(PROCESS_LOG_MODULE) == module.name):
                        entry[PROCESS_LOG_TIME] = f"{time.perf_counter()-started:.3f}"
                        break
            return config
        finally:
            if task is not None:
                task.free()

    def clear_cache_data(self):
        for task in list(self.tasks.values()):
            if task is not None:
                task.free()
        super().clear_cache_data()

    def finalize(self, model, **kwargs):
        self._sequence_weights = []
        super().finalize(model, **kwargs)

    def name(self):
        return 'fp8_gsq'
