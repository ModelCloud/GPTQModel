"""Per-module FP8 capture and fitting task for calibrated processor integration."""

import torch

from .config import FP8Config
from .gsq_calibration import GSQInputGram
from .gsq_fp8 import refine_fp8_weight


class FP8GSQTask:
    """Capture a frozen teacher and consume its calibration statistics once.

    This task does not install a replacement module. It returns actual packed
    FP8 storage and inverse scales for the processor's export step.
    """

    def __init__(self, weight, config, *, max_gram_bytes=1024**3):
        if not isinstance(config, FP8Config):
            raise TypeError('FP8 GSQ task requires FP8Config')
        if config.gsq is None or not config.gsq.enabled:
            raise ValueError('FP8 calibrated GSQ task requires enabled GSQ')
        config._validate_gsq()
        if weight.ndim != 2 or not weight.is_floating_point() or not torch.isfinite(weight).all():
            raise ValueError('FP8 GSQ task requires finite floating [out,in] teacher weights')
        import copy

        self.config = copy.deepcopy(config)
        self.teacher = weight.detach().float().clone()
        self.capture = GSQInputGram(weight.shape[1], device=weight.device, max_bytes=max_gram_bytes)
        self._consumed = False

    def add_batch(self, inputs, *, source_weight=1., mask=None):
        self.capture.add(inputs, source_weight=source_weight, mask=mask)

    def quantize(self):
        from ..nn_modules.qlinear.fp8 import TorchFP8Linear

        if self._consumed:
            raise RuntimeError('FP8 GSQ task already consumed')
        gram, stats = self.capture.take()
        self._consumed = True
        try:
            rows, columns = self.teacher.shape
            transport = torch.nn.Linear(columns, rows, bias=False, device='meta')
            transport.weight = torch.nn.Parameter(self.teacher.detach(), requires_grad=False)
            packed = TorchFP8Linear(
                bits=8, group_size=-1, sym=True, desc_act=False, in_features=columns,
                out_features=rows, bias=False, **self.config.quant_linear_init_kwargs())
            packed.pack_original(transport, None, None, smooth=self.config.smooth)
            weight, scales = packed.weight, packed.weight_scale_inv
            device = self.teacher.device
            result = refine_fp8_weight(
                weight.to(device), scales.to(device), target=self.teacher,
                hessian=gram, config=self.config.gsq, method=self.config.weight_scale_method,
                block_size=self.config.weight_block_size)
            result['diagnostics'] = {key: result[key] for key in ('before', 'after', 'history')}
            result['diagnostics'].update(stats, objective='activation_reconstruction',
                                         capture='source_weighted_input_gram')
            return result
        finally:
            self.free()

    def free(self):
        self.teacher = None
        self.capture.clear()
        self._consumed = True
