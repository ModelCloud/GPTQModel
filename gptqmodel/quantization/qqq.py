from copy import deepcopy

import torch

from ..quantization import GPTQ, Quantizer
from ..quantization.quantizer import QQQQuantizer


class QQQ(GPTQ):
    def create_quantizer(self, name: str) -> Quantizer:
        return QQQQuantizer(self.qcfg, name=name)

    @torch.inference_mode()
    def quantize(
        self,
        blocksize=128,
    ):
        wq, scale, zero, g_idx, duration, avg_loss, damp_percent = super().quantize(blocksize=blocksize)

        # post int8 quant
        scale_extra = None
        if self.qcfg.group_size != self.columns:
            qcfg = deepcopy(self.qcfg)
            qcfg.group_size = -1
            qcfg.bits = 8
            qcfg.sym = True
            qcfg.mse = 0.0

            quantizer_extra = QQQQuantizer(qcfg, name=self.quantizer.name)
            quantizer_extra.configure(
                perchannel=True,
            )
            quantizer_extra.find_params(self.module.weight.data.clone(), weight=True)
            scale_extra = quantizer_extra.scale
        return wq, scale, zero, g_idx, duration, avg_loss, damp_percent, scale_extra
