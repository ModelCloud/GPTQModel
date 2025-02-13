from typing import Callable, Tuple, Dict
import torch
from gptqmodel import QuantizeConfig
from gptqmodel.looper.loop_processor import LoopProcessor
from torch.nn import Module
from torch import Tensor

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.writer import (QUANT_LOG_DAMP, QUANT_LOG_FWD_TIME, QUANT_LOG_LAYER,
                     QUANT_LOG_LOSS, QUANT_LOG_MODULE, QUANT_LOG_TIME)
from gptqmodel.quantization import GPTQ
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.progress import ProgressBar

logger = setup_logger()

class GPTQProcessor(LoopProcessor):
    def __init__(self, calibration_data, qcfg: QuantizeConfig):
        super().__init__(calibration_data=calibration_data, qcfg=qcfg)
        self.durations = []
        self.avg_losses = []
        self.module_names = []
        self.quant_log = []

    def preprocess(self, module: NamedModule, buffered_fwd: bool):
        bits = self.qcfg.bits
        sym = self.qcfg.sym
        mse = self.qcfg.mse

        # dynamic overrides
        if self.qcfg.dynamic is not None:
            bits = self.qcfg.dynamic_get(module.full_name, "bits", bits)
            sym = self.qcfg.dynamic_get(module.full_name, "sym", sym)
            mse = self.qcfg.dynamic_get(module.full_name, "mse", mse)

        tmp = GPTQ(module)

        # models like DeepSeek v3/r1 has > 256 $ of sub-modules per layer
        # use buffered mode go vram don't explode: gptq needs to store fwd inputs per each layer fwd
        # all sub-modules within a single layer needs to store all the inputs.
        # deepseek has massive # of sub-modules per layer, causing vram pressure
        # buffered mode is slower due to gpu<->cpu movement
        if buffered_fwd:  # TODO tweak this number for masive MoE
            logger.info(f"Experimental: enabling fwd buffered mode for: `{module.name}`")
            tmp.fwd_inputs_buffered = True

        tmp.quantizer.configure(
            bits,
            perchannel=True,
            sym=sym,
            mse=mse,
        )
        return tmp

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            # gptq is mutable.
            g = gptq[name]  # noqa: F821
            g.add_batch(inp[0].data, out.data)  # noqa: F821
        return tmp

    def process(self, module: NamedModule, state: Dict[str, ], pb: ProgressBar , fwd_time: int):
        # pb.set_description(f"Quantizing {name} in layer {module_index} of {layer_count - 1}")
        gptq = self.tasks

        group_size = self.qcfg.group_size
        desc_act = self.qcfg.desc_act
        damp_percent = self.qcfg.damp_percent
        static_groups = self.qcfg.static_groups

        # dynamic overrides
        if self.qcfg.dynamic is not None:
            group_size = self.qcfg.dynamic_get(module.full_name, "group_size", group_size)
            desc_act = self.qcfg.dynamic_get(module.full_name, "desc_act", desc_act)
            damp_percent = self.qcfg.dynamic_get(module.full_name, "damp_percent", damp_percent)
            static_groups = self.qcfg.dynamic_get(module.full_name, "static_groups", static_groups)

        # logger.info(f"Quantizing module START: {name}, {gptq[name].shape()}")
        ## Need to return the quantized_weight for offloading
        scale, zero, g_idx, duration, avg_loss, damp_percent, q_full_weight = gptq[module.name].quantize(
            percdamp=damp_percent,
            group_size=group_size,
            actorder=desc_act,
            static_groups=static_groups,
        )
        ## Assign the quantized weight to the weight
        #gptq[name].layer.weight.data = q_full_weight.to(device=gptq[name].device)

        ## Offload the quantized weight to CPU for EoRA
        #quantized_weights['model.layers.%d.%s' % (module_index, name)] = q_full_weights.cpu()

        # if task is not None:
        #     task.get_logger().report_scalar(
        #         title='Quantization Loss',
        #         series=f'layer_{module_index}_loss',
        #         value=avg_loss,
        #         iteration=name_index,
        #     )
        #
        #     task.get_logger().report_scalar(
        #         title='Quantization Time',
        #         series=f'layer_{module_index}_time',
        #         value=duration,
        #         iteration=name_index,
        #     )
        self.durations.append(duration)
        self.avg_losses.append(avg_loss)
        self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stat = {QUANT_LOG_LAYER: module.layer_index, QUANT_LOG_MODULE: module.name, QUANT_LOG_LOSS: f"{avg_loss:.5f}",
                QUANT_LOG_DAMP: f"{damp_percent:.5f}", QUANT_LOG_TIME: f"{duration:.3f}",
                QUANT_LOG_FWD_TIME: f"{fwd_time:.3f}"}
        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        self.quant_log.append(stat)
        logger.info(stat)

        # quantizers[layer_name] = (
        #     gptq[name].quantizer.to(CPU),
        #     move_to(scale, CPU),
        #     move_to(zero, CPU),
        #     move_to(g_idx, CPU),
        # )
        gptq[module.name].free()
        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        return {
            "scale": scale,
            "zero": zero,
            "g_idx": g_idx,
            "duration": duration,
            "avg_loss": avg_loss,
            "damp_percent": damp_percent,
            "q_full_weight": q_full_weight,
        }

    def post_process(self, module: NamedModule, state: Dict[str,]):
        module.weight.data = state["q_full_weight"] # module.layer.weight or module.weight?
        pass

    def clear_input(self):
        self.inputs_cache = []

    def finalize(self, module: NamedModule, state: Dict[str,]):
        pass