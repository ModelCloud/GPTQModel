from typing import Callable, Tuple

import torch
from gptqmodel import QuantizeConfig
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import STAT_GPTQ_AVG_LOSS, STAT_GPTQ_DAMP_PERCENT, STAT_GPTQ_DURATION, NamedModule
from gptqmodel.models import BaseGPTQModel
from gptqmodel.models.writer import (QUANT_LOG_DAMP, QUANT_LOG_FWD_TIME, QUANT_LOG_LAYER,
                                     QUANT_LOG_LOSS, QUANT_LOG_MODULE, QUANT_LOG_TIME)
from gptqmodel.quantization import GPTQ
from gptqmodel.quantization.gptq import CPU
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.model import move_to
from torch.nn import Module

logger = setup_logger()

class GPTQProcessor(LoopProcessor):
    def __init__(self, calibration_data, qcfg: QuantizeConfig):
        super().__init__(calibration_data=calibration_data, qcfg=qcfg)
        self.durations = []
        self.avg_losses = []
        self.module_names = []
        self.quant_log = []
        self.quantizers = {}

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

    def process(self, module: NamedModule):
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
        wq, scale, zero, g_idx, duration, avg_loss, damp_percent  = gptq[module.name].quantize(
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
                QUANT_LOG_FWD_TIME: f"{module.state.get('fwd_time'):.3f}"}
        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        self.quant_log.append(stat)
        logger.info(stat)

        self.quantizers[module.full_name] = (
            gptq[module.name].quantizer.to(CPU),
            move_to(scale, CPU),
            move_to(zero, CPU),
            move_to(g_idx, CPU),
        )
        w = module.weight.data
        module.weight.data = None # Processor should fix this

        gptq[module.name].free()
        # logger.info(f"Quantizing module END: {name}, {gptq[name].shape()}")
        module.state[module.full_name] = {
            "w": w, # fp16, non-quantized weight
            "wq": wq, # fp16, quantized weight but not int4 (packed qweight)
            STAT_GPTQ_DURATION: duration, # stat
            STAT_GPTQ_AVG_LOSS: avg_loss, # stat
            STAT_GPTQ_DAMP_PERCENT: damp_percent, # stat
        }

    def post_process(self, module: NamedModule):
        # prepare for module.foward post generate
        module.weight.data = module.state["wq"] # module.layer.weight or module.weight?

    def submodule_finalize(self, module: NamedModule):
        # generate complete, safe to move to cpu
        module.weight.data = None
        wq = module.state.pop("wq").cpu()
        module.weight.data = wq

    def model_finalize(self, gptq_model: BaseGPTQModel, **kwargs):
        backend = kwargs.pop("backend")
        gptq_model.qlinear_kernel = gptq_model.pack_model(
            model=gptq_model.model,
            quantizers=self.quantizers,
            bits=self.qcfg.bits,
            group_size=self.qcfg.group_size,
            backend=backend,
            desc_act=self.qcfg.desc_act,
            format=self.qcfg.format,
            lm_head_name=gptq_model.lm_head,
            dynamic=self.qcfg.dynamic,
            parallel_packing=self.qcfg.parallel_packing,
            pack_dtype=self.qcfg.pack_dtype,
        )
        gptq_model.quantized = True

        del self.quantizers

