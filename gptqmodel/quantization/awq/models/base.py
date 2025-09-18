import torch
import torch.nn as nn
from gptqmodel.quantization.awq.modules.act import ScaledActivation
from gptqmodel.quantization.awq.utils.module import set_op_by_name

# Since we support different `AutoModelForxxx` from transformers
# we need to define a custom mapping dict as below:
TRANSFORMERS_AUTO_MAPPING_DICT = {
    "mpt": "AutoModelForCausalLM",
    "llama": "AutoModelForCausalLM",
    "opt": "AutoModelForCausalLM",
    "RefinedWeb": "AutoModelForCausalLM",
    "RefinedWebModel": "AutoModelForCausalLM",
    "exaone": "AutoModelForCausalLM",
    "falcon": "AutoModelForCausalLM",
    "bloom": "AutoModelForCausalLM",
    "gptj": "AutoModelForCausalLM",
    "gpt_bigcode": "AutoModelForCausalLM",
    "mistral": "AutoModelForCausalLM",
    "mixtral": "AutoModelForCausalLM",
    "gpt_neox": "AutoModelForCausalLM",
    "aquila": "AutoModelForCausalLM",
    "Yi": "AutoModelForCausalLM",
    "qwen": "AutoModelForCausalLM",
    "baichuan": "AutoModelForCausalLM",
    "llava": "AutoModelForVision2Seq",
    "qwen2": "AutoModelForCausalLM",
    "qwen2_vl": "AutoModelForVision2Seq",
    "qwen3": "AutoModelForCausalLM",
    "qwen3_moe": "AutoModelForCausalLM",
    "gemma": "AutoModelForCausalLM",
    "gemma2": "AutoModelForCausalLM",
    "stablelm": "AutoModelForCausalLM",
    "starcoder2": "AutoModelForCausalLM",
    "llava_next": "AutoModelForVision2Seq",
    "phi3": "AutoModelForCausalLM",
    "phi3_v": "AutoModelForCausalLM",
    "cohere": "AutoModelForCausalLM",
    "deepseek_v2": "AutoModelForCausalLM",
    "deepseek_v3": "AutoModelForCausalLM",
    "minicpm": "AutoModelForCausalLM",
    "minicpm3": "AutoModelForCausalLM",
    "internlm2": "AutoModelForCausalLM",
    "qwen2_5_vl": "AutoModelForVision2Seq",
    "qwen2_5_omni": "AutoModelForTextToWaveform",
}


class BaseAWQForCausalLM(nn.Module):
    @staticmethod
    def fuse_layers(model):
        pass

    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)

        if scale_dict["is_scalable"]:
            if not isinstance(scale_dict["scale_layer"], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(
                    scale_dict["scale_shape"], dtype=param.dtype, device=param.device
                )

                # scale activation
                scaled_act = ScaledActivation(scale_dict["scale_layer"], scale_like)
                set_op_by_name(layer, scale_dict["scale_name"], scaled_act)
