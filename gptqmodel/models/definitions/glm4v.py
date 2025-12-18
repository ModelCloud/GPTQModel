# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForImageTextToText
from transformers.activations import ACT2FN
from ..base import BaseQModel
from torch import nn

class Glm4vTextMLPNew(nn.Module):
    def __init__(self, config, ori_mlp=None):
        super().__init__()
        self.config = config
        dtype = None
        device = None
        if ori_mlp is not None:
            dtype = ori_mlp.gate_up_proj.weight.dtype
            device = ori_mlp.gate_up_proj.weight.device

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype, device=device)
        self.activation_fn = ACT2FN[config.hidden_act]

        if ori_mlp is not None:
            gate_w, up_w = ori_mlp.gate_up_proj.weight.data.split(config.intermediate_size, dim=0)
            self.gate_proj.weight.data.copy_(gate_w)
            self.up_proj.weight.data.copy_(up_w)
            self.down_proj.weight.data.copy_(ori_mlp.down_proj.weight.data)
        
    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up   = self.up_proj(hidden_states)
        return self.down_proj(up * self.activation_fn(gate))

class Glm4vGPTQ(BaseQModel):
    loader = AutoModelForImageTextToText

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.glm4v.modeling_glm4v as glm4v_modeling

            glm4v_modeling.Glm4vTextMLP= Glm4vTextMLPNew
