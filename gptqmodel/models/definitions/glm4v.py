# Contact: qubitium@modelcloud.ai, x.com/qubitium

from torch import nn

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader


def _defused_glm4v_mlp_forward(self, hidden_states):
    """Run the native GLM-4V MLP from its split gate/up projections."""

    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    return self.down_proj(up * self.activation_fn(gate))


class Glm4vGPTQ(BaseQModel):
    fusable = FusableConfig("glm4v").merge(LlamaFusableConfig(), profiles=("qkv",))

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForImageTextToText",
    )

    pre_lm_head_norm_module = ["model.language_model.norm", "model.norm"]

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

    @classmethod
    def after_defuser_conversion(cls, model):
        """Split a fused `Glm4vTextMLP.gate_up_proj` the Defuser pass left behind.

        The native Transformers 5.22 `Glm4vTextMLP` owns one fused
        `gate_up_proj` linear, while this definition declares `gate_proj` and
        `up_proj`. Defuser replaces the class when it can patch it before
        construction, but a model built directly with
        `loader.from_config()` (and the module-plan registry sweep) never gets
        that class patch, so the fused weight survives conversion. Strict
        module resolution then raises on the missing `mlp.gate_proj` before the
        first MLP projection is quantized.

        Split the fused weight with the same layout Defuser's
        `LinearGlm4vTextMLP` produces, and bind the split forward so the module
        remains runnable when Defuser did not replace the class. The pass is
        idempotent: a module that already has `gate_proj`/`up_proj` has no
        `gate_up_proj` and is skipped.
        """

        for module in list(model.modules()):
            if module.__class__.__name__ != "Glm4vTextMLP":
                continue
            gate_up = getattr(module, "gate_up_proj", None)
            down_proj = getattr(module, "down_proj", None)
            if not isinstance(gate_up, nn.Linear) or not isinstance(down_proj, nn.Linear):
                continue
            if gate_up.out_features % 2:
                continue
            half = gate_up.out_features // 2
            if down_proj.in_features != half:
                continue

            weight = gate_up.weight
            bias = gate_up.bias
            gate_proj = nn.Linear(
                gate_up.in_features,
                half,
                bias=bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            up_proj = nn.Linear(
                gate_up.in_features,
                half,
                bias=bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            gate_weight, up_weight = weight.chunk(2, dim=0)
            gate_proj.weight = nn.Parameter(gate_weight.clone(), requires_grad=False)
            up_proj.weight = nn.Parameter(up_weight.clone(), requires_grad=False)
            if bias is not None:
                gate_bias, up_bias = bias.chunk(2, dim=0)
                gate_proj.bias = nn.Parameter(gate_bias.clone(), requires_grad=False)
                up_proj.bias = nn.Parameter(up_bias.clone(), requires_grad=False)

            module.gate_proj = gate_proj
            module.up_proj = up_proj
            module.forward = _defused_glm4v_mlp_forward.__get__(module, type(module))
            del module.gate_up_proj
