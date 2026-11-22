# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class _NativeLoader(LocalModelLoader):
    def __get__(self, instance, owner):
        # Puzzle and Omni own different architectures and their own migration.
        return self if owner is NemotronHQModel else BaseQModel.loader


class _LegacySiblingPolicy:
    def __get__(self, instance, owner):
        return owner is not NemotronHQModel


class _NativeExtraTensors:
    def __get__(self, instance, owner):
        return {"prefixes": ["mtp"]} if owner is NemotronHQModel else None


# Decoder-layer declaration for the Nemotron-H mixer family.
#
# The language backbone of the multimodal Nemotron Omni variant is the same
# Nemotron-H decoder, mounted under a different decoder prefix. Both
# definitions must publish the identical layer inventory: a change that only
# reaches one of them leaves every routed expert, shared expert, router and SSM
# convolution of the other in the source dtype with no error (issue #4355).
NEMOTRON_H_LAYER_MODULES = {
    "norm": ("norm:!",),
    # NemotronH's MLP is `down_proj(act(up_proj(x)))` with no gate, and
    # the default activation is `relu2`. `relu(x)^2` is not positively
    # homogeneous, so `up_proj`'s output is not `down_proj`'s input
    # scale source (fold error 2.2e-2).
    # `out_proj` consumes the SSM output produced from `in_proj`, so it must be
    # a later subset. A layer has one mixer type, so sharing index 3 with the
    # dense MLP gate/up (which never coexist with `out_proj`) is free.
    "mixer": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "in_proj:2", "out_proj:3", "gate_proj:3", "up_proj:3:nosrc", "down_proj:4"),
    # The SSM conv is checkpoint-owned but is not a quantizable Linear.
    "mixer.conv1d": ("conv1d:!",),
    # Routed top-k router keeps its checkpoint dtype.
    "mixer.gate": ("gate:!",),
    # Shared experts are dense Linears; declare them so they do not stay
    # in the source dtype just because the MoE root is nested. They read the
    # block input norm, and their own `up_proj` is not a fold source: the
    # expert MLP is `down_proj(relu2(up_proj(x)))`, the same non-homogeneous
    # activation that makes the dense `mixer.up_proj` unsafe.
    "mixer.shared_experts": ("up_proj:0:nosrc", "down_proj:1"),
    # Latent-MoE layouts (`moe_latent_size` set) own a low-rank pair around the
    # routed experts. The native forward is `experts(fc1_latent_proj(x))`
    # followed by `fc2_latent_proj`, so the declarations must keep that order
    # for sequential quantization. Distinct group ids avoid merging these
    # blocks into the Mamba/attention groups above.
    "mixer.fc1_latent_proj:5": ("fc1_latent_proj:0",),
    # Routed experts are the same `relu2` MLP, so `up_proj` is not a source
    # here either.
    "mixer.experts:moe": {
        "#": ("up_proj:0:nosrc", "down_proj:1"),
    },
    "mixer.fc2_latent_proj:6": ("fc2_latent_proj:0",),
}


class NemotronHQModel(BaseQModel):
    loader = _NativeLoader(
        config_class="gptqmodel.models.ports.nemotron_h.NemotronHConfig",
        model_class="gptqmodel.models.ports.nemotron_h.NemotronHForCausalLM",
    )
    require_trust_remote_code = _LegacySiblingPolicy()
    require_monkeypatch = _LegacySiblingPolicy()
    out_of_model_tensors = _NativeExtraTensors()
    layer_modules_strict = False
    # Routed-expert cardinality lives on `n_routed_experts` (not `num_experts`).
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm_f"

    # Transformers 5.22 renamed the native tree root ``backbone`` -> ``model``
    # through the ``nemotron_h`` checkpoint conversion mapping. The definition
    # must describe the runtime tree, so it uses ``model`` directly.
    module_tree = [
        "model",
        "layers",
        "#",
        NEMOTRON_H_LAYER_MODULES,
    ]

    @classmethod
    def _resolve_effective_module_tree(cls, model, quantize_config, detector=None):
        tree = super()._resolve_effective_module_tree(model, quantize_config, detector=detector)
        if cls is NemotronHQModel and getattr(getattr(model, "config", None), "model_type", None) == "nemotron_h":
            tree[-1]["mixer"] = tuple(dict.fromkeys((*tree[-1]["mixer"], "A_log:!", "D:!", "dt_bias:!")))
            tree[-1]["mixer.norm"] = ("norm:!",)
        return tree

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        if type(self) is NemotronHQModel:
            # SSM recurrence/gating and routed/shared latent MoE need a joint
            # replay objective before any affine fold can be qualified.
            return []
        return super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)

    def monkey_patch(self):
        """Compatibility no-op: native Transformers owns optional-kernel dispatch."""
