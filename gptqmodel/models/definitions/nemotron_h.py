# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils.logger import setup_logger
from ..base import BaseQModel


log = setup_logger()


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
    "mixer": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1", "in_proj:2", "out_proj:2", "gate_proj:3", "up_proj:3:nosrc", "down_proj:4"),
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
    require_trust_remote_code = True
    require_monkeypatch = True
    layer_modules_strict = False
    # Routed-expert cardinality lives on `n_routed_experts` (not `num_experts`).
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm_f"
    # require_pkgs = ["transformers<=4.48.3"]

    # Transformers 5.22 renamed the native tree root ``backbone`` -> ``model``
    # through the ``nemotron_h`` checkpoint conversion mapping. The definition
    # must describe the runtime tree, so it uses ``model`` directly.
    module_tree = [
        "model",
        "layers",
        "#",
        NEMOTRON_H_LAYER_MODULES,
    ]

    def monkey_patch(self):
        if not self.load_quantized_model:
            return

        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_attn_2_available,
            is_mamba_2_ssm_available,
        )

        if is_mamba_2_ssm_available():
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
        else:
            mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

        if is_causal_conv1d_available():
            from gptqmodel.hf_kernels.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        else:
            causal_conv1d_update, causal_conv1d_fn = None, None

        if is_flash_attn_2_available():
            pass

        is_fast_path_available = all(
            (
                selective_state_update,
                mamba_chunk_scan_combined,
                mamba_split_conv1d_scan_combined,
                causal_conv1d_fn,
                causal_conv1d_update,
            )
        )

        def forward(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            if is_fast_path_available and "cuda" in self.in_proj.qweight.device.type:
                return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
            dtype = hidden_states.dtype
            if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

            return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)

        # Native 5.22 exposes ``self.model.model.layers``; older remote code
        # exposed ``self.model.backbone.layers``.
        layers = None
        for root_name in ("model", "backbone"):
            root = getattr(self.model, root_name, None)
            candidate = getattr(root, "layers", None)
            if candidate is not None:
                layers = candidate
                break
        if layers is None:
            return

        for layer in layers:
            mixer = getattr(layer, "mixer", None)
            if mixer is None or mixer.__class__.__name__ != "NemotronHMamba2Mixer":
                continue
            mixer_class = type(mixer)
            if not all(hasattr(mixer_class, name) for name in ("torch_forward", "cuda_kernels_forward")):
                log.info.once(
                    "NemotronH Mamba2 mixer does not expose the legacy Torch/fast forward methods; "
                    "leaving the upstream forward in place because it selects kernels internally."
                )
                continue
            mixer_class.forward = forward
