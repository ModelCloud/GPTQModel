# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPT-QModel support for the text/VLM portion of Qwen-Drive.

The released Qwen-Drive checkpoint stores its Qwen3.5 VLM under ``vlm``.  The
planning expert and perception model are separate artifacts, so this
definition deliberately quantizes and delegates only the VLM text path.
"""

from transformers import AutoModel

from .qwen3_5 import Qwen3_5QModel


class QwenDriveQModel(Qwen3_5QModel):
    """Quantization definition for the ``vlm.*`` part of Qwen-Drive."""

    loader = AutoModel

    # The official package is imported by the GPT-QModel HF registration hook.
    # Qwen-Drive's top-level model is natively registered there and does not
    # use Transformers' trust_remote_code mechanism.
    require_trust_remote_code = False
    require_load_processor = False
    # The official package is imported by ``ensure_qwen_drive_registered``.
    # Do not use ``require_pkgs`` here: a source checkout made importable via
    # PYTHONPATH has no distribution metadata for ``check_versions`` to read.

    layer_modules_strict = False

    lm_head = "vlm.lm_head"
    pre_lm_head_norm_module = "vlm.model.language_model.norm"
    rotary_embedding = "vlm.model.language_model.rotary_emb"

    # This is the Qwen3.5 decoder topology with the additional ``vlm`` root
    # used by QwenDriveForPlanning.  Vision modules and the planning expert
    # remain base/pass-through modules and are not quantized by this tree.
    module_tree = [
        "vlm",
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "q_proj:0:in=x",
                "k_norm:!",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "linear_attn": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0:in=x",
                "in_proj_z:1:in=x",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        },
    ]

    def after_model_load(self, model, load_quantized_model=False):
        # QwenDriveForPlanning constructs this module from config even when
        # loading the root VLM checkpoint.  The released root safetensors do
        # not contain planner tensors, and planning is intentionally outside
        # this quantization definition.  Remove the module before the lazy
        # source, materialization, or save logic can observe it.
        del load_quantized_model
        modules = getattr(model, "_modules", None)
        if isinstance(modules, dict):
            modules.pop("planning_expert", None)
        # StageInputsCapture invokes the wrapped HF model directly while it
        # intercepts the first decoder layer.  The official planning wrapper
        # has no generic forward method, so make its supported root-VLM path
        # explicit for calibration as well as ordinary inference.
        model.forward = model.vlm.forward
        return model

    def forward(self, *args, **kwargs):
        """Run the text/VLM forward path, excluding the absent planner."""

        return self.model.vlm(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate text generation to the wrapped Qwen3.5 VLM."""

        return self.model.vlm.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.vlm.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.vlm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.vlm.get_output_embeddings()


__all__ = ["QwenDriveQModel"]
