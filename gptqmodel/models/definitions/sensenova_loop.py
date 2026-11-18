# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from ..looped_capture import LoopedLayerCaptureMixin
from .qwen3_5 import Qwen3_5QModel


class SenseNovaLoopQModel(LoopedLayerCaptureMixin, Qwen3_5QModel):
    """GPTQ definition for OpenSenseNova's SenseNova-Loop decoder.

    `sensenova/SenseNova-Loop-9B` declares `model_type = "sensenova_loop"` and
    ships `modeling_sensenova_loop.py`. GPT-QModel builds it from the
    package-owned port in `gptqmodel.models.ports.sensenova_loop` instead, so
    neither `trust_remote_code` nor the checkpoint `auto_map` is used. The
    physical layer stack is Qwen3.5's VL layout (`model.language_model.layers`,
    `model.visual`), so this definition declares the same tree as
    `Qwen3_5QModel`; the loop only changes *how often* those physical layers
    run, not what they contain.

    Two SenseNova-Loop features matter for quantization:

    * **Looped block.** `loop_settings = {start: 4, end: 28, eval_count: 2}`
      executes physical layers 4..27 twice and reuses the *same* physical layer
      objects for both visits. One quantized copy per physical layer is
      therefore correct, and the calibration Hessian accumulates over both
      visits. The default capture aborts the forward at the first layer, so it
      would only ever see a layer's first visit and the post-loop layers would
      be calibrated on one-loop inputs; this definition therefore opts into
      `LoopedLayerCaptureMixin`, which runs one full forward per sample and
      replays every invocation of the layer being quantized.
    * **mHC residual-stream routers.** `mhc = {start: 4, end: 28, streams: 2}`
      gives every layer in 4..27 an `mhc_attn` / `mhc_ffn` `MHCRouter`. These
      are not `nn.Linear`: each holds a raw FP32 `weight` (`[2*streams +
      streams**2, streams*hidden]`) plus `bias`, the router forward raises if
      they are not FP32, and the checkpoint stores them as FP32
      (`_keep_in_fp32_modules[_strict] = ["mhc_attn", "mhc_ffn"]`). They are absent from
      the module tree so they are never quantized, and `should_quantize_module`
      keeps them dense even if a future checkpoint ships them as `nn.Linear`.

    Auxiliary `mtp.*` tensors (the checkpoint ships `mtp.safetensors`) and
    the adaptive-count `model.language_model.loop_gate.logits` tensor listed
    in the released index are ignored by the fixed-count model itself and are
    preserved verbatim here.
    """

    out_of_model_tensors = {"prefixes": ["mtp", "model.language_model.loop_gate"]}

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.sensenova_loop.SenseNovaLoopConfig",
        model_class="gptqmodel.models.ports.sensenova_loop.SenseNovaLoopForConditionalGeneration",
        processor_class="gptqmodel.models.ports.sensenova_loop.Qwen3VLProcessor",
        tokenizer_class="gptqmodel.models.ports.sensenova_loop.Qwen2Tokenizer",
    )

    require_trust_remote_code = False

    # Fusion is withheld: the mHC decoder layer runs its own routed forward,
    # so admitting q/k/v or gate/up roles needs a separate qualification.
    fusable = None

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "q_proj:0:in=x", "k_norm:!", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            "linear_attn:!": (
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

    @classmethod
    def should_quantize_module(cls, model, module_name, module, quantize_config):
        """Keep the FP32-only mHC routers dense on load and re-save."""

        if ".mhc_attn" in module_name or ".mhc_ffn" in module_name:
            return False

        return super().should_quantize_module(model, module_name, module, quantize_config)


__all__ = ["SenseNovaLoopQModel"]
