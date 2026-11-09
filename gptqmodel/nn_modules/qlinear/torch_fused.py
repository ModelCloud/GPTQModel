# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...looper.linear_mode import LinearMode
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import TORCH_HAS_FUSED_OPS
from ._fused_train_state import capture_fused_train_state, restore_fused_train_state
from ._runtime_cache import build_group_ret_idx, tensor_cache_key


log = setup_logger()

# TODO: CPU works, not yet working for cuda fused int4 ops
def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    # assert scales.dtype == torch.bfloat16
    # assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .contiguous()
    )


def _cpu_int4pack_mm(x, qweight, group_size, scales_and_zeros):
    """Run BF16 int4pack with FP32 operands without expanding stored weights."""
    compute_dtype = torch.float32 if x.dtype == torch.bfloat16 else scales_and_zeros.dtype
    out = torch.ops.aten._weight_int4pack_mm_for_cpu(
        x.to(compute_dtype), qweight, group_size, scales_and_zeros.to(compute_dtype)
    )
    return out.to(x.dtype)


class Int4PackedOp(torch.nn.Module):
    def __init__(self, qweight_uint8, scales_and_zeros, group_size):
        super().__init__()
        self.register_buffer("qweight_uint8", qweight_uint8, persistent=False)
        self.register_buffer("scales_and_zeros", scales_and_zeros, persistent=False)
        self.group_size = group_size

    def forward(self, x):
        return _cpu_int4pack_mm(x, self.qweight_uint8, self.group_size, self.scales_and_zeros)


class TorchFusedLinear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH_FUSED]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 50, FORMAT.GPTQ_V2: 50}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.XPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch_fused"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TORCH_FUSED),
            adapter=adapter,
            register_buffers=register_buffers,
            enable_wf_unsqueeze=kwargs.pop("enable_wf_unsqueeze", True),
            **kwargs)

        self.linear_mode = None # either train or inference
        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return

        super().optimize()

    def clear_weight_cache(self) -> None:
        """Drop derived state that depends on the loaded packed buffers.

        ``ret_idx`` is indexing metadata, not a packed weight, but it is only
        valid for the ``g_idx`` it was built from. ``load_state_dict`` copies
        into the existing buffers, so neither the object identity nor the
        version of ``g_idx`` is guaranteed to change; the base class calls this
        hook on load and on device/dtype moves instead.

        Once transformed, the fused op also owns a ``scales_and_zeros`` tensor
        derived from the checkpoint's scales and zero points. A state-dict load
        copies those buffers in place, leaving that derived tensor stale. The
        packed ``qweight`` stays valid, so ``linear_mode`` is preserved and the
        op is simply rebuilt from the current packed weight and metadata; no
        dense weight is created.
        """
        if restore_fused_train_state(self):
            self.linear_mode = None
            self.torch_fused_op = None
            self.scales_and_zeros = None

        self.ret_idx = None
        self._ret_idx_key = None
        self._ret_idx_identity = None
        self._fused_state_key = None
        self._refresh_fused_op()

    def _refresh_fused_op(self) -> None:
        """Rebuild the packed fused op after its source buffers changed."""
        if getattr(self, "linear_mode", None) != LinearMode.INFERENCE:
            return
        qweight = getattr(self, "qweight", None)
        scales = getattr(self, "scales", None)
        qzeros = getattr(self, "qzeros", None)
        if qweight is None or scales is None or qzeros is None:
            return
        if qweight.device.type != "cpu":
            # The xpu path reads ``scales``/``qzeros`` directly; nothing derived.
            return
        key = tuple(tensor_cache_key(tensor) for tensor in (qweight, scales, qzeros))
        if getattr(self, "_fused_state_key", None) == key:
            return
        # Preserve the original compute precision, but do not narrow source
        # metadata replaced by a load, move, or mutation at a wider dtype.
        current = getattr(self, "scales_and_zeros", None)
        dtype = current.dtype if isinstance(current, torch.Tensor) else scales.dtype
        dtype = torch.promote_types(dtype, torch.promote_types(scales.dtype, qzeros.dtype))
        self.scales_and_zeros = pack_scales_and_zeros(scales.to(dtype), qzeros.to(dtype))
        self.torch_fused_op = Int4PackedOp(
            self.qweight, self.scales_and_zeros, self.group_size
        ).eval()
        self._fused_state_key = key

    def _build_ret_idx(self) -> torch.Tensor:
        key = tensor_cache_key(self.g_idx)
        existing = getattr(self, "ret_idx", None)
        total = self.g_idx.shape[0]
        if (
            isinstance(existing, torch.Tensor)
            and existing.numel() == total
            and getattr(self, "_ret_idx_key", None) == key
        ):
            return existing

        ret_idx = build_group_ret_idx(self.g_idx)
        self.ret_idx = ret_idx
        # The permutation changed, so any cached identity verdict for the
        # previous ``g_idx`` is stale. Callers may rebuild ``ret_idx``
        # explicitly before the next forward; the base class recomputes the
        # verdict from the new key.
        self._ret_idx_identity = None
        self._ret_idx_key = key
        return ret_idx

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from ...utils.model import convert_gptq_v1_to_v2_format_module

        if mode and self.linear_mode == LinearMode.INFERENCE:
            restore_fused_train_state(self)
            self.linear_mode = None
            self.torch_fused_op = None
            self.scales_and_zeros = None

        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            # training starts
            if mode:
                # one time clone v1 qzeros and save both v1 and v2 qzeros in memory
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)

            # training switching to inference/eval
            else:
                if hasattr(self, "qzeros_data_v1"):
                    # switch qzero back to v1 for inference/eval
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)

        return super().train(mode=mode)

    def transform_xpu(self, dtype):
        self.scales = self.scales.to(dtype).contiguous()
        # Unpack qzeros
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(zeros.shape[0], -1)
        # Unpack and reorder qweight
        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq
        )
        ret_idx = self._build_ret_idx()
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]).index_select(0, ret_idx).t()
        # Pack qweight
        packed = torch.zeros(weight.shape[0], weight.shape[1] // self.pack_factor, dtype=torch.int32, device=weight.device)
        for col in range(weight.shape[1] // self.pack_factor):
            for i in range(self.pack_factor):
                packed_col = weight[:, col * self.pack_factor + i].to(torch.int32)
                packed[:, col] |= packed_col << (i * self.bits)

        self.qweight = packed.contiguous()
        self.qzeros = zeros.contiguous()

    def transform_cpu(self, dtype, do_scales_and_zeros: bool = True):
        if do_scales_and_zeros:
            # The CPU op consumes affine offsets, not GPTQ integer zero codes.
            # Import here because torch_aten_kernel also imports the fused op.
            from .torch_aten_kernel import _cpu_int4pack_scale_dtype, _cpu_int4pack_zero_offsets

            zero_codes = torch.bitwise_right_shift(
                torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
                self.wf_unsqueeze_zero,
            ).to(torch.uint8)
            zero_codes = torch.bitwise_and(zero_codes, self.maxq).reshape(self.scales.shape)
            compute_dtype = _cpu_int4pack_scale_dtype(self.scales, zero_codes, self.bits, dtype)
        else:
            compute_dtype = dtype
        self.scales = self.scales.to(compute_dtype).contiguous()
        # Unpack and reorder qweight
        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq
        )
        ret_idx = self._build_ret_idx()
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]).index_select(0, ret_idx).t()
        self.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(weight.int(), 1).contiguous()

        if do_scales_and_zeros:
            self.qzeros = _cpu_int4pack_zero_offsets(zero_codes, self.scales, self.bits).to(compute_dtype).contiguous()
            self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)

    def transform(self, dtype, device):
        if device == "xpu":
            self.transform_xpu(dtype)
        elif device == "cpu":
            self.transform_cpu(dtype)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        input_dtype = x.dtype
        if input_dtype == torch.float32:
            x = x.to(torch.bfloat16)
        if not self.training and not x.requires_grad and self.linear_mode is None and TORCH_HAS_FUSED_OPS:
            # one-time transform per module for xpu aten fused ops
            capture_fused_train_state(self)
            self.transform(x.dtype, x.device.type)
            self.linear_mode = LinearMode.INFERENCE
            if x.device.type == "cpu":
                self._refresh_fused_op()
                # self.torch_fused_op.forward = torch.compile(
                #     self.torch_fused_op.forward, options={"max_autotune": True}
                # )
        elif self.linear_mode is None:
            self.linear_mode = LinearMode.TRAIN

        if self.linear_mode == LinearMode.INFERENCE:
            out = self._fused_op_forward(x).reshape(out_shape)
        else:
            # make sure dequant dtype matches input x
            num_itr = self.g_idx.shape[0] // x.shape[-1]
            weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
            out = torch.matmul(x, weights).reshape(out_shape)

        # Add bias and adapter
        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        if input_dtype == torch.float32:
            out = out.to(torch.float32)

        return out

    def _needs_activation_gather(self, device_type: str) -> bool:
        """Whether ``_fused_op_forward`` must gather with ``ret_idx``.

        XPU keeps the main-branch behavior and always gathers, because that
        path has no coverage for activation views with a non-zero storage
        offset. CPU may skip the gather when the permutation is the identity.
        """
        return device_type == "xpu" or not self._ret_idx_is_identity()

    @torch.no_grad
    def _fused_op_forward(self, x):
        self._refresh_fused_op()
        # The identity fast path is CPU-only. The XPU kernel has no coverage for
        # activation views with a non-zero storage offset, so keep the
        # main-branch unconditional gather (and its zero-offset copy) there.
        if self._needs_activation_gather(x.device.type):
            x = x[:, self._build_ret_idx().to(x.device)].contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()
        # fused ops optimized for xpu using torch.ops
        # note _weight_int4pack_mm_with_scales_and_zeros is added by intel for xpu only
        if x.device.type == "xpu":
            out = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
                x, self.qweight, self.group_size, self.scales, self.qzeros
            )
        elif x.device.type == "cpu":
            out = self.torch_fused_op(x)
        else:
            raise NotImplementedError

        return out

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchFusedLinear):
            raise ValueError(
                "Only models loaded using TorchFusedLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.GPTQ_TORCH_FUSED"
            )

        if isinstance(module, TorchFusedLinear):
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = torch.nn.Parameter(module.bias)

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = ["TorchFusedLinear", "dequantize_model"]
