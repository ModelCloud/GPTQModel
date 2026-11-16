# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...looper.linear_mode import LinearMode
from ...models._const import DEVICE, PLATFORM, normalize_device
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import TORCH_HAS_FUSED_OPS
from ._dequant_dtype import resolve_dequant_dtype
from ._fused_train_state import (
    capture_fused_train_state,
    refinalize_fused_state_after_apply,
    restore_fused_train_state,
)
from ._numerics import matmul_fp32_reduce
from ._runtime_cache import build_group_ret_idx, tensor_cache_key


log = setup_logger()

# ``aten::_weight_int4pack_mm_for_cpu`` accepts only these group sizes. A
# per-channel checkpoint (``group_size == -1``) is packed as one group per
# input row, so its resolved group size is ``in_features``. Declaring -1/16 as
# supported without this check let AUTO select this kernel on CPU for a
# checkpoint whose first forward then raised inside the aten op.
CPU_INT4PACK_GROUP_SIZES = (32, 64, 128, 256)
# ``aten::_convert_weight_to_int4pack_for_cpu`` requires the packed row width
# to be a multiple of 16.  The XPU path packs the same logical layout with a
# nibble pack factor of 8, so it keeps the smaller declaration and the CPU
# contract is checked separately.
CPU_INT4PACK_OUT_FEATURES_DIVISIBLE_BY = 16


def _validate_cpu_int4pack_out_features(
    cls,
    *,
    out_features: Optional[int],
    device: Optional[DEVICE | torch.device],
):
    """Reject a CPU output width the packed int4 kernel cannot consume."""
    if device is None or normalize_device(device) != DEVICE.CPU:
        return True, None
    if out_features is None or out_features % CPU_INT4PACK_OUT_FEATURES_DIVISIBLE_BY == 0:
        return True, None
    return False, NotImplementedError(
        f"{cls}: CPU int4pack requires out_features to be divisible by "
        f"{CPU_INT4PACK_OUT_FEATURES_DIVISIBLE_BY}: actual out_features = "
        f"`{out_features}`. Select `backend=\"gptq_torch\"` for this "
        "checkpoint on CPU."
    )


def _validate_cpu_int4pack_group_size(
    cls,
    *,
    group_size: int,
    in_features: Optional[int],
    device: Optional[DEVICE | torch.device],
):
    """Reject a CPU group size the packed int4 kernel cannot consume.

    XPU keeps the declared -1/16 support: its fused op is a different kernel
    with a different contract.
    """
    if device is None or normalize_device(device) != DEVICE.CPU:
        return True, None
    if group_size == -1:
        if in_features is None:
            # The model-wide selector runs before a module's in_features is
            # known. Per-module selection re-validates with the resolved size,
            # so defer the decision instead of rejecting every per-channel
            # checkpoint here.
            return True, None
        resolved = in_features
    else:
        resolved = group_size
    if resolved in CPU_INT4PACK_GROUP_SIZES:
        return True, None
    return False, NotImplementedError(
        f"{cls}: CPU int4pack only supports group_size "
        f"{list(CPU_INT4PACK_GROUP_SIZES)}, or a per-channel group equal to "
        f"in_features: resolved group_size = `{resolved}`. Select "
        "`backend=\"gptq_torch\"` for this checkpoint on CPU."
    )


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


@torch.compiler.disable
def _cpu_int4pack_mm(x, qweight, group_size, scales_and_zeros, out_dtype=None):
    """Run int4pack with FP32 operands without expanding stored weights.

    BF16 inputs, and FP16 inputs whose caller requests an FP32 product,
    accumulate in FP32. ``out_dtype`` lets a biased layer keep that product
    until the bias has been added, so the epilogue rounds once instead of
    overflowing or rounding the pre-bias sum.

    Inductor fuses the operand casts into a Pointwise node and then cannot
    lower ``aten::_weight_int4pack_mm_for_cpu`` on it (#4068). Disabling
    tracing here keeps the op and its casts eager inside an otherwise
    compiled model, which is a deliberate graph break rather than a silent
    fallback to a different kernel.
    """
    compute_dtype = (
        torch.float32
        if x.dtype == torch.bfloat16 or out_dtype == torch.float32
        else scales_and_zeros.dtype
    )
    out = torch.ops.aten._weight_int4pack_mm_for_cpu(
        x.to(compute_dtype), qweight, group_size, scales_and_zeros.to(compute_dtype)
    )
    return out.to(x.dtype if out_dtype is None else out_dtype)


class Int4PackedOp(torch.nn.Module):
    def __init__(self, qweight_uint8, scales_and_zeros, group_size, out_dtype=None):
        super().__init__()
        self.register_buffer("qweight_uint8", qweight_uint8, persistent=False)
        self.register_buffer("scales_and_zeros", scales_and_zeros, persistent=False)
        self.group_size = group_size
        self.out_dtype = out_dtype

    def forward(self, x):
        return _cpu_int4pack_mm(
            x,
            self.qweight_uint8,
            self.group_size,
            self.scales_and_zeros,
            out_dtype=self.out_dtype,
        )


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
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]
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

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = False,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE | torch.device] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
    ):
        ok, err = super()._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            device=device,
            trainable=trainable,
            adapter=adapter,
        )
        if not ok:
            return ok, err
        ok, err = _validate_cpu_int4pack_group_size(
            cls, group_size=group_size, in_features=in_features, device=device
        )
        if not ok:
            return ok, err
        return _validate_cpu_int4pack_out_features(
            cls, out_features=out_features, device=device
        )

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        super().optimize(backend=backend, mode=mode, fullgraph=fullgraph)

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

        The checkpoint snapshot is deliberately *not* restored here. This hook
        also runs for moves and explicit cache clears, where the caller expects
        the live packed buffers to stay authoritative. Restoring the snapshot
        would overwrite uncommitted metadata changes and put the checkpoint
        layout back into the inference kernel. Only an explicit checkpoint load,
        export, or training resume restores it.
        """
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

    def _apply(self, fn, recurse: bool = True):
        """Re-finalize the checkpoint layout after an invalidating move.

        The runtime int4pack layout is device- and scale-dtype-coupled, so a
        cross-device or low-precision move restores the checkpoint snapshot and
        lets the next forward transform the new target. A same-device move to
        FP32 keeps the live layout because it only widens the offsets.
        """

        scales = getattr(self, "scales", None)
        transformed = self.linear_mode == LinearMode.INFERENCE and isinstance(scales, torch.Tensor)
        # The packed layout is keyed by the activation dtype it was built for,
        # not by ``scales.dtype``: the checkpoint scale buffer now keeps its
        # loaded dtype while the affine metadata follows the activation dtype.
        runtime_dtype = getattr(self, "_cpu_transform_dtype", None) if transformed else None
        if runtime_dtype is None and transformed:
            runtime_dtype = scales.dtype
        device = scales.device if transformed else None
        result = super()._apply(fn, recurse=recurse)
        refinalize_fused_state_after_apply(self, runtime_dtype, device)
        return result

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
        # ``compute_scales`` is the packed runtime metadata; ``self.scales``
        # keeps the checkpoint dtype so a later forward or save cannot persist
        # a narrowed scale (issue #4178).
        compute_scales = self.scales.to(compute_dtype).contiguous()
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
            self.qzeros = _cpu_int4pack_zero_offsets(zero_codes, compute_scales, self.bits).to(compute_dtype).contiguous()
            self.scales_and_zeros = pack_scales_and_zeros(compute_scales, self.qzeros)
        self._cpu_transform_dtype = dtype

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
        # The CPU fused op now returns FP32 when a bias is present, so both
        # FP16 and BF16 can add the bias before the single output cast.
        fused_bias_safe = self.bias is None or x.dtype in (torch.bfloat16, torch.float16)
        if (
            not self.training
            and not x.requires_grad
            and self.linear_mode is None
            and TORCH_HAS_FUSED_OPS
            and fused_bias_safe
        ):
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
            out = self._fused_op_forward(x)
            if self.bias is not None:
                # BF16 has FP32's range; add its bias before the visible cast.
                out = (out.float() + self.bias.to(out.device, torch.float32)).to(x.dtype)
            out = out.reshape(out_shape)
        else:
            # make sure dequant dtype matches input x
            num_itr = self.g_idx.shape[0] // x.shape[-1]
            weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
            if self.bias is not None:
                out = matmul_fp32_reduce(x, weights, self.bias).reshape(out_shape)
            else:
                out = torch.matmul(x, weights).reshape(out_shape)

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
            # A biased FP16 layer keeps the FP32 product until the bias is
            # added, then casts once; BF16 already has FP32's range.
            self.torch_fused_op.out_dtype = (
                torch.float32
                if self.bias is not None and x.dtype in (torch.float16, torch.bfloat16)
                else None
            )
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
    # Dense weights must match the model's compute dtype.  The checkpoint
    # scale dtype is not a compute-dtype signal: GPTQ checkpoints store FP16
    # scales even for BF16 models, so using it produced FP16 layers inside a
    # BF16 model and crashed the first matmul.
    target_dtype = None
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchFusedLinear):
            raise ValueError(
                "Only models loaded using TorchFusedLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.GPTQ_TORCH_FUSED"
            )

        if isinstance(module, TorchFusedLinear):
            if target_dtype is None:
                scales = getattr(module, "scales", None)
                target_dtype = resolve_dequant_dtype(
                    model,
                    fallback=(
                        scales.dtype
                        if scales is not None and scales.is_floating_point()
                        else torch.float16
                    ),
                )
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            new_module.weight = nn.Parameter(
                module.dequantize_weight().T.detach().to("cpu", target_dtype)
            )
            # ``nn.Parameter(None)`` is a shape-[0] tensor, not an absent
            # bias. Every bias-less layer then failed in forward.
            new_module.bias = (
                nn.Parameter(module.bias.detach().to("cpu", target_dtype))
                if module.bias is not None
                else None
            )

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
