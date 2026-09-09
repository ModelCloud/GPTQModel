"""QQQ deployed candidate values for GSQ; optimizer integration is separate.

QQQ grouped W4 codes undergo a second INT8 rounding step. Decode each hard
candidate before mixing probabilities; decoding an expected nibble would erase
both the true grid and its useful gradient.
"""

import torch


def qqq_candidate_values(codes, scales, *, group_size, in_features, channel_scales=None):
    """Decode [..., candidate] nibbles with weight axes [out,in].

    ``scales`` is the raw [out,groups] producer table. Grouped exports also
    require the FP32 per-output-channel scale. Returns FP32 effective weights,
    including the channel multiplier, before activation quantization. This
    models the Torch runtime grid, not a native-kernel accuracy guarantee.
    """
    if codes.ndim not in (2, 3) or codes.shape[1] != in_features or in_features <= 0:
        raise ValueError("QQQ codes must be [out,in] or [out,in,candidates]")
    if group_size == -1:
        group_size = in_features
    if group_size <= 0 or in_features % group_size:
        raise ValueError("QQQ group size must divide input width")
    if scales.shape != (codes.shape[0], in_features // group_size):
        raise ValueError("QQQ scales must be [out,groups]")
    if scales.device != codes.device or not scales.is_floating_point():
        raise ValueError("QQQ scales must be floating and on the codes device")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("QQQ scales must be finite and positive")
    if not torch.isfinite(codes).all() or (codes < 0).any() or (codes > 15).any():
        raise ValueError("QQQ nibbles must be finite and in [0,15]")
    if codes.is_floating_point() and (codes != codes.round()).any():
        raise ValueError("QQQ candidates must be integer nibbles")
    indices = torch.arange(in_features, device=codes.device) // group_size
    extra_axis = codes.ndim == 3
    values = codes.float()
    if group_size == in_features:
        # Packer divides in source scale precision before FP32 storage.
        channel = (scales / 16).float()
        if not torch.isfinite(channel).all() or (channel <= 0).any():
            raise ValueError("QQQ channel scales must remain finite and positive after storage")
        signed = torch.where(values >= 8, values - 16, values)
        multiplier = channel.unsqueeze(-1) if extra_axis else channel
        return signed * 16 * multiplier
    if channel_scales is None or channel_scales.numel() != codes.shape[0]:
        raise ValueError("grouped QQQ requires one channel scale per output")
    if channel_scales.device != codes.device or not channel_scales.is_floating_point():
        raise ValueError("QQQ channel scales must be floating and on the codes device")
    channel = channel_scales.reshape(-1, 1).float()
    if not torch.isfinite(channel).all() or (channel <= 0).any():
        raise ValueError("QQQ channel scales must be finite and positive")
    ratio = (scales / channel).half().float()
    if not torch.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError("QQQ group ratios must remain finite and positive in FP16")
    ratio = ratio[:, indices]
    if extra_axis:
        ratio = ratio.unsqueeze(-1)
        channel = channel.unsqueeze(-1)
    return ((values - 8) * ratio).round().clamp(-128, 127) * channel
