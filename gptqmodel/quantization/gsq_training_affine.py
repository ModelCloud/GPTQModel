"""Affine scalar parameterization for staged training; lifecycle integration is separate."""

import torch

from .gsq_training import GSQScalarTrainingModule


class GSQAffineTrainingModule(GSQScalarTrainingModule):
    """Train assignments/scales while retaining fixed per-group integer zero points.

    Accept integer codes from an existing initializer, not a replacement GPTQ
    prior. The inherited relaxation differentiates (code - zero) * scale.
    This is a parameterization adapter, not a complete AWQ training/export path.
    """

    def __init__(self, codes, scales, zeros, group_size, *, bits, noise, **kwargs):
        if isinstance(bits, bool) or bits not in (2, 3, 4):
            raise ValueError('Affine staged GSQ requires W2/W3/W4')
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError('Affine staged GSQ requires positive group size')
        if (codes.ndim != 2 or not torch.isfinite(codes).all()
                or not torch.equal(codes, codes.round()) or (codes < 0).any() or (codes >= 2**bits).any()):
            raise ValueError('Affine staged GSQ requires legal integer codes')
        shape = (codes.shape[0], (codes.shape[1]+group_size-1)//group_size)
        if (scales.shape != shape or zeros.shape != shape or scales.device != codes.device
                or zeros.device != codes.device or not scales.is_floating_point()
                or not torch.isfinite(scales).all() or (scales <= 0).any()
                or not torch.isfinite(zeros).all() or not torch.equal(zeros, zeros.round())
                or (zeros < 0).any() or (zeros >= 2**bits).any()):
            raise ValueError('Affine staged GSQ requires matching positive scales and legal integer zeros')
        # Unit scales initialize exact code candidates, avoiding divide/multiply
        # roundoff when recovering codes from dequantized initializer weights.
        centered = codes.to(scales.dtype) - 2**(bits-1)
        super().__init__(centered, torch.ones_like(scales), group_size,
                         bits=bits, noise=noise, **kwargs)
        self.scales = torch.nn.Parameter(scales.detach().float().clone())
        self.bits = bits
        self.register_buffer('zeros', zeros.detach().clone())
        offset = zeros[:, self.group_index].to(centered.dtype) - 2**(bits-1)
        self.candidates = self.candidates - offset.unsqueeze(0)
        if self.initial is not None:
            self.initial.sub_(offset)

    @torch.no_grad()
    def hard_codes(self):
        selected = self.logits.masked_fill(~self.valid, -torch.inf).argmax(0, keepdim=True)
        return self.candidates.gather(0, selected).squeeze(0) + self.zeros[:, self.group_index]

    @torch.no_grad()
    def export_affine(self, *, packing, scale_dtype=torch.float16):
        """Prepare stored-scale weights and reject nonrepresentable hard exports.

        This checks producer arithmetic; actual backend packing/reload remains
        a separate lifecycle requirement. Never clamp failed learned scales.
        """
        return prepare_affine_export(self.hard_codes(), self.scales, self.zeros, self.group_index,
                                     bits=self.bits, packing=packing, scale_dtype=scale_dtype)


@torch.no_grad()
def prepare_affine_export(codes, scales, zeros, group_index, *, bits, packing, scale_dtype=torch.float16):
    """Convert validated staged affine records into representable packing inputs."""
    from .gsq_scalar import affine_codes

    if bits not in (2, 3, 4) or isinstance(bits, bool):
        raise ValueError('Affine staged export requires W2/W3/W4')
    if (codes.ndim != 2 or scales.ndim != 2 or scales.shape != zeros.shape
            or scales.shape[0] != codes.shape[0] or group_index.shape != (codes.shape[1],)
            or group_index.dtype not in (torch.int32, torch.int64)
            or any(t.device != codes.device for t in (scales, zeros, group_index))
            or (group_index < 0).any() or (group_index >= scales.shape[1]).any()):
        raise ValueError('Invalid affine export geometry')
    if any(not torch.isfinite(t).all() or not torch.equal(t, t.round())
           or (t < 0).any() or (t >= 2**bits).any() for t in (codes, zeros)):
        raise ValueError('Invalid affine export codes or zero points')
    if packing not in ('gptq', 'awq_gemm') or (packing == 'awq_gemm' and bits != 4):
        raise ValueError('Affine staged export supports GPTQ or W4 AWQ GEMM')
    if scale_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('Affine staged export requires FP16/BF16 scale storage')
    scales = scales.detach().to(scale_dtype).clone()
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Learned affine scales are not positive finite values in storage dtype')
    codes = codes.float()
    zeros = zeros.detach().float().clone()
    groups = group_index.to(torch.int32)
    weight = (codes-zeros[:, groups])*scales[:, groups].float()
    recovered = affine_codes(weight, scales, zeros, groups, bits,
                             packing=packing, scale_dtype=scale_dtype)
    if not torch.isfinite(weight).all() or not torch.equal(recovered, codes):
        raise ValueError('Affine staged hard assignments do not survive producer arithmetic')
    return dict(weight=weight, scales=scales, zeros=zeros, g_idx=groups, codes=codes)


def pack_llama_affine_stages(fitted, records, *, group_size, scale_dtype=torch.float16):
    """Pack all seven fitted affine projections through the CPU AWQ GEMM path.

    Caller owns transformed teacher capture, AWQ initialization and replay.
    The supplied fitted block and records are never modified.
    """
    import copy

    from ..nn_modules.qlinear.torch_awq import AwqTorchLinear

    exported = copy.deepcopy(fitted).cpu()
    names = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    for name in names:
        stage = name if name in records else ('attention' if name.startswith('self_attn.') else 'mlp')
        key = 'weight' if stage == name else name+'.weight'
        record = records[stage]
        linear = exported.get_submodule(name)
        groups = torch.arange(linear.in_features, dtype=torch.int32)//group_size
        inputs = prepare_affine_export(record['codes'][key].cpu(), record['scales'][key].cpu(),
                                       record['zeros'][key].cpu(), groups, bits=4,
                                       packing='awq_gemm', scale_dtype=scale_dtype)
        linear = linear.float()
        with torch.no_grad():
            linear.weight.copy_(inputs['weight'])
        packed = AwqTorchLinear(bits=4, group_size=group_size, sym=False, desc_act=False,
                                in_features=linear.in_features, out_features=linear.out_features,
                                bias=linear.bias is not None)
        packed.pack(linear, inputs['scales'], inputs['zeros'])
        parent, leaf = name.rsplit('.', 1)
        setattr(exported.get_submodule(parent), leaf, packed)
    return exported
