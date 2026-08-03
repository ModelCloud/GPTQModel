# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Tests for the combined SUPPORTS_FORMAT_BIT_MAP capability declaration and the
continuous gptq_v2 3-bit -> planar relayout (`convert_to_planar`) used by Pangolin."""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear import BaseQuantLinear, FormatSupport
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT


torch_cuda_available = torch.cuda.is_available()


def _make_continuous_module(bits: int = 3, in_features: int = 256, out_features: int = 128,
                            group_size: int = 32, seed: int = 0, cls=TorchLinear,
                            sym: bool = False):
    torch.manual_seed(seed + bits)
    maxq = (1 << bits) - 1
    groups = in_features // group_size
    linear = nn.Linear(in_features, out_features, bias=True)
    scales = torch.rand(out_features, groups) * 0.01 + 0.005
    if sym:
        # sym packing centers zeros at 2^(bits-1), matching the continuous
        # 3-bit fused-path requirement checked in TritonV2Linear.post_init.
        zeros = torch.full((out_features, groups), float((maxq + 1) // 2))
    else:
        zeros = torch.randint(0, maxq + 1, (out_features, groups)).float()
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    module = cls(
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        format=FORMAT.GPTQ_V2,
        register_buffers=False,
    )
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone())
    # pack_block stores the raw zero codes, i.e. v2 (no +1 bias) semantics.
    module.qzero_format(format=2)
    return module


class TestFormatBitMapDeclaration:
    def test_derived_legacy_attributes(self):
        assert TritonV2Linear.SUPPORTS_BITS == [2, 3, 4, 5, 6, 7, 8]
        assert TritonV2Linear.SUPPORTS_FORMATS == {
            FORMAT.GPTQ: 40, FORMAT.GPTQ_V2: 40, FORMAT.GPTQ_P: 40,
        }
        assert TorchLinear.SUPPORTS_BITS == [2, 3, 4, 5, 6, 7, 8]

    def test_supported_bits_per_format(self):
        assert TritonV2Linear.supported_bits(FORMAT.GPTQ_P) == (2, 3, 4, 5, 6, 7, 8)
        assert TritonV2Linear.supported_bits(FORMAT.GPTQ_V2) == (2, 3, 4, 8)
        assert TritonV2Linear.supported_bits(FORMAT.GPTQ) == (2, 3, 4, 8)
        # unknown / unspecified format falls back to the union
        assert TritonV2Linear.supported_bits(None) == (2, 3, 4, 5, 6, 7, 8)

    def test_all_kernels_derive_consistent_attributes(self):
        from gptqmodel.utils.importer import iter_quant_linear_kernels

        for cls in iter_quant_linear_kernels():
            # Subclasses may inherit a map but override SUPPORTS_FORMATS (e.g.
            # priority-0 opt-outs); only classes declaring the map derive both.
            fbm = cls.__dict__.get("SUPPORTS_FORMAT_BIT_MAP")
            if not fbm:
                continue
            assert cls.SUPPORTS_FORMATS == {fmt: fs.priority for fmt, fs in fbm.items()}
            assert cls.SUPPORTS_BITS == sorted({b for fs in fbm.values() for b in fs.bits})
            for fs in fbm.values():
                assert isinstance(fs, FormatSupport)
                assert isinstance(fs.priority, int)
                assert all(isinstance(b, int) for b in fs.bits)

    def test_validate_rejects_format_bit_mismatch(self):
        common = dict(
            group_size=32, desc_act=False, sym=False, pack_dtype=torch.int32,
            in_features=256, out_features=128,
        )
        ok, err = TritonV2Linear._validate(bits=6, format=FORMAT.GPTQ_V2, **common)
        assert not ok
        assert isinstance(err, NotImplementedError)
        assert "bits" in str(err)

        ok, err = TritonV2Linear._validate(bits=6, format=FORMAT.GPTQ_P, **common)
        assert ok, err

        ok, err = TritonV2Linear._validate(bits=3, format=FORMAT.GPTQ_V2, **common)
        assert ok, err

    def test_validate_rejects_planar_only_bits_without_format(self):
        ok, err = TorchLinear._validate(
            bits=5, group_size=32, desc_act=False, sym=False, pack_dtype=torch.int32,
            in_features=256, out_features=128, format=FORMAT.GPTQ,
        )
        assert not ok
        assert isinstance(err, NotImplementedError)
        assert "bits" in str(err)

    def test_base_class_default_map_is_none(self):
        assert BaseQuantLinear.SUPPORTS_FORMAT_BIT_MAP is None

    def test_all_kernels_declare_resolvable_map(self):
        from gptqmodel.utils.importer import iter_quant_linear_kernels

        for cls in iter_quant_linear_kernels():
            fbm = cls.SUPPORTS_FORMAT_BIT_MAP
            assert isinstance(fbm, dict) and fbm, f"{cls.__name__} has no SUPPORTS_FORMAT_BIT_MAP"
            # verify_supports_params requires the map in cls.__dict__ (inheritance
            # does not count), so every concrete kernel must declare its own.
            assert "SUPPORTS_FORMAT_BIT_MAP" in cls.__dict__, (
                f"{cls.__name__} inherits SUPPORTS_FORMAT_BIT_MAP instead of declaring it"
            )
            cls.verify_supports_params()  # must not raise
            for fmt, fs in fbm.items():
                assert isinstance(fmt, FORMAT), f"{cls.__name__}: key {fmt!r} is not a FORMAT"
                assert isinstance(fs, FormatSupport), f"{cls.__name__}[{fmt}] is not a FormatSupport"
                assert isinstance(fs.priority, int), f"{cls.__name__}[{fmt}] priority must be int"
                assert isinstance(fs.bits, tuple) and fs.bits, f"{cls.__name__}[{fmt}] bits must be non-empty tuple"
                assert len(set(fs.bits)) == len(fs.bits), f"{cls.__name__}[{fmt}] bits has duplicates"
                assert all(isinstance(b, int) and 1 <= b <= 16 for b in fs.bits), f"{cls.__name__}[{fmt}] bad bits"

    def test_supported_bits_covers_every_declared_format(self):
        from gptqmodel.utils.importer import iter_quant_linear_kernels

        for cls in iter_quant_linear_kernels():
            fbm = cls.SUPPORTS_FORMAT_BIT_MAP
            union = tuple(sorted({b for fs in fbm.values() for b in fs.bits}))
            for fmt, fs in fbm.items():
                assert cls.supported_bits(fmt) == tuple(fs.bits)
                assert set(fs.bits) <= set(union)
            assert cls.supported_bits(None) == union

    def test_auto_select_priority_ordering(self):
        from gptqmodel.utils.importer import AUTO_BACKEND_KERNEL_MAPPING

        from gptqmodel.quantization import METHOD

        gptq_map = AUTO_BACKEND_KERNEL_MAPPING[METHOD.GPTQ]
        for fmt, backend_map in gptq_map.items():
            classes = list(backend_map.values())
            priorities = [cls.SUPPORTS_FORMAT_BIT_MAP[fmt].priority for cls in classes]
            # auto-select order is highest-priority first, and priority <= 0 is excluded
            assert priorities == sorted(priorities, reverse=True), f"{fmt}: {priorities}"
            assert all(p > 0 for p in priorities), f"{fmt}: auto map contains opted-out kernel"

    def test_priority_zero_opts_out_of_auto_selection(self):
        from gptqmodel.utils.importer import AUTO_BACKEND_KERNEL_MAPPING, iter_quant_linear_kernels

        auto_classes = {
            cls
            for fmt_map in AUTO_BACKEND_KERNEL_MAPPING.values()
            for backend_map in fmt_map.values()
            for cls in backend_map.values()
        }
        for cls in iter_quant_linear_kernels():
            if all(p <= 0 for p in cls.SUPPORTS_FORMATS.values()):
                assert cls not in auto_classes, f"{cls.__name__} has priority<=0 but is auto-selectable"

    def test_public_validate_format_bit_matrix(self):
        common = dict(
            group_size=32, desc_act=False, sym=True, pack_dtype=torch.int32,
            in_features=256, out_features=128, dynamic=None, device=None, trainable=None,
        )
        for cls in (TorchLinear, TritonV2Linear):
            for fmt in (FORMAT.GPTQ, FORMAT.GPTQ_V2, FORMAT.GPTQ_P):
                declared = set(cls.SUPPORTS_FORMAT_BIT_MAP[fmt].bits)
                for bits in (2, 3, 4, 5, 6, 7, 8):
                    ok, err = cls.validate(bits=bits, format=fmt, **common)
                    if bits in declared:
                        assert ok, f"{cls.__name__} {fmt}:{bits} unexpectedly rejected: {err}"
                    else:
                        assert not ok, f"{cls.__name__} {fmt}:{bits} unexpectedly accepted"
                        assert isinstance(err, NotImplementedError)

    def test_marlin_map_matches_legacy_declaration(self):
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

        assert MarlinLinear.SUPPORTS_FORMAT_BIT_MAP == {
            FORMAT.GPTQ: FormatSupport(priority=90, bits=(4, 8)),
            FORMAT.GPTQ_V2: FormatSupport(priority=90, bits=(4, 8)),
            FORMAT.MARLIN: FormatSupport(priority=90, bits=(4, 8)),
        }
        assert MarlinLinear.SUPPORTS_BITS == [4, 8]
        assert MarlinLinear.supported_bits(FORMAT.GPTQ_P) == (4, 8)  # unknown format -> union

    def test_dynamic_bits_validation_is_format_aware(self):
        dynamic = {r".*\.down_proj": {"bits": 5}}
        ok, err = TritonV2Linear._validate_dynamic_bits(
            bits=4, dynamic=dynamic, format=FORMAT.GPTQ_V2,
        )
        assert not ok
        assert isinstance(err, NotImplementedError)

        ok, err = TritonV2Linear._validate_dynamic_bits(
            bits=3, dynamic=dynamic, format=FORMAT.GPTQ_P,
        )
        assert ok, err


class TestConvertToPlanar:
    def test_relayout_preserves_dequantized_weight(self):
        module = _make_continuous_module()
        ref = module.dequantize_weight().clone()
        assert not module.planar

        assert module.convert_to_planar()
        assert module.planar
        out = module.dequantize_weight()
        assert torch.equal(out, ref)

    def test_relayout_preserves_continuous_state_dict(self):
        # The checkpoint format label stays continuous after the runtime relayout,
        # so state_dict/save must still emit the original continuous buffers.
        module = _make_continuous_module()
        before = {k: v.clone() for k, v in module.state_dict().items()}

        assert module.convert_to_planar()
        after = module.state_dict()
        assert set(after) == set(before)
        for key in before:
            assert torch.equal(after[key].cpu(), before[key].cpu()), key

        # The serialized state must reload as a working continuous module.
        reloaded = _make_continuous_module(seed=123)
        # assign=True: the fixture packs under inference_mode, so in-place buffer
        # copies are disallowed outside it.
        reloaded.load_state_dict({k: v.clone() for k, v in after.items()}, assign=True)
        assert not reloaded.planar
        x = torch.randn(4, 256, dtype=torch.float16) * 0.5
        assert torch.allclose(reloaded(x).float(), module(x).float(), atol=1e-3, rtol=1e-3)

    def test_relayout_keeps_no_host_copies(self):
        # The continuous buffers are re-derived lazily at save time; the
        # relayout must not pin a permanent host-side duplicate per layer.
        module = _make_continuous_module()
        assert module.convert_to_planar()
        assert getattr(module, "_checkpoint_qweight", None) is None
        assert getattr(module, "_checkpoint_qzeros", None) is None
        assert module._continuous_relayout

    def test_relayout_then_strict_reload_on_same_module(self):
        # A strict load_state_dict on an already-relayouted module must accept
        # continuous-shaped tensors (buffers reset before the copy).
        module = _make_continuous_module()
        state = {k: v.clone() for k, v in module.state_dict().items()}
        ref = module.dequantize_weight().clone()

        assert module.convert_to_planar()
        assert module.planar
        # inference_mode: the fixture packs under inference_mode, so in-place
        # buffer copies are only legal inside it.
        with torch.inference_mode():
            module.load_state_dict(state)
        assert not module.planar
        assert not module._continuous_relayout
        assert torch.equal(module.dequantize_weight(), ref)

        # The reloaded continuous module can relayout again.
        assert module.convert_to_planar()
        assert module.planar
        assert torch.equal(module.dequantize_weight(), ref)

    def test_relayout_is_idempotent(self):
        module = _make_continuous_module()
        assert module.convert_to_planar()
        qweight = module.qweight.clone()
        assert module.convert_to_planar()
        assert torch.equal(module.qweight, qweight)

    def test_relayout_preserves_forward(self):
        module = _make_continuous_module()
        module.eval()
        x = torch.randn(4, module.in_features, dtype=torch.float16)
        with torch.inference_mode():
            ref = module(x)
        assert module.convert_to_planar()
        with torch.inference_mode():
            out = module(x)
        assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)

    def test_relayout_matches_planar_packing_reference(self):
        from gptqmodel.utils.planar_packing import planar_unpack_cols, planar_unpack_rows

        module = _make_continuous_module()
        codes, zeros = module._unpack_continuous_codes()
        codes = codes.to(torch.int32)
        zeros = zeros.reshape(module.scales.shape).to(torch.int32)

        assert module.convert_to_planar()
        assert torch.equal(planar_unpack_rows(module.qweight, module.bits), codes)
        assert torch.equal(
            planar_unpack_cols(module.qzeros, module.bits).reshape(module.scales.shape), zeros
        )

    def test_relayout_refuses_v1_zeros(self):
        module = _make_continuous_module()
        module.qzero_format(format=1)
        assert not module.convert_to_planar()
        assert not module.planar

    def test_relayout_refuses_non_3bit(self):
        module = _make_continuous_module(bits=4)
        assert not module.convert_to_planar()
        assert not module.planar


@pytest.mark.cuda
@pytest.mark.skipif(not torch_cuda_available, reason="Pangolin routing requires CUDA")
class TestPangolinContinuous3Bit:
    def _cuda_module(self):
        module = _make_continuous_module(cls=TritonV2Linear, sym=True)
        module.qweight = module.qweight.cuda()
        module.qzeros = module.qzeros.cuda()
        module.scales = module.scales.cuda()
        module.g_idx = module.g_idx.cuda()
        module.bias = module.bias.cuda()
        return module

    def test_post_init_relayout_and_forward_matches_reference(self):
        from gptqmodel.utils.pangolin import ensure_pangolin_runtime_available

        module = self._cuda_module()
        ref_weight = module.dequantize_weight().clone()

        module.post_init()
        if ensure_pangolin_runtime_available() and torch.cuda.get_device_capability(0) >= (8, 0):
            assert module.planar
        if not module.planar:
            pytest.skip("Pangolin runtime unavailable; relayout not performed")

        assert torch.equal(module.dequantize_weight(), ref_weight)

        module.eval()
        x = torch.randn(1, module.in_features, dtype=torch.float16, device="cuda")
        with torch.inference_mode():
            out = module(x)
        ref = x @ ref_weight.to(dtype=torch.float16, device="cuda") + module.bias
        assert torch.allclose(out.float(), ref.float(), rtol=5e-3, atol=5e-3)

    def test_saved_v2_checkpoint_reloads_and_infers_via_pangolin(self):
        """gptq_v2:3 save -> load -> Pangolin inference: the relayouted module must
        serialize the continuous layout, and the reloaded checkpoint must relayout
        again at post_init and produce accurate Pangolin outputs."""
        module = self._cuda_module()
        module.eval()
        ref_weight = module.dequantize_weight().clone()
        continuous_qweight = module.qweight.detach().cpu().clone()
        continuous_qzeros = module.qzeros.detach().cpu().clone()

        module.post_init()
        if not module.planar:
            pytest.skip("relayout not performed on this device")

        # Save after relayout: state_dict must emit the continuous layout the
        # gptq_v2 format label describes, not the runtime planar buffers.
        state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        assert torch.equal(state["qweight"], continuous_qweight)
        assert torch.equal(state["qzeros"], continuous_qzeros)

        # Reload into a fresh module (fresh continuous buffers, then overwrite).
        reloaded = _make_continuous_module(cls=TritonV2Linear, sym=True, seed=77)
        reloaded.load_state_dict(state, assign=True)
        assert not reloaded.planar
        reloaded.qweight = reloaded.qweight.cuda()
        reloaded.qzeros = reloaded.qzeros.cuda()
        reloaded.scales = reloaded.scales.cuda()
        reloaded.g_idx = reloaded.g_idx.cuda()
        reloaded.bias = reloaded.bias.cuda()
        reloaded.eval()

        reloaded.post_init()
        assert reloaded.planar

        x = torch.randn(1, reloaded.in_features, dtype=torch.float16, device="cuda") * 0.5
        with torch.inference_mode():
            out = reloaded(x)
        ref = x @ ref_weight.to(dtype=torch.float16, device="cuda") + reloaded.bias
        assert out.shape == ref.shape
        assert torch.allclose(out.float(), ref.float(), rtol=5e-3, atol=5e-3)

        # M=1 must actually route through the native Pangolin GEMV (bias is added
        # by forward after the kernel, so add it back for the comparison).
        pang = reloaded._forward_pangolin(x)
        assert pang is not None
        pang = pang[:, : reloaded.out_features] + reloaded.bias
        assert torch.allclose(pang.float(), out.float(), rtol=5e-3, atol=5e-3)

    def test_pangolin_routes_after_relayout(self):
        from gptqmodel.utils.pangolin import ensure_pangolin_runtime_available

        module = self._cuda_module()
        module.post_init()
        if not module.planar:
            pytest.skip("relayout not performed on this device")
        assert ensure_pangolin_runtime_available()

        x = torch.randn(1, module.in_features, dtype=torch.float16, device="cuda")
        out = module._forward_pangolin(x)
        assert out is not None
        assert out.shape == (1, module.padded_out_features)
