# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear


def test_exllamav3_cache_signature_tracks_in_place_buffer_updates():
    module = object.__new__(ExllamaV3Linear)
    module.trellis = torch.zeros((1, 1, 32), dtype=torch.int16)
    module.suh = torch.ones(16, dtype=torch.float16)
    module.svh = torch.ones(16, dtype=torch.float16)
    module.su = None
    module.sv = None
    module.bias = None
    module.mcg = torch.zeros((), dtype=torch.int32)
    module.mul1 = None

    before = module._current_signature()
    module.suh.add_(1)
    after = module._current_signature()

    assert before != after


def test_inference_packed_sign_mutation_rebuilds_native_handle(monkeypatch):
    import gptqmodel.exllamav3.modules.quant.exl3 as native_module

    made = []

    class FakeNative:
        def __init__(self, **kwargs):
            self.trellis = kwargs["trellis"]
            self.su_at_build = kwargs["su"].clone()
            self.sv_at_build = kwargs["sv"].clone()
            self.unloaded = False
            made.append(self)

        def unload(self):
            self.unloaded = True

    monkeypatch.setattr(native_module, "LinearEXL3", FakeNative)
    monkeypatch.setattr(ExllamaV3Linear, "validate_device", classmethod(lambda cls, device: None))

    with torch.inference_mode():
        packed = {
            "trellis": torch.zeros((8, 8, 32), dtype=torch.int16),
            "su": torch.zeros(8, dtype=torch.int16),
            "sv": torch.zeros(8, dtype=torch.int16),
        }
        module = ExllamaV3Linear.from_tensors(
            in_features=128, out_features=128, name="proj", tensors=packed
        )
        first = module._ensure_inner()

    assert module.trellis is packed["trellis"]
    assert all(not getattr(module, name).is_inference() for name in ("su", "sv"))
    assert module._ensure_inner() is first

    with torch.inference_mode():
        module.su[0] = 1
        second = module._ensure_inner()
    assert second is not first and first.unloaded
    assert second.su_at_build[0].item() == 1
    assert module._ensure_inner() is second

    with torch.inference_mode():
        module.sv[0] = 2
        third = module._ensure_inner()
    assert third is not second and second.unloaded
    assert third.sv_at_build[0].item() == 2

    # Trellis values are read through the same packed tensor, so an edit does
    # not require a new handle or a persistent decoded-weight copy.
    with torch.inference_mode():
        module.trellis[0, 0, 0] = 3
    assert module._ensure_inner() is third
    assert third.trellis[0, 0, 0].item() == 3
    assert all(
        not isinstance(value, torch.Tensor) or value.numel() != 128 * 128
        for holder in (module, third)
        for value in vars(holder).values()
    )

    with torch.inference_mode():
        replacement = torch.full((8,), 4, dtype=torch.int16)
        module.su = replacement
        fourth = module._ensure_inner()
    assert fourth is not third and third.unloaded
    assert fourth.su_at_build[0].item() == 4
    assert module.su is not replacement and not module.su.is_inference()

    state = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    state["sv"][0] = 5
    with torch.inference_mode():
        module.load_state_dict(state)
        fifth = module._ensure_inner()
    assert fifth is not fourth and fourth.unloaded
    assert fifth.sv_at_build[0].item() == 5

    module.clear_weight_cache()
    assert fifth.unloaded and module._inner is None
    assert module._ensure_inner() is made[-1]
    assert len(made) == 6


@pytest.mark.cuda
def test_native_inference_packed_sign_mutation_matches_torch_oracle():
    from gptqmodel.models._const import DEVICE, normalize_device

    if not torch.cuda.is_available() or normalize_device(torch.device("cuda", 0)) != DEVICE.CUDA:
        pytest.skip("requires NVIDIA CUDA")

    from test_exllamav3_torch_padding import _oracle_weight

    generator = torch.Generator().manual_seed(3732)
    trellis = torch.randint(-32768, 32768, (8, 8, 32), dtype=torch.int16, generator=generator)
    x = (torch.randn((2, 128), generator=generator) * 0.02).to("cuda", dtype=torch.float16)
    with torch.inference_mode():
        module = ExllamaV3Linear.from_tensors(
            in_features=128,
            out_features=128,
            name="proj",
            tensors={
                "trellis": trellis.to("cuda"),
                "su": torch.zeros(8, dtype=torch.int16, device="cuda"),
                "sv": torch.zeros(8, dtype=torch.int16, device="cuda"),
            },
        )
        try:
            before = module(x)
        except ModuleNotFoundError as exc:
            if "ExLlamaV3 torch.ops kernels are not properly installed" not in str(exc):
                raise
            pytest.skip("native EXL3 torch.ops extension is unavailable")
        first = module._inner
        module.su[0] = 1
        after = module(x)

    signs = torch.ones(128, dtype=torch.float16)
    reference = {"trellis": trellis, "suh": signs, "svh": signs}
    expected_before = (x.cpu().float() @ _oracle_weight(reference)).to(torch.float16)
    changed_signs = signs.clone()
    changed_signs[0] = -1
    expected_after = (
        x.cpu().float() @ _oracle_weight({**reference, "suh": changed_signs})
    ).to(torch.float16)

    assert module._inner is not first
    assert not torch.equal(before.cpu(), after.cpu())
    torch.testing.assert_close(before.cpu(), expected_before, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(after.cpu(), expected_after, rtol=2e-3, atol=2e-3)
