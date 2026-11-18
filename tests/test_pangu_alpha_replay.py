# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""PanGu-alpha's last decoder layer must replay with ``ln_f`` + ``custom_query``.

Issue #4093 item 5: ``GPTPanguModel.forward`` applies ``ln_f`` before the last
block and passes it ``custom_query = wqe(position_ids)``, so that block's
``q_proj`` reads query embeddings. Without a replay hook GPTQModel replayed
layer 0's kwargs (``custom_query=None``) and calibrated the last layer on the
wrong input (arg0 relative difference 0.953; GPTQ worse than RTN).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.pangu_alpha import PanguAlphaQModel


def _tiny_pangu_checkpoint(path, *, dtype=torch.float32, **overrides):
    """Save a tiny package-owned PanGu-alpha checkpoint with the real tokenizer."""

    from gptqmodel.models.ports.pangu_alpha import GPTPanguForCausalLM
    from gptqmodel.models.ports.pangu_alpha.tokenization_pangu import GPTPanguTokenizer
    from tests.test_pangu_alpha_local_port import real_assets, tiny_config

    real_assets(path)
    torch.manual_seed(0)
    model = GPTPanguForCausalLM(tiny_config(**overrides)).to(dtype).eval()
    model.save_pretrained(path)
    GPTPanguTokenizer(path / "vocab.model", pad_token="<eot>").save_pretrained(path)
    return path


class _PanguTransformer(nn.Module):
    def __init__(self, layers: int = 3, dim: int = 16, vocab: int = 64):
        super().__init__()
        self.h = nn.ModuleList(
            [nn.Module() for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.wqe = nn.Embedding(vocab, dim)


class _PanguModel(nn.Module):
    def __init__(self, layers: int = 3, dim: int = 16, vocab: int = 64):
        super().__init__()
        self.transformer = _PanguTransformer(layers=layers, dim=dim, vocab=vocab)


def _instance(model: _PanguModel) -> PanguAlphaQModel:
    instance = PanguAlphaQModel.__new__(PanguAlphaQModel)
    nn.Module.__init__(instance)
    instance.model = model
    return instance


def test_last_layer_replay_applies_final_norm_and_custom_query():
    torch.manual_seed(0)
    model = _PanguModel()
    last_layer = model.transformer.h[-1]
    hidden = torch.randn(2, 7, 16)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)

    layer_input = [hidden.clone()]
    additional_inputs = {"position_ids": position_ids}

    returned = _instance(model).prepare_layer_replay_kwargs(
        layer=last_layer,
        layer_input=layer_input,
        additional_inputs=additional_inputs,
        target_device=torch.device("cpu"),
    )

    assert returned is additional_inputs
    # HF normalizes before the top-query block; replay must match.
    torch.testing.assert_close(
        layer_input[0], model.transformer.ln_f(hidden), rtol=0.0, atol=0.0
    )
    assert layer_input[0].device == torch.device("cpu")
    assert torch.is_tensor(returned["custom_query"])
    torch.testing.assert_close(
        returned["custom_query"],
        model.transformer.wqe(position_ids),
        rtol=0.0,
        atol=0.0,
    )
    assert returned["custom_query"].device == torch.device("cpu")


def test_last_layer_replay_materializes_disk_offloaded_modules(tmp_path):
    """The replay hook must restore ``ln_f``/``wqe`` from a disk-offloaded shell.

    Regression for #5135: with the default ``offload_to_disk=True`` the looper
    offloads the non-layer base modules after input capture, leaving ``ln_f``
    and ``wqe`` as meta placeholders behind an Accelerate ``SequentialHook``.
    ``move_to`` did not recognize that hook and raised
    ``NotImplementedError: Cannot move a module that still contains meta
    tensors without offload hooks`` at the last layer.
    """

    from gptqmodel.utils.offload import is_meta_module, offload_to_disk

    torch.manual_seed(0)
    model = _PanguModel(layers=2, dim=32, vocab=64)
    norm_state = {
        name: tensor.clone()
        for name, tensor in model.transformer.ln_f.state_dict().items()
    }
    wqe_state = {
        name: tensor.clone()
        for name, tensor in model.transformer.wqe.state_dict().items()
    }
    hidden = torch.randn(2, 7, 32)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)

    reference_norm = nn.LayerNorm(32)
    reference_norm.load_state_dict(norm_state)
    reference_wqe = nn.Embedding(64, 32)
    reference_wqe.load_state_dict(wqe_state)

    offload_to_disk(
        module=["transformer.ln_f", "transformer.wqe"],
        model=model,
        disk_path=str(tmp_path / "offload"),
        force=True,
    )
    assert is_meta_module(model.transformer.ln_f)
    assert is_meta_module(model.transformer.wqe)

    layer_input = [hidden.clone()]
    additional_inputs = {"position_ids": position_ids}
    returned = _instance(model).prepare_layer_replay_kwargs(
        layer=model.transformer.h[-1],
        layer_input=layer_input,
        additional_inputs=additional_inputs,
        target_device=torch.device("cpu"),
    )

    assert not is_meta_module(model.transformer.ln_f)
    assert not is_meta_module(model.transformer.wqe)
    torch.testing.assert_close(
        layer_input[0], reference_norm(hidden), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        returned["custom_query"], reference_wqe(position_ids), rtol=0.0, atol=0.0
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the wqe device-mismatch replay needs a non-CPU target device",
)
def test_last_layer_replay_moves_wqe_to_target_device():
    """Regression for #5135: ``wqe`` may stay on CPU while replay runs on CUDA.

    With ``offload_to_disk=False`` the base modules are materialized on CPU but
    the last decoder layer is replayed on the target device. The old hook moved
    ``position_ids`` to the target and then indexed the CPU embedding, raising
    ``RuntimeError: Expected all tensors to be on the same device``.
    """

    torch.manual_seed(0)
    model = _PanguModel(layers=2, dim=32, vocab=64)
    target_device = torch.device("cuda:0")
    hidden = torch.randn(2, 7, 32)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)
    reference_custom_query = model.transformer.wqe(position_ids)

    layer_input = [hidden.to(target_device)]
    additional_inputs = {"position_ids": position_ids}
    returned = _instance(model).prepare_layer_replay_kwargs(
        layer=model.transformer.h[-1],
        layer_input=layer_input,
        additional_inputs=additional_inputs,
        target_device=target_device,
    )

    assert model.transformer.wqe.weight.device == target_device
    assert returned["custom_query"].device == target_device
    torch.testing.assert_close(
        returned["custom_query"].cpu(),
        reference_custom_query,
        rtol=0.0,
        atol=0.0,
    )


def test_non_last_layer_replay_is_untouched():
    torch.manual_seed(0)
    model = _PanguModel()
    hidden = torch.randn(1, 5, 16)
    layer_input = [hidden.clone()]
    additional_inputs = {"position_ids": torch.arange(5).unsqueeze(0)}

    _instance(model).prepare_layer_replay_kwargs(
        layer=model.transformer.h[0],
        layer_input=layer_input,
        additional_inputs=additional_inputs,
        target_device=torch.device("cpu"),
    )

    assert torch.equal(layer_input[0], hidden)
    assert "custom_query" not in additional_inputs


def test_last_layer_replay_rebuilds_missing_position_ids():
    torch.manual_seed(0)
    model = _PanguModel()
    hidden = torch.randn(3, 4, 16)

    additional_inputs = _instance(model).prepare_layer_replay_kwargs(
        layer=model.transformer.h[-1],
        layer_input=[hidden.clone()],
        additional_inputs={},
        target_device=torch.device("cpu"),
    )

    assert additional_inputs["custom_query"].shape == (3, 4, 16)
    assert additional_inputs["custom_query"].dtype == model.transformer.wqe.weight.dtype


def test_pangu_last_layer_gptq_beats_rtn(tmp_path):
    """Acceptance criterion: GPTQ must be closer than RTN on the top-query layer."""

    from gptqmodel import GPTQModel
    from gptqmodel.models.ports.pangu_alpha import GPTPanguForCausalLM
    from gptqmodel.models.ports.pangu_alpha.tokenization_pangu import GPTPanguTokenizer
    from gptqmodel.quantization.config import GPTQConfig, RTNConfig
    from tests.test_pangu_alpha_local_port import real_assets, tiny_config

    path = real_assets(tmp_path / "tiny")
    torch.manual_seed(0)
    hf = GPTPanguForCausalLM(
        tiny_config(vocab_size=128, hidden_size=64, intermediate_size=256, num_layers=4, max_position_embeddings=64)
    ).eval()
    hf.save_pretrained(path)
    GPTPanguTokenizer(path / "vocab.model", pad_token="<eot>").save_pretrained(path)
    generator = torch.Generator().manual_seed(0)

    def sample():
        input_ids = torch.randint(4, 128, (1, 32), generator=generator)
        mask = torch.ones(1, 32, dtype=torch.long)
        mask[:, -int(torch.randint(1, 12, (1,), generator=generator).item()):] = 0
        return {"input_ids": input_ids, "attention_mask": mask}

    calibration = [sample() for _ in range(16)]
    position_ids = torch.arange(32).unsqueeze(0).expand(4, -1)
    custom_query = hf.transformer.wqe(position_ids)
    with torch.no_grad():
        reference = hf.transformer.h[-1].attn.q_proj(custom_query).clone()

    errors = {}
    for label, config in (
        ("RTN", RTNConfig(bits=4, group_size=32, device="cpu")),
        ("GPTQ", GPTQConfig(bits=4, group_size=32, device="cpu")),
    ):
        model = GPTQModel.load(str(path), quantize_config=config, dtype=torch.float32)
        model.quantize(CalibrationConfig(dataset=calibration, batch_size=1))
        save_path = tmp_path / label
        model.save(str(save_path))
        del model

        reloaded = GPTQModel.load(str(save_path), device="cpu", dtype=torch.float32)
        with torch.no_grad():
            quantized = reloaded.model.transformer.h[-1].attn.q_proj(custom_query)
        errors[label] = float((quantized - reference).norm() / reference.norm())
        del reloaded

    print(f"[pangu-quality] {errors}")
    assert errors["GPTQ"] < errors["RTN"], (
        f"GPTQ last-layer q_proj error {errors['GPTQ']:.4f} must beat RTN "
        f"{errors['RTN']:.4f}; the top-query layer is calibrated on the wrong input"
    )


def test_pangu_unpadded_calibration_reaches_every_gptq_module(tmp_path, monkeypatch):
    """Unpadded rows must still accumulate a non-zero Hessian (#5134).

    PanGu-alpha's block code turns an unpadded 0/1 tokenizer mask into an
    all-zero extended additive mask. If the keep-mask classifier reads that
    as "all padding", every module records zero samples and silently falls back
    to RTN. Assert the sample counts the quantizer actually accumulated, not
    just the downstream quality comparison.
    """

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig
    from gptqmodel.quantization.gptq import GPTQ

    path = _tiny_pangu_checkpoint(
        tmp_path / "tiny",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_layers=4,
        max_position_embeddings=64,
    )

    generator = torch.Generator().manual_seed(0)
    calibration = [
        {
            "input_ids": torch.randint(4, 128, (1, 32), generator=generator),
            # No padding: PanGu's block-level extended mask is all zeros.
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(4)
    ]

    sample_counts = {}
    original_quantize = GPTQ.quantize

    def spy_quantize(self, blocksize=128):
        name = getattr(self._named_module, "full_name", None) or self.name
        sample_counts[name] = int(self.nsamples)
        return original_quantize(self, blocksize=blocksize)

    monkeypatch.setattr(GPTQ, "quantize", spy_quantize)

    model = GPTQModel.load(
        str(path),
        quantize_config=GPTQConfig(bits=4, group_size=32, device="cpu"),
        dtype=torch.float32,
    )
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=1))

    print(f"[pangu-hessian] modules={len(sample_counts)} samples={sample_counts}")
    assert sample_counts, "no GPTQ module was quantized"
    empty = {name: count for name, count in sample_counts.items() if count <= 0}
    assert not empty, f"modules with an empty Hessian: {empty}"
    assert any(
        "h.3" in name and "q_proj" in name for name in sample_counts
    ), sample_counts


@pytest.fixture(scope="module")
def tiny_pangu_2l(tmp_path_factory):
    target = tmp_path_factory.mktemp("tiny_pangu_2l") / "model"
    return _tiny_pangu_checkpoint(
        target,
        vocab_size=128,
        # The default offload helper skips payloads below 4 KiB. Keep the
        # final norm above that so the lazy/offload path really replaces it
        # with a meta placeholder instead of leaving it resident.
        hidden_size=1024,
        intermediate_size=256,
        num_layers=2,
        num_heads=8,
        max_position_embeddings=64,
    )


def test_last_layer_replay_materializes_lazy_shell(tiny_pangu_2l):
    """The hook must materialize a meta ``ln_f``/``wqe`` from the lazy turtle.

    Regression for #5135: ``GPTQModel.load`` builds the shell on meta and only
    materializes the first decoder layer for input capture. Before the fix
    ``move_to(ln_f)`` raised ``NotImplementedError`` because a lazy shell has no
    Accelerate offload hook to restore from.
    """

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig
    from gptqmodel.utils.offload import is_meta_module

    model = GPTQModel.load(
        str(tiny_pangu_2l),
        quantize_config=GPTQConfig(
            bits=4, group_size=32, device="cpu", offload_to_disk=True
        ),
        dtype=torch.float32,
    )
    transformer = model.model.transformer
    assert is_meta_module(transformer.ln_f)
    assert is_meta_module(transformer.wqe)

    hidden = torch.randn(1, 8, 1024)
    position_ids = torch.arange(8).unsqueeze(0)

    from gptqmodel.models.ports.pangu_alpha import GPTPanguForCausalLM

    reference = GPTPanguForCausalLM.from_pretrained(str(tiny_pangu_2l)).eval()
    with torch.no_grad():
        reference_norm = reference.transformer.ln_f(hidden)
        reference_custom_query = reference.transformer.wqe(position_ids)

    layer_input = [hidden.clone()]
    returned = model.prepare_layer_replay_kwargs(
        layer=transformer.h[-1],
        layer_input=layer_input,
        additional_inputs={"position_ids": position_ids},
        target_device=torch.device("cpu"),
    )

    assert not is_meta_module(transformer.ln_f)
    assert not is_meta_module(transformer.wqe)
    torch.testing.assert_close(
        layer_input[0], reference_norm, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        returned["custom_query"], reference_custom_query, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    "device,offload_to_disk",
    [
        pytest.param("cpu", True, id="cpu-offload"),
        pytest.param(
            "cuda:0",
            True,
            id="cuda-offload",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
        pytest.param(
            "cuda:0",
            False,
            id="cuda-resident",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_pangu_2_layer_quantize_completes(
    tiny_pangu_2l, tmp_path, device, offload_to_disk
):
    """End-to-end #5135: the last layer must replay in every load configuration."""

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig

    generator = torch.Generator().manual_seed(0)
    calibration = [
        {
            "input_ids": torch.randint(4, 128, (1, 32), generator=generator),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(4)
    ]

    model = GPTQModel.load(
        str(tiny_pangu_2l),
        quantize_config=GPTQConfig(
            bits=4,
            group_size=32,
            device=device,
            offload_to_disk=offload_to_disk,
        ),
        dtype=torch.float32,
    )
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=1))

    save_path = tmp_path / f"quantized-{device.replace(':', '_')}-{offload_to_disk}"
    model.save(str(save_path))
    del model

    reloaded = GPTQModel.load(
        str(save_path), device=device, dtype=torch.float32
    )
    reloaded.model.eval()
    with torch.no_grad():
        logits = reloaded.model(
            input_ids=calibration[0]["input_ids"].to(device),
            attention_mask=calibration[0]["attention_mask"].to(device),
        ).logits
    assert torch.isfinite(logits).all(), "reloaded PanGu logits contain non-finite values"
