import types

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig


class _TrackingRotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(4))
        self.calls = []

    def forward(self, x, position_ids):
        self.calls.append((self.inv_freq.device, x.device, position_ids.device))
        if self.inv_freq.device != x.device:
            raise RuntimeError(
                f"rotary/device mismatch: inv_freq={self.inv_freq.device}, x={x.device}"
            )
        return x, x


class _MetaSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, device="meta")

    def forward(self, x, position_embeddings=None, position_ids=None):
        assert position_embeddings is not None
        assert position_ids is not None
        cos, sin = position_embeddings
        assert cos.device == x.device
        assert sin.device == x.device
        assert position_ids.device == x.device
        return x


def _make_processor(rotary: nn.Module) -> AWQProcessor:
    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        group_size=128,
    )
    model = types.SimpleNamespace(model=types.SimpleNamespace(rotary_emb=rotary))
    return AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        gptq_model=types.SimpleNamespace(
            rotary_embedding=None,
        ),
        model=model,
        require_fwd=True,
        calculate_w_wq_diff=False,
        calibration_concat_separator=None,
    )


def test_module_forward_builds_rotary_embeddings_on_module_device():
    rotary = _TrackingRotary()
    processor = _make_processor(rotary)
    attn = _MetaSelfAttention()
    hidden_states = torch.empty(2, 3, 4, device="meta")

    output = processor._module_forward(hidden_states, attn, {})

    assert output.device == torch.device("meta")
    assert rotary.inv_freq.device == torch.device("cpu")
    assert rotary.calls == []
    assert "meta" in processor._rotary_cache

    cached_rotary = processor._rotary_cache["meta"]
    assert cached_rotary is not rotary
    assert cached_rotary.calls == [
        (torch.device("meta"), torch.device("meta"), torch.device("meta"))
    ]


def test_refresh_forward_kwargs_uses_device_local_rotary_cache():
    rotary = _TrackingRotary()
    processor = _make_processor(rotary)
    hidden_states = torch.empty(2, 3, 4, device="meta")
    processor.inputs_cache = types.SimpleNamespace(
        attention_masks=[],
        position_ids=[],
        layer_inputs=[[hidden_states]],
        layer_input_kwargs=[{}],
    )

    processor._refresh_forward_kwargs_from_cache()

    position_ids = processor._module_forward_kwargs["position_ids"]
    position_embeddings = processor._module_forward_kwargs["position_embeddings"]
    cached_rotary = processor._rotary_cache["meta"]

    assert position_ids.device == torch.device("meta")
    assert position_embeddings[0].device == torch.device("meta")
    assert position_embeddings[1].device == torch.device("meta")
    assert rotary.inv_freq.device == torch.device("cpu")
    assert rotary.calls == []
    assert cached_rotary.calls == [
        (torch.device("meta"), torch.device("meta"), torch.device("meta"))
    ]
