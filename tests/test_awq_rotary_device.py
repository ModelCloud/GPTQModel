import types

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
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


class _RecordingMetaSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, device="meta")
        self.position_embeddings = "unset"

    def forward(self, x, position_embeddings=None, position_ids=None):
        self.position_embeddings = position_embeddings
        assert position_ids is not None
        return x


class _TypedTrackingRotary(nn.Module):
    """Rotary embedding that requires the per-layer-type selector."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(4))
        self.calls = []


class _LayerTypeTrackingRotary(_TypedTrackingRotary):
    def forward(self, x, position_ids, layer_type):
        self.calls.append(layer_type)
        return x, x


class _AttentionTypeTrackingRotary(_TypedTrackingRotary):
    def forward(self, x, position_ids, attention_type):
        self.calls.append(attention_type)
        return x, x


class _TypedSelfAttention(nn.Module):
    def __init__(self, parameter_name: str = "layer_type", value: str = "sliding_attention") -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, device="meta")
        setattr(self, parameter_name, value)

    def forward(self, x, position_embeddings=None, position_ids=None):
        assert position_embeddings is not None
        assert position_ids is not None
        return x


class _TypedDecoderLayer(nn.Module):
    """Decoder layer passed as `module2inspect` for a later AWQ group."""

    def __init__(self, parameter_name: str = "layer_type", value: str = "full_attention") -> None:
        super().__init__()
        self.self_attn = _TypedSelfAttention(parameter_name, value)

    def forward(self, x, position_embeddings=None, position_ids=None):
        assert position_embeddings is not None
        assert position_ids is not None
        return x


def _make_processor(rotary: nn.Module, *, preserve_explicit_position_embeddings: bool = False) -> AWQProcessor:
    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ,
        format=FORMAT.GEMM,
        group_size=128,
    )
    model = types.SimpleNamespace(model=types.SimpleNamespace(rotary_emb=rotary))
    return AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        gptq_model=types.SimpleNamespace(
            rotary_embedding=None,
            awq_preserve_explicit_position_embeddings=preserve_explicit_position_embeddings,
        ),
        model=model,
        calculate_w_wq_diff=False,
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


def test_module_forward_rebuilds_explicit_position_embeddings_by_default():
    rotary = _TrackingRotary()
    processor = _make_processor(rotary)
    attn = _RecordingMetaSelfAttention()
    hidden_states = torch.empty(2, 3, 4, device="meta")

    output = processor._module_forward(
        hidden_states,
        attn,
        {"position_embeddings": None},
    )

    assert output.device == torch.device("meta")
    assert isinstance(attn.position_embeddings, tuple)
    assert processor._rotary_cache["meta"].calls


def test_module_forward_preserves_explicit_none_position_embeddings_when_model_opts_in():
    rotary = _TrackingRotary()
    processor = _make_processor(rotary, preserve_explicit_position_embeddings=True)
    attn = _RecordingMetaSelfAttention()
    hidden_states = torch.empty(2, 3, 4, device="meta")

    output = processor._module_forward(
        hidden_states,
        attn,
        {"position_embeddings": None},
    )

    assert output.device == torch.device("meta")
    assert attn.position_embeddings is None
    assert rotary.calls == []
    assert processor._rotary_cache == {}


def test_module_forward_preserves_explicit_rotary_embeddings_when_model_opts_in():
    rotary = _TrackingRotary()
    processor = _make_processor(rotary, preserve_explicit_position_embeddings=True)
    attn = _RecordingMetaSelfAttention()
    hidden_states = torch.empty(1, 3, 4, device="meta")
    position_embeddings = (
        torch.empty(1, 3, 4, device="meta"),
        torch.empty(1, 3, 4, device="meta"),
    )

    output = processor._module_forward(
        hidden_states,
        attn,
        {"position_embeddings": position_embeddings},
    )

    assert output.device == torch.device("meta")
    assert attn.position_embeddings is position_embeddings
    assert rotary.calls == []
    assert processor._rotary_cache == {}


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


def test_module_forward_passes_layer_type_to_typed_rotary():
    """Issue #4038: Gemma 3/4 and OLMo 3 rotary need a per-layer-type selector.

    AWQ's scale search used to call `rotary(x, position_ids)` unconditionally,
    so Gemma3RotaryEmbedding raised `TypeError: missing 1 required positional
    argument: 'layer_type'` and every Gemma 3 text model failed to quantize.
    """

    rotary = _LayerTypeTrackingRotary()
    processor = _make_processor(rotary)
    attn = _TypedSelfAttention("layer_type", "full_attention")
    hidden_states = torch.empty(1, 3, 4, device="meta")

    output = processor._module_forward(hidden_states, attn, {})

    assert output.device == torch.device("meta")
    assert processor._rotary_cache["meta"].calls == ["full_attention"]


def test_module_forward_passes_attention_type_to_typed_rotary():
    """OLMo 3 spells the rotary selector `attention_type`."""

    rotary = _AttentionTypeTrackingRotary()
    processor = _make_processor(rotary)
    attn = _TypedSelfAttention("attention_type", "full_attention")
    hidden_states = torch.empty(1, 3, 4, device="meta")

    output = processor._module_forward(hidden_states, attn, {})

    assert output.device == torch.device("meta")
    assert processor._rotary_cache["meta"].calls == ["full_attention"]


def test_module_forward_reads_layer_type_from_nested_self_attn():
    """A later AWQ group can pass the decoder layer, not the attention block."""

    rotary = _LayerTypeTrackingRotary()
    processor = _make_processor(rotary)
    layer = _TypedDecoderLayer("layer_type", "sliding_attention")
    hidden_states = torch.empty(1, 3, 4, device="meta")

    output = processor._module_forward(hidden_states, layer, {})

    assert output.device == torch.device("meta")
    assert processor._rotary_cache["meta"].calls == ["sliding_attention"]
