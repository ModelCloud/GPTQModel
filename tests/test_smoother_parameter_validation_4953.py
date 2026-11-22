# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Smoother descriptors must reject silently destructive parameters (#4953).

``SmoothMAD(k=0)`` clamps every row to its median, and
``SmoothPercentile(percentile=0.99)`` reads a fraction as the 0.99th *percent*
ile (the field is on a 0-100 scale). Both quantized without complaint and
produced a model with NLL 60-87. The descriptors, the dict/JSON parser and the
``SmootherConfig`` wrapper must all fail before a model is loaded.
"""

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import config as C


INVALID = [
    ("mad-k-zero", lambda: C.SmoothMAD(k=0.0), "k", "> 0"),
    ("mad-k-negative", lambda: C.SmoothMAD(k=-1.0), "k", "> 0"),
    ("mad-k-nan", lambda: C.SmoothMAD(k=float("nan")), "k", "finite"),
    ("percentile-fraction", lambda: C.SmoothPercentile(percentile=0.99), "percentile", "0-100"),
    ("percentile-zero", lambda: C.SmoothPercentile(percentile=0.0), "percentile", "1 < value <= 100"),
    ("percentile-above-100", lambda: C.SmoothPercentile(percentile=101.0), "percentile", "1 < value <= 100"),
    (
        "asym-low-ge-high",
        lambda: C.SmoothPercentileAsymmetric(low=99.0, high=1.0),
        "low",
        "high",
    ),
    ("asym-low-negative", lambda: C.SmoothPercentileAsymmetric(low=-1.0), "low", "0 <= value <= 100"),
    ("asym-high-above-100", lambda: C.SmoothPercentileAsymmetric(high=101.0), "high", "0 <= value <= 100"),
    ("mse-steps-zero", lambda: C.SmoothMSE(steps=0), "steps", ">= 1"),
    ("mse-steps-float", lambda: C.SmoothMSE(steps=1.5), "steps", "integer"),
    ("mse-maxshrink-zero", lambda: C.SmoothMSE(maxshrink=0.0), "maxshrink", "0 < value <= 1"),
    ("mse-maxshrink-above-1", lambda: C.SmoothMSE(maxshrink=1.5), "maxshrink", "0 < value <= 1"),
    ("outlier-pct-100", lambda: C.SmoothOutlier(pct=100.0), "pct", "0 <= value < 100"),
    ("outlier-pct-negative", lambda: C.SmoothOutlier(pct=-1.0), "pct", "0 <= value < 100"),
    ("softnorm-k-zero", lambda: C.SmoothSoftNorm(k=0.0), "k", "> 0"),
    ("softnorm-k-negative", lambda: C.SmoothSoftNorm(k=-1.0), "k", "> 0"),
    ("log-percentile-fraction", lambda: C.SmoothLog(percentile=0.5), "percentile", "0-100"),
    ("log-mu-zero", lambda: C.SmoothLog(mu=0.0), "mu", "> 0"),
    ("log-mu-negative", lambda: C.SmoothLog(mu=-1.0), "mu", "> 0"),
    ("rowcol-axis", lambda: C.SmoothRowCol(axis="rows"), "axis", "'row' or 'col'"),
    ("threshold-float", lambda: C.SmoothMAD(group_size_threshold=1.5), "group_size_threshold", "integer"),
    ("threshold-negative", lambda: C.SmoothMAD(group_size_threshold=-1), "group_size_threshold", ">= 0"),
]


@pytest.mark.parametrize(
    "factory,field,range_text",
    [(factory, field, range_text) for _, factory, field, range_text in INVALID],
    ids=[case_id for case_id, *_ in INVALID],
)
def test_invalid_smoother_parameters_are_rejected(factory, field, range_text):
    with pytest.raises((ValueError, TypeError)) as excinfo:
        factory()

    message = str(excinfo.value)
    assert field in message, message
    assert range_text in message, message


def test_dict_and_config_wrappers_validate_the_same_bounds():
    """The public dict/JSON and ``SmootherConfig`` paths must fail too."""

    with pytest.raises(ValueError, match="`k` must be > 0"):
        C.SmootherConfig(smooth={"type": "mad", "k": 0})
    with pytest.raises(ValueError, match="percentile"):
        C.SmootherConfig(smooth={"type": "percentile", "percentile": 0.99})
    with pytest.raises(ValueError, match="`maxshrink`"):
        C.SmootherConfig(smooth={"type": "mse", "maxshrink": 0})
    with pytest.raises(ValueError, match="`axis`"):
        C.SmootherConfig(smooth={"type": "rowcol", "axis": "diagonal"})

    with pytest.raises(ValueError, match="`k` must be > 0"):
        C.Fallback(smooth={"type": "mad", "k": 0})
    assert isinstance(
        C.Fallback(smooth={"type": "mad", "k": 2.0}).smooth, C.SmoothMAD
    )

    with pytest.raises(ValueError, match="`k` must be > 0"):
        C.RTNConfig(
            bits=4,
            group_size=32,
            smoother={"type": "mad", "k": 0},
        )
    with pytest.raises(ValueError, match="`percentile`"):
        C.RTNConfig(
            bits=4,
            group_size=32,
            smoother=C.SmootherConfig(smooth=C.SmoothPercentile(percentile=0.99)),
        )


LEGAL_BOUNDARIES = [
    lambda: C.SmoothMAD(k=0.01),
    lambda: C.SmoothPercentile(percentile=1.5),
    lambda: C.SmoothPercentile(percentile=100.0),
    lambda: C.SmoothPercentileAsymmetric(low=0.0, high=100.0),
    lambda: C.SmoothMSE(steps=1, maxshrink=1.0),
    lambda: C.SmoothMSE(maxshrink=0.001),
    lambda: C.SmoothOutlier(pct=0.0),
    lambda: C.SmoothOutlier(pct=99.99),
    lambda: C.SmoothSoftNorm(k=0.01),
    lambda: C.SmoothLog(percentile=1.5, mu=0.001),
    lambda: C.SmoothRowCol(axis="COL"),
    lambda: C.SmoothMAD(group_size_threshold=0),
    lambda: C.SmootherConfig(smooth={"type": "percentile", "percentile": 99.0}),
]


@pytest.mark.parametrize("factory", LEGAL_BOUNDARIES)
def test_legal_boundary_values_still_construct(factory):
    smoother = factory()
    assert smoother is not None
    if isinstance(smoother, C.SmoothRowCol):
        assert smoother.axis == "col"


def _build_tiny_source(tmp_path):
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).eval().save_pretrained(source)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    ).save_pretrained(source)
    return source


def test_valid_smoother_config_still_quantizes(tmp_path):
    """A legal smoother must not change the quantization contract."""

    source = _build_tiny_source(tmp_path)
    quantize_config = C.RTNConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        smoother=C.SmootherConfig(smooth=C.SmoothMAD(k=2.75)),
    )
    model = GPTQModel.load(str(source), quantize_config=quantize_config, dtype=torch.float32)
    model.quantize(backend=BACKEND.GPTQ_TORCH)

    quantized = [
        module for module in model.model.modules() if isinstance(module, BaseQuantLinear)
    ]
    assert quantized, "the valid smoother config must still quantize the model"

    with torch.inference_mode():
        output = model.model(
            input_ids=torch.tensor([[1, 3, 5, 7]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    assert torch.isfinite(output.logits).all()
