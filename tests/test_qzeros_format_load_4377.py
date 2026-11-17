# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Issue #4377: a v1 ``state_dict`` must convert into a fresh v2-runtime module.

A v1 checkpoint stores ``zero_point - 1`` while a ``REQUIRES_FORMAT_V2`` kernel
decodes logical words. Loading the checkpoint from disk converts the words, but
copying ``state_dict()`` from that loaded module into a freshly constructed
kernel skipped the conversion: the fresh module still reports the constructor
placeholder ``_qzeros_format == 1``, which looked like a match for the incoming
v1 marker even though the kernel decodes v2. The copy therefore kept v1 words
and every zero point moved one step.
"""

import torch

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


IN_FEATURES = 32
OUT_FEATURES = 8
GROUP = 32
BITS = 4


def _build(fmt: FORMAT = FORMAT.GPTQ) -> TorchLinear:
    return TorchLinear(
        bits=BITS,
        group_size=GROUP,
        sym=False,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
        format=fmt,
    )


def _pack(module: TorchLinear) -> TorchLinear:
    """Fill ``module`` with known logical (v2) zero points and weights."""

    torch.manual_seed(0)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.2)
    scales = torch.rand(OUT_FEATURES, IN_FEATURES // GROUP) * 0.05 + 1e-3
    zeros = torch.randint(0, 1 << BITS, (OUT_FEATURES, IN_FEATURES // GROUP), dtype=torch.int32)
    module.pack_original(linear, scales, zeros)
    return module


def _v1_loaded_module() -> TorchLinear:
    """A v2-runtime kernel that was loaded from a v1 ``gptq`` checkpoint."""

    module = _pack(_build(FORMAT.GPTQ))
    # The loader records the on-disk layout; the live words stay v2.
    module._checkpoint_format_on_disk = FORMAT.GPTQ
    return module.eval()


def test_fresh_v2_module_converts_a_copied_v1_state_dict():
    source = _v1_loaded_module()
    assert getattr(type(source), "REQUIRES_FORMAT_V2", False)
    assert source.qzero_format() == 2

    state = source.state_dict()
    assert state["qzeros_format"].item() == 1, (
        "the source must emit a v1 marker for a v1 checkpoint module"
    )

    fresh = _build(FORMAT.GPTQ)
    # The fresh module has never held words, so its runtime label is still the
    # constructor placeholder. Loading must not mistake that for a match.
    fresh.load_state_dict(state)
    fresh.post_init()
    fresh.eval()

    assert fresh.qzero_format() == source.qzero_format() == 2
    assert torch.equal(fresh.qzeros, source.qzeros), (
        "the copied v1 words were not converted to the v2 runtime layout"
    )

    x = torch.randn(3, IN_FEATURES)
    with torch.inference_mode():
        expected = source(x)
        actual = fresh(x)
    torch.testing.assert_close(
        actual,
        expected,
        atol=0,
        rtol=0,
        msg=lambda message: f"fresh v2 module diverged from the v1-loaded source: {message}",
    )


def test_fresh_v2_module_keeps_a_copied_v2_state_dict():
    """A v2 source needs no conversion; the fresh kernel must stay v2."""

    source = _pack(_build(FORMAT.GPTQ_V2))
    source._checkpoint_format_on_disk = FORMAT.GPTQ_V2
    source.eval()
    assert source.qzero_format() == 2
    state = source.state_dict()
    assert state["qzeros_format"].item() == 2

    fresh = _build(FORMAT.GPTQ_V2)
    fresh.load_state_dict(state)
    fresh.post_init()
    fresh.eval()

    assert fresh.qzero_format() == 2
    assert torch.equal(fresh.qzeros, source.qzeros)

    x = torch.randn(3, IN_FEATURES)
    with torch.inference_mode():
        expected = source(x)
        actual = fresh(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
