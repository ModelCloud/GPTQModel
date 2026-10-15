# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""VPTQ (vector post-training quantization) coverage.

Three parts of the VPTQ path can silently corrupt weights, so each one is
pinned against an independent oracle:

1. the bit-level index packer, which must stay byte-identical to the upstream
   `microsoft/VPTQ` layout (`(residual << index_bits) | index` in a
   little-endian 32-bit word stream) so checkpoints stay interoperable
2. the quantizer -> module reconstruction round trip, including permutation,
   per-channel normalization and residual refinement
3. the Torch inference kernel, checked for activation-dtype preservation and
   for accuracy against an independent Torch oracle

The oracles below are written independently of the production code: the packer
oracle is a pure-Python bit writer and the reconstruction oracle walks the
codebooks with explicit element indexing.
"""

import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.vptq import VPTQLinear
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig, VPTQConfig
from gptqmodel.quantization.vptq import VPTQ, _chunked_assign
from gptqmodel.utils.vptq import (
    pack_vptq_indices,
    unpack_vptq_indices,
    vptq_group_for_column,
    vptq_group_setting,
    vptq_index_bits,
    vptq_packed_words,
)


# ---------------------------------------------------------------------------
# independent oracles
# ---------------------------------------------------------------------------
def oracle_pack(indices, index_bits, res_indices=None, res_bits=0):
    """Pure-Python reference packer for the upstream little-endian bitstream."""

    rows = len(indices)
    columns = len(indices[0])
    total_bits = index_bits + res_bits
    words = (columns * total_bits + 31) // 32
    packed = [[0] * words for _ in range(rows)]
    for row in range(rows):
        for column in range(columns):
            value = int(indices[row][column])
            if res_indices is not None:
                value |= int(res_indices[row][column]) << index_bits
            for bit in range(total_bits):
                if (value >> bit) & 1:
                    position = column * total_bits + bit
                    packed[row][position // 32] |= 1 << (position % 32)
    return packed


def oracle_unpack(words, index_bits, num_elements, res_bits=0):
    """Pure-Python inverse of :func:`oracle_pack`."""

    total_bits = index_bits + res_bits
    indices = []
    res_indices = []
    for element in range(num_elements):
        value = 0
        for bit in range(total_bits):
            position = element * total_bits + bit
            if (int(words[position // 32]) >> (position % 32)) & 1:
                value |= 1 << bit
        indices.append(value & ((1 << index_bits) - 1))
        if res_bits:
            res_indices.append((value >> index_bits) & ((1 << res_bits) - 1))
    return indices, (res_indices if res_bits else None)


def oracle_reconstruct(result, module):
    """Rebuild the dense weight from a quantizer result with explicit indexing."""

    dtype = module.codebook_dtype
    vector_len = module.vector_len
    rows = module.num_indices * vector_len
    device = module.centroids.weight.device

    def lookup(codebook, index_rows):
        table = codebook.to(device=device, dtype=dtype).reshape(-1, vector_len)
        dense = torch.zeros((rows, len(index_rows[0])), dtype=dtype, device=device)
        for row, index_row in enumerate(index_rows):
            for column, index in enumerate(index_row):
                dense[row * vector_len:(row + 1) * vector_len, column] = table[int(index)]
        return dense

    main = torch.cat(
        [
            lookup(result.centroids[group], result.indices[group])
            for group in range(1, module.group_num + 1)
        ],
        dim=1,
    )
    if module.enable_residual:
        main = main + torch.cat(
            [
                lookup(result.res_centroids[group], result.res_indices[group])
                for group in range(1, module.group_num + 1)
            ],
            dim=1,
        )

    if module.row_padding > 0:
        main = main[: -module.row_padding]

    if module.enable_outlier:
        outlier_table = result.centroids[0].to(device=device, dtype=dtype).reshape(
            -1, module.outlier_vector_len
        )
        outlier = torch.zeros(
            (module.outlier_num_indices * module.outlier_vector_len, module.outlier_size),
            dtype=dtype,
            device=device,
        )
        for row, index_row in enumerate(result.indices[0]):
            for column, index in enumerate(index_row):
                outlier[
                    row * module.outlier_vector_len:(row + 1) * module.outlier_vector_len, column
                ] = outlier_table[int(index)]
        if module.outlier_row_padding > 0:
            outlier = outlier[: -module.outlier_row_padding]
        main = torch.cat([outlier, main], dim=1)

    if module.enable_perm:
        main = main[:, torch.argsort(result.perm.to(device=device, dtype=torch.int64))]

    if module.enable_norm:
        scale = result.weight_scale.to(device=device, dtype=dtype)
        bias = result.weight_bias.to(device=device, dtype=dtype)
        if module.norm_dim == 0:
            main = main * scale + bias
        else:
            main = main * scale.unsqueeze(1) + bias.unsqueeze(1)

    return main.to(torch.float32)


def build_quantizer(weight, qcfg, dtype=torch.float32):
    """Attach a VPTQ quantizer to a real `nn.Linear` holding `weight`."""

    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=dtype)
    linear.weight.data.copy_(weight.to(dtype))
    return VPTQ(module=linear, qcfg=qcfg)


def build_module(qcfg, in_features, out_features, dtype=torch.float32, register_buffers=False):
    """Create a `VPTQLinear` matching `qcfg`."""

    return VPTQLinear(
        bits=qcfg.bits,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        vector_lens=qcfg.vector_lens,
        num_centroids=qcfg.num_centroids,
        num_res_centroids=qcfg.num_res_centroids,
        npercent=qcfg.npercent,
        group_size=qcfg.group_size,
        group_num=qcfg.group_num,
        enable_norm=qcfg.enable_norm,
        norm_dim=qcfg.norm_dim,
        enable_perm=qcfg.enable_perm,
        dtype=dtype,
        register_buffers=register_buffers,
    )


def synthetic_problem(out_features=48, in_features=64, seed=0):
    """Return a small `(weight, hessian)` pair with a well conditioned Hessian."""

    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator) * 0.05
    features = torch.randn(256, in_features, generator=generator) @ torch.randn(
        in_features, in_features, generator=generator
    ) * 0.2
    hessian = (features.double().t() @ features.double()) / features.shape[0]
    return weight, hessian


# ---------------------------------------------------------------------------
# index packing
# ---------------------------------------------------------------------------
def test_vptq_index_bits_and_word_counts():
    assert [vptq_index_bits(value) for value in (1, 2, 3, 4, 5, 8, 9, 16, 4096)] == [
        0, 1, 2, 2, 3, 3, 4, 4, 12,
    ]
    # Word counts round up: 32 bits fit one word, one extra bit needs two.
    assert vptq_packed_words(32, 1) == 1
    assert vptq_packed_words(33, 1) == 2
    assert vptq_packed_words(1, 32) == 1
    assert vptq_packed_words(2, 32) == 2
    assert vptq_packed_words(8, 12) == 3
    with pytest.raises(ValueError):
        vptq_index_bits(0)
    with pytest.raises(ValueError):
        vptq_packed_words(1, 0)


@pytest.mark.parametrize(
    ("index_bits", "res_bits"),
    [(1, 0), (2, 0), (4, 0), (8, 0), (12, 0), (12, 4), (6, 4), (16, 0)],
)
@pytest.mark.parametrize("num_elements", [1, 2, 7, 8, 9, 16, 31, 32, 33, 65])
def test_vptq_pack_matches_oracle(index_bits, res_bits, num_elements):
    """Packed words must match the independent bit writer bit-for-bit."""

    generator = torch.Generator().manual_seed(index_bits * 100 + num_elements)
    indices = torch.randint(0, 1 << index_bits, (3, num_elements), generator=generator, dtype=torch.int64)
    res_indices = None
    if res_bits:
        res_indices = torch.randint(
            0, 1 << res_bits, (3, num_elements), generator=generator, dtype=torch.int64
        )

    packed = pack_vptq_indices(indices, index_bits, res_indices=res_indices, res_bits=res_bits)
    expected = oracle_pack(
        indices.tolist(), index_bits, res_indices.tolist() if res_bits else None, res_bits
    )

    assert packed.shape == (3, vptq_packed_words(num_elements, index_bits + res_bits))
    assert packed.dtype == torch.int32
    # int32 words hold the raw little-endian bit pattern; compare unsigned.
    assert (packed.to(torch.int64) & 0xFFFFFFFF).tolist() == expected

    unpacked, unpacked_res = unpack_vptq_indices(packed, index_bits, num_elements, res_bits=res_bits)
    assert torch.equal(unpacked, indices)
    if res_bits:
        assert torch.equal(unpacked_res, res_indices)
    else:
        assert unpacked_res is None


def test_vptq_pack_boundary_values_and_padding():
    """Boundary codes, word-straddling elements and the zero tail padding."""

    index_bits, res_bits = 12, 4
    total_bits = index_bits + res_bits
    num_elements = 9  # 9 * 16 = 144 bits -> 5 words with 16 padding bits
    indices = torch.zeros((1, num_elements), dtype=torch.int64)
    res_indices = torch.zeros((1, num_elements), dtype=torch.int64)

    # smallest code, largest code, and the largest code in the last element
    indices[0, 0] = 0
    indices[0, 1] = (1 << index_bits) - 1
    indices[0, 2] = 1
    indices[0, -1] = (1 << index_bits) - 1
    res_indices[0, 0] = (1 << res_bits) - 1
    res_indices[0, 1] = 0
    res_indices[0, -1] = (1 << res_bits) - 1

    packed = pack_vptq_indices(indices, index_bits, res_indices=res_indices, res_bits=res_bits)
    assert (packed.to(torch.int64) & 0xFFFFFFFF).tolist() == oracle_pack(
        indices.tolist(), index_bits, res_indices.tolist(), res_bits
    )

    # tail padding bits must stay zero: only the 144 low bits may be set
    padding_bits = packed.shape[-1] * 32 - num_elements * total_bits
    assert padding_bits == 16
    last_word = int(packed[0, -1]) & 0xFFFFFFFF
    assert last_word >> (32 - padding_bits) == 0

    unpacked, unpacked_res = unpack_vptq_indices(packed, index_bits, num_elements, res_bits=res_bits)
    assert torch.equal(unpacked, indices)
    assert torch.equal(unpacked_res, res_indices)

    # codes outside the declared bit width are rejected instead of silently truncated
    with pytest.raises(ValueError):
        pack_vptq_indices(torch.full((1, 2), 1 << index_bits, dtype=torch.int64), index_bits)


def test_vptq_pack_empty_and_invalid():
    empty = pack_vptq_indices(torch.zeros((2, 0), dtype=torch.int64), 4)
    assert empty.shape == (2, 0)
    indices, res_indices = unpack_vptq_indices(empty, 4, 0, res_bits=0)
    assert indices.shape == (2, 0) and res_indices is None

    with pytest.raises(ValueError):
        pack_vptq_indices(torch.zeros((1, 2), dtype=torch.int64), 0)
    with pytest.raises(ValueError):
        pack_vptq_indices(torch.zeros((1, 2), dtype=torch.int64), 4, res_bits=2)


def test_vptq_group_layout():
    """Group resolution mirrors the reference layout, including padding."""

    assert vptq_group_setting(64, npercent=0.0, group_size=-1, group_num=-1) == (0, 64, 1)
    assert vptq_group_setting(64, npercent=0.0, group_size=16, group_num=-1) == (0, 16, 4)
    # 64 - 1 outlier channels do not divide by 3: the remainder pads the outlier set
    assert vptq_group_setting(64, npercent=0.0, group_size=-1, group_num=3) == (1, 21, 3)

    outlier_size, group_size, group_num = vptq_group_setting(64, npercent=25.0, group_size=-1, group_num=2)
    assert (outlier_size, group_num) == (16, 2)
    assert outlier_size + group_size * group_num == 64

    assert vptq_group_for_column(0, outlier_size=16, group_size=24) == 0
    assert vptq_group_for_column(15, outlier_size=16, group_size=24) == 0
    assert vptq_group_for_column(16, outlier_size=16, group_size=24) == 1
    assert vptq_group_for_column(39, outlier_size=16, group_size=24) == 1
    assert vptq_group_for_column(40, outlier_size=16, group_size=24) == 2

    with pytest.raises(ValueError):
        vptq_group_setting(64, npercent=0.0, group_size=16, group_num=4)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_vptq_config_defaults_and_roundtrip():
    qcfg = VPTQConfig()
    assert qcfg.method == METHOD.VPTQ and qcfg.format == FORMAT.VPTQ
    # 12-bit main index + 4-bit residual over 8-element vectors -> 2 bpw label
    assert qcfg.bits == 2
    # reference implementation weights codebook training by the Hessian diagonal
    assert qcfg.kmeans_mode == "hessian"

    payload = qcfg.to_dict()
    assert payload["method"] == METHOD.VPTQ.value
    assert payload["format"] == FORMAT.VPTQ.value

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, VPTQConfig)
    assert restored.vector_lens == qcfg.vector_lens
    assert restored.num_centroids == qcfg.num_centroids
    assert restored.num_res_centroids == qcfg.num_res_centroids
    assert restored.bits == qcfg.bits
    assert restored.quant_linear_init_kwargs() == qcfg.quant_linear_init_kwargs()


def test_vptq_config_validation():
    with pytest.raises(ValueError):
        VPTQConfig(vector_lens=[-1, 0])
    with pytest.raises(ValueError):
        VPTQConfig(num_centroids=[-1, 100])
    with pytest.raises(ValueError):
        VPTQConfig(num_centroids=[-1, 64], group_size=8, group_num=4)
    with pytest.raises(ValueError):
        VPTQConfig(npercent=10.0)
    with pytest.raises(ValueError):
        VPTQConfig(kmeans_mode="kmeans")


def test_vptq_config_derives_bits_from_codebooks():
    assert VPTQConfig(num_centroids=[-1, 256], num_res_centroids=[-1, -1], vector_lens=[-1, 8]).bits == 1
    assert VPTQConfig(num_centroids=[-1, 256], num_res_centroids=[-1, -1], vector_lens=[-1, 4]).bits == 2
    assert VPTQConfig(num_centroids=[-1, 65536], num_res_centroids=[-1, -1], vector_lens=[-1, 8]).bits == 2


# ---------------------------------------------------------------------------
# quantization -> module round trip
# ---------------------------------------------------------------------------
VPTQ_VARIANTS = [
    ("baseline", {}),
    ("group_num", {"group_num": 4}),
    ("perm", {"enable_perm": True}),
    ("norm", {"enable_norm": True}),
    ("residual", {"num_res_centroids": [-1, 16]}),
    ("outlier", {"npercent": 12.5, "vector_lens": [4, 8], "num_centroids": [16, 64]}),
    (
        "all",
        {
            "npercent": 12.5,
            "vector_lens": [4, 8],
            "num_centroids": [16, 64],
            "num_res_centroids": [-1, 8],
            "enable_perm": True,
            "enable_norm": True,
            "group_num": 2,
        },
    ),
]


@pytest.mark.parametrize(("variant", "overrides"), VPTQ_VARIANTS, ids=[item[0] for item in VPTQ_VARIANTS])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_vptq_quantize_module_roundtrip(variant, overrides, dtype):
    """Codebooks + indices must rebuild exactly the quantizer's dense weight."""

    del variant
    weight, hessian = synthetic_problem()
    settings = {
        "num_centroids": [-1, 64],
        "num_res_centroids": [-1, -1],
        "vector_lens": [-1, 4],
        "kmeans_iters": 12,
        "block_size": 32,
    }
    settings.update(overrides)
    qcfg = VPTQConfig(**settings)
    quantizer = build_quantizer(weight, qcfg, dtype=dtype)
    result, loss = quantizer.quantize_weight(weight, hessian, target_dtype=dtype)

    assert math.isfinite(loss) and loss >= 0.0

    module = build_module(qcfg, weight.shape[1], weight.shape[0], dtype=dtype)
    module.pack(linear=quantizer.layer, vptq_result=result)

    dequantized = module.dequantize()
    assert dequantized.shape == weight.shape
    assert dequantized.dtype == dtype
    # quantizer reconstruction (vectorized) == module unpack+lookup == oracle
    assert torch.equal(dequantized, result.quantized_weight)
    assert torch.equal(dequantized.float(), oracle_reconstruct(result, module))


def test_vptq_codebook_larger_than_available_vectors():
    """A codebook bigger than the vector count keeps its configured size.

    Small layers (e.g. `k_proj`/`v_proj` with a 65536-entry codebook) have
    fewer weight sub-vectors than codebook entries. The learned codebook must
    still be stored at the configured size, otherwise `pack()` cannot load it
    into the module whose buffers follow the configuration.
    """

    weight, hessian = synthetic_problem(out_features=16, in_features=64)
    qcfg = VPTQConfig(
        vector_lens=[-1, 8],
        num_centroids=[-1, 256],
        num_res_centroids=[-1, -1],
        kmeans_iters=4,
    )
    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)

    # 64 columns * ceil(16 / 8) = 128 vectors, i.e. fewer than 256 centroids
    assert result.centroids[1].shape == (256, 8)
    assert result.indices[1].shape == (2, 64)

    module = build_module(qcfg, weight.shape[1], weight.shape[0])
    module.pack(linear=quantizer.layer, vptq_result=result)
    dequantized = module.dequantize()
    assert dequantized.shape == weight.shape
    assert torch.equal(dequantized, result.quantized_weight)
    assert torch.equal(dequantized.float(), oracle_reconstruct(result, module))
    # every vector owns a centroid, so only k-means rounding remains
    assert (dequantized - weight).abs().max().item() < 1e-6


def test_vptq_residual_reduces_hessian_error():
    """The residual codebook must not increase the reconstruction error."""

    weight, hessian = synthetic_problem()
    base = VPTQConfig(
        num_centroids=[-1, 16], num_res_centroids=[-1, -1], vector_lens=[-1, 4], kmeans_iters=12
    )
    with_residual = VPTQConfig(
        num_centroids=[-1, 16],
        num_res_centroids=[-1, 16],
        vector_lens=[-1, 4],
        kmeans_iters=12,
    )

    _, base_loss = build_quantizer(weight, base).quantize_weight(weight, hessian)
    _, residual_loss = build_quantizer(weight, with_residual).quantize_weight(weight, hessian)
    assert residual_loss < base_loss


def test_vptq_nearest_centroid_tie_breaks_to_lower_index():
    """An exact tie must pick the lower codebook index; one ULP must not."""

    codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    tie = torch.tensor([[0.5, 0.0]])
    assignment, _ = _chunked_assign(tie, codebook)
    assert int(assignment[0]) == 0

    below = torch.tensor([[0.5 - 1e-6, 0.0]])
    above = torch.tensor([[0.5 + 1e-6, 0.0]])
    assert int(_chunked_assign(below, codebook)[0][0]) == 0
    assert int(_chunked_assign(above, codebook)[0][0]) == 1


def test_vptq_quantizer_accumulates_calibration_hessian():
    """`add_batch` must average `x^T x` over the calibration tokens."""

    qcfg = VPTQConfig(num_centroids=[-1, 16], num_res_centroids=[-1, -1], vector_lens=[-1, 4])
    quantizer = build_quantizer(torch.zeros(8, 6), qcfg)

    generator = torch.Generator().manual_seed(3)
    batch = torch.randn(2, 5, 6, generator=generator, dtype=torch.float64)
    quantizer.add_batch(batch)
    quantizer.add_batch(batch.flip(0))

    # both batches hold the same tokens, so the mean Hessian is the single-batch one
    tokens = batch.reshape(-1, 6)
    expected = (tokens.t() @ tokens) / tokens.shape[0]
    hessian = quantizer.materialize_hessian()
    assert torch.allclose(hessian, expected, rtol=1e-12, atol=1e-12)
    assert quantizer.nsamples == batch.shape[0] * batch.shape[1] * 2


# ---------------------------------------------------------------------------
# inference kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_vptq_inference_matches_torch_oracle(dtype):
    """The kernel must preserve the activation dtype and match a dense oracle."""

    weight, hessian = synthetic_problem(out_features=48, in_features=64)
    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, 8],
        vector_lens=[-1, 4],
        kmeans_iters=12,
        enable_norm=True,
        enable_perm=True,
    )
    quantizer = build_quantizer(weight, qcfg, dtype=dtype)
    result, _ = quantizer.quantize_weight(weight, hessian, target_dtype=dtype)

    module = build_module(qcfg, weight.shape[1], weight.shape[0], dtype=dtype)
    module.pack(linear=quantizer.layer, vptq_result=result)

    generator = torch.Generator().manual_seed(7)
    activations = torch.randn(5, 64, generator=generator).to(dtype)
    actual = module(activations)
    assert actual.dtype == dtype

    # visible-output comparison: oracle rounded to the same dtype
    oracle_weight = oracle_reconstruct(result, module).to(torch.float64)
    oracle = (activations.to(torch.float64) @ oracle_weight.t()).to(dtype)
    drift = (actual.to(torch.float64) - oracle.to(torch.float64)).abs()
    assert float(drift.max()) <= 2e-3

    # fp32 arithmetic comparison, separate from the mandatory output rounding
    fp32_actual = module(activations.to(torch.float32))
    assert fp32_actual.dtype == torch.float32
    fp32_oracle = activations.to(torch.float64) @ oracle_weight.t()
    fp32_drift = (fp32_actual.to(torch.float64) - fp32_oracle).abs()
    assert float(fp32_drift.max()) <= 1e-5


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vptq_cuda_quantize_and_infer(dtype):
    """Quantization and inference must run on the target accelerator."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for VPTQ accelerator coverage")

    device = torch.device("cuda:0")
    weight, hessian = synthetic_problem(out_features=64, in_features=128)
    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, 8],
        vector_lens=[-1, 4],
        kmeans_iters=12,
        enable_norm=True,
        enable_perm=True,
    )

    quantizer = build_quantizer(weight, qcfg, dtype=dtype)
    quantizer.layer.to(device)
    result, loss = quantizer.quantize_weight(
        weight.to(device=device, dtype=dtype), hessian.to(device), target_dtype=dtype
    )
    assert math.isfinite(loss)
    assert result.indices[1].device.type == "cuda"

    module = build_module(qcfg, weight.shape[1], weight.shape[0], dtype=dtype).to(device)
    module.pack(linear=quantizer.layer, vptq_result=result)
    assert module.centroids.weight.device == device

    dequantized = module.dequantize()
    assert dequantized.device == device
    assert torch.equal(dequantized.cpu(), result.quantized_weight.cpu())
    assert torch.equal(dequantized.cpu().float(), oracle_reconstruct(result, module).cpu())

    generator = torch.Generator().manual_seed(11)
    activations = torch.randn(4, weight.shape[1], generator=generator).to(device=device, dtype=dtype)
    actual = module(activations)
    assert actual.dtype == dtype and actual.device == device

    oracle_weight = oracle_reconstruct(result, module).to(device=device, dtype=torch.float64)
    oracle = (activations.to(torch.float64) @ oracle_weight.t()).to(dtype)
    assert float((actual.to(torch.float64) - oracle.to(torch.float64)).abs().max()) <= 2e-3


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
_CALIBRATION_TEXTS = [
    "tiny vptq calibration sample one with enough tokens to survive minimum length filtering",
    "tiny vptq calibration sample two with repeated words to exercise the vector quantization path",
    "another synthetic calibration example that is intentionally verbose so filtering does not remove it",
] * 2


def _build_local_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
    """Persist a minimal tokenizer because the GPT-QModel load path expects local files."""

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _CALIBRATION_TEXTS,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)
    return fast_tokenizer


def _build_tiny_qwen3_fixture(model_dir: Path):
    """Save a tiny dense HF checkpoint that still exercises the real qwen3 path."""

    config = Qwen3Config(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3ForCausalLM(config).save_pretrained(model_dir)
    return config, _build_local_tokenizer(model_dir)


def _build_calibration_dataset(tokenizer: PreTrainedTokenizerFast):
    return [
        {
            "input_ids": tokenizer(text, return_tensors="pt")["input_ids"],
            "attention_mask": tokenizer(text, return_tensors="pt")["attention_mask"],
        }
        for text in _CALIBRATION_TEXTS
    ]


@pytest.mark.cpu
@pytest.mark.slow
def test_tiny_qwen3_vptq_quantization_smoke(tmp_path: Path):
    """Quantize, save and reload a tiny model, then compare greedy decoding."""

    native_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    native_dir.mkdir()

    _, tokenizer = _build_tiny_qwen3_fixture(native_dir)
    calibration = _build_calibration_dataset(tokenizer)

    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, 8],
        vector_lens=[-1, 8],
        kmeans_iters=8,
        device="cpu",
    )
    model = GPTQModel.load(str(native_dir), quantize_config=qcfg, backend=BACKEND.VPTQ_TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.VPTQ_TORCH, calibration_data_min_length=1)

    expected_modules = [
        f"model.model.layers.0.self_attn.{projection}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    ] + [
        f"model.model.layers.0.mlp.{projection}" for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    for name in expected_modules:
        module = dict(model.named_modules())[name]
        assert isinstance(module, VPTQLinear), name

    # each quantized layer must reproduce its own dequantized weight exactly
    for name in expected_modules:
        module = dict(model.named_modules())[name]
        activations = torch.randn(2, module.in_features, dtype=torch.float32)
        assert torch.allclose(
            module(activations),
            activations @ module.dequantize().float().t(),
            atol=1e-5,
            rtol=1e-5,
        )

    prompt = "tiny vptq calibration sample"
    inputs = tokenizer(prompt, return_tensors="pt")
    model.save(quantized_dir)

    reloaded = GPTQModel.load(str(quantized_dir), device="cpu", dtype=torch.bfloat16)
    assert reloaded.quantize_config.method == METHOD.VPTQ
    assert reloaded.quantize_config.format == FORMAT.VPTQ
    assert reloaded.quantize_config.bits == qcfg.bits
    assert reloaded.quantize_config.num_centroids == qcfg.num_centroids
    assert reloaded.quantize_config.vector_lens == qcfg.vector_lens

    reloaded_modules = dict(reloaded.named_modules())
    for name in expected_modules:
        assert isinstance(reloaded_modules[name], VPTQLinear), name
        assert torch.equal(reloaded_modules[name].dequantize(), dict(model.named_modules())[name].dequantize())

    # the packed checkpoint must compute what dense inference with the
    # reconstructed weights computes
    with torch.no_grad():
        packed_logits = reloaded.model(**inputs).logits
        packed_tokens = reloaded.model.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert torch.isfinite(packed_logits).all()

    for name in expected_modules:
        module = reloaded_modules[name]
        # `reloaded.named_modules()` is rooted at the GPT-QModel wrapper
        parent_path, attribute = name[len("model."):].rsplit(".", 1)
        dense = torch.nn.Linear(
            module.in_features, module.out_features, bias=False, dtype=module.codebook_dtype
        )
        dense.weight.data.copy_(module.dequantize())
        setattr(reloaded.model.get_submodule(parent_path), attribute, dense)

    with torch.no_grad():
        dense_logits = reloaded.model(**inputs).logits
        dense_tokens = reloaded.model.generate(**inputs, max_new_tokens=4, do_sample=False)

    assert torch.allclose(packed_logits, dense_logits, atol=2e-3, rtol=2e-3)
    assert torch.equal(packed_tokens, dense_tokens)
    assert packed_tokens.shape[-1] == inputs["input_ids"].shape[-1] + 4


@pytest.mark.cuda
@pytest.mark.slow
def test_tiny_qwen3_vptq_cuda_smoke(tmp_path: Path):
    """Run the real calibration/quantize/pack lifecycle on an accelerator."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the VPTQ accelerator smoke test")

    native_dir = tmp_path / "native"
    native_dir.mkdir()
    _, tokenizer = _build_tiny_qwen3_fixture(native_dir)
    calibration = _build_calibration_dataset(tokenizer)

    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, 8],
        vector_lens=[-1, 8],
        kmeans_iters=8,
        device="cuda:0",
    )
    model = GPTQModel.load(
        str(native_dir),
        quantize_config=qcfg,
        backend=BACKEND.VPTQ_TORCH,
        dtype=torch.bfloat16,
    )
    model.quantize(calibration, batch_size=1, backend=BACKEND.VPTQ_TORCH, calibration_data_min_length=1)

    module = dict(model.named_modules())["model.model.layers.0.self_attn.q_proj"]
    assert isinstance(module, VPTQLinear)
    # packed modules are created on CPU and moved to the accelerator afterwards
    module = module.to("cuda")
    assert module.centroids.weight.device.type == "cuda"

    dense = module.dequantize()
    activations = torch.randn(2, module.in_features, dtype=torch.bfloat16, device=dense.device)
    assert module(activations).dtype == torch.bfloat16
    assert torch.allclose(
        module(activations.float()),
        activations.float() @ dense.float().t(),
        atol=1e-4,
        rtol=1e-4,
    )
