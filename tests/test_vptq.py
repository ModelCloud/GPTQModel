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
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.vptq import VPTQLinear
from gptqmodel.quantization import vptq as vptq_module
from gptqmodel.quantization.config import FORMAT, METHOD, Fallback, QuantizeConfig, VPTQConfig
from gptqmodel.quantization.vptq import (
    VPTQ,
    VPTQQuantResult,
    _chunked_assign,
    _matrix_to_vectors,
    _weighted_kmeans,
)
from gptqmodel.utils.vptq import (
    pack_vptq_indices,
    unpack_vptq_indices,
    vptq_group_for_column,
    vptq_group_setting,
    vptq_index_bits,
    vptq_outlier_enabled,
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

    # the last group is zero-padded up to `group_size`; those columns are not
    # part of the weight matrix
    if main.shape[1] > module.in_features:
        main = main[:, : module.in_features]

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


_ORACLE_EPS = 1e-12


def oracle_group_setting(
    in_features,
    *,
    npercent,
    group_size,
    group_num,
    enable_outlier,
):
    """Resolve the VPTQ group layout without calling the production helper."""

    initial_outlier_size = int(math.ceil((npercent / 100.0) * in_features))
    initial_outlier_size = max(0, min(in_features, initial_outlier_size))
    if not enable_outlier:
        initial_outlier_size = 0
    remaining_columns = in_features - initial_outlier_size

    if group_num != -1:
        if enable_outlier:
            pad = remaining_columns % group_num
            outlier_size = initial_outlier_size + pad
            resolved_group_size = (in_features - outlier_size) // group_num
        else:
            outlier_size = 0
            resolved_group_size = int(math.ceil(remaining_columns / group_num))
        resolved_group_num = group_num
    elif group_size != -1:
        outlier_size = initial_outlier_size
        resolved_group_num = int(math.ceil((in_features - outlier_size) / group_size))
        resolved_group_size = group_size
    else:
        outlier_size = initial_outlier_size
        resolved_group_num = 1
        resolved_group_size = in_features - outlier_size

    return outlier_size, resolved_group_size, resolved_group_num


def oracle_matrix_to_vectors(data, vector_len):
    """Split a matrix into vectors along rows using only test-local arithmetic."""

    rows, columns = data.shape
    pad = (-rows) % vector_len
    if pad:
        data = torch.cat(
            [data, torch.zeros((pad, columns), dtype=data.dtype, device=data.device)],
            dim=0,
        )
    vectors = data.permute(1, 0).reshape(columns, -1, vector_len).reshape(-1, vector_len)
    return vectors, rows + pad


def oracle_vectors_to_matrix(vectors, columns, padded_rows, rows):
    """Inverse of :func:`oracle_matrix_to_vectors`."""

    data = vectors.reshape(columns, padded_rows // vectors.shape[-1], vectors.shape[-1])
    return data.permute(1, 0, 2).reshape(padded_rows, columns)[:rows]


def oracle_assign(data, centroids):
    """Return nearest-centroid assignments and true squared distances."""

    centroids_norm = (centroids * centroids).sum(dim=1)
    distance = centroids_norm.unsqueeze(0) - 2.0 * (data @ centroids.t())
    best_distance, best_index = distance.min(dim=1)
    return best_index, best_distance + (data * data).sum(dim=1)


def oracle_weighted_kmeans(data, num_centroids, weights, *, iters, tol, seed):
    """Independent Lloyd implementation with the production initialization order."""

    num_rows = data.shape[0]
    generator = torch.Generator(device=data.device)
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_rows, generator=generator, device=data.device)
    if num_centroids <= num_rows:
        init_index = permutation[:num_centroids]
    else:
        repeats = (num_centroids + num_rows - 1) // num_rows
        init_index = permutation.repeat(repeats)[:num_centroids]
    centroids = data[init_index].clone()

    if weights is None:
        weights = torch.ones(num_rows, dtype=data.dtype, device=data.device)
    else:
        weights = weights.to(dtype=data.dtype, device=data.device)

    previous_inertia = None
    for _ in range(max(1, int(iters))):
        assignment, residual = oracle_assign(data, centroids)
        inertia = (residual.double() * weights.double()).sum().item()

        weighted = data * weights.unsqueeze(1)
        sums = torch.zeros_like(centroids).index_add_(0, assignment, weighted)
        counts = torch.zeros(
            num_centroids, dtype=data.dtype, device=data.device
        ).index_add_(0, assignment, weights)
        updated = sums / counts.clamp_min(_ORACLE_EPS).unsqueeze(1)
        empty = counts <= 0
        if bool(empty.any()):
            updated[empty] = centroids[empty]
        centroids = updated

        if previous_inertia is not None:
            denominator = max(abs(previous_inertia), _ORACLE_EPS)
            if abs(previous_inertia - inertia) <= tol * denominator:
                break
        previous_inertia = inertia

    return centroids


def oracle_inverse_hessian_cholesky(hessian, percdamp):
    """Independent `(H + damp * mean(diag(H)) I)^-1 = U^T U` reference."""

    damped = hessian.clone()
    diagonal = torch.arange(damped.shape[0], device=damped.device)
    damped[diagonal, diagonal] += float(percdamp) * float(torch.diagonal(hessian).mean())
    cholesky = torch.linalg.cholesky(damped)
    inverse = torch.cholesky_inverse(cholesky)
    return torch.linalg.cholesky(inverse, upper=True)


def oracle_vptq_pass(
    weight,
    hessian,
    group_bounds,
    vector_lens,
    centroids,
    res_centroids,
    enable_residual,
    outlier_size,
    group_size,
    block_size,
    percdamp,
):
    """Independent sequential VPTQ pass with GPTQ-style error propagation."""

    rows, columns = weight.shape
    device = weight.device
    index_tensors = {}
    res_index_tensors = {}
    for cidx, (start, end) in enumerate(group_bounds):
        codebook = centroids.get(cidx)
        if codebook is None or end <= start:
            index_tensors[cidx] = None
            res_index_tensors[cidx] = None
            continue
        vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
        num_indices = math.ceil(rows / vector_len)
        index_tensors[cidx] = torch.zeros(
            (num_indices, end - start), dtype=torch.int64, device=device
        )
        res_codebook = None if res_centroids is None else res_centroids.get(cidx)
        if enable_residual and res_codebook is not None:
            res_index_tensors[cidx] = torch.zeros_like(index_tensors[cidx])
        else:
            res_index_tensors[cidx] = None

    inv_hessian = oracle_inverse_hessian_cholesky(hessian, percdamp).to(torch.float32)
    inv_scale = 1.0 / torch.diagonal(inv_hessian)

    weight = weight.clone().float()
    quantized = torch.zeros_like(weight)
    qerror = torch.zeros_like(weight)

    for block_start in range(0, columns, block_size):
        block_end = min(block_start + block_size, columns)
        block_weight = weight[:, block_start:block_end].clone()
        block_qweight = torch.zeros_like(block_weight)
        block_error = torch.zeros_like(block_weight)
        block_inv = inv_hessian[block_start:block_end, block_start:block_end]

        for k in range(block_end - block_start):
            column_index = block_start + k
            column = block_weight[:, k : k + 1]
            cidx = (
                0
                if column_index < outlier_size
                else 1 + (column_index - outlier_size) // group_size
            )
            codebook = centroids.get(cidx)

            if codebook is None:
                q_column = column.clone()
                assignment = None
                res_assignment = None
            else:
                codebook = codebook.to(dtype=column.dtype)
                vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
                vectors, padded_rows = oracle_matrix_to_vectors(column, vector_len)
                assignment, _ = oracle_assign(vectors, codebook)
                selected = codebook[assignment]

                res_assignment = None
                res_codebook = (
                    None if res_centroids is None else res_centroids.get(cidx)
                )
                if res_codebook is not None:
                    res_codebook = res_codebook.to(dtype=column.dtype)
                    res_assignment, _ = oracle_assign(vectors - selected, res_codebook)
                    selected = selected + res_codebook[res_assignment]

                q_column = oracle_vectors_to_matrix(selected, 1, padded_rows, rows)

            block_qweight[:, k : k + 1] = q_column
            if assignment is not None and index_tensors.get(cidx) is not None:
                group_start = group_bounds[cidx][0]
                index_tensors[cidx][:, column_index - group_start] = assignment
            if res_assignment is not None and res_index_tensors.get(cidx) is not None:
                group_start = group_bounds[cidx][0]
                res_index_tensors[cidx][:, column_index - group_start] = res_assignment

            error = (column - q_column) * inv_scale[column_index]
            if k + 1 < block_end - block_start:
                block_weight[:, k + 1 :] -= error @ block_inv[k : k + 1, k + 1 :]
            block_error[:, k : k + 1] = error

        quantized[:, block_start:block_end] = block_qweight
        qerror[:, block_start:block_end] = block_weight - block_qweight
        if block_end < columns:
            weight[:, block_end:] -= (
                block_error @ inv_hessian[block_start:block_end, block_end:]
            )

    return quantized, qerror, index_tensors, res_index_tensors


def oracle_vptq_quantize(weight, hessian, qcfg, target_dtype=torch.float32):
    """Complete independent VPTQ reference used to audit the production quantizer."""

    vector_lens = [int(item) for item in qcfg.vector_lens]
    num_centroids = [int(item) for item in qcfg.num_centroids]
    num_res_centroids = [int(item) for item in qcfg.num_res_centroids]
    enable_outlier = (
        vector_lens[0] > 0
        and num_centroids[0] > 1
        and qcfg.npercent > 0
    )
    outlier_size, group_size, group_num = oracle_group_setting(
        weight.shape[1],
        npercent=qcfg.npercent,
        group_size=qcfg.group_size,
        group_num=qcfg.group_num,
        enable_outlier=enable_outlier,
    )

    weight = weight.clone()
    hessian = hessian.to(device=weight.device, dtype=torch.float64).clone()
    dead = torch.diagonal(hessian) == 0
    if bool(dead.any()):
        hessian[dead, dead] = 1.0
        weight[:, dead] = 0.0

    mean_diag = torch.diagonal(hessian).mean()
    if float(mean_diag) > 0:
        hessian = hessian / mean_diag

    weight_scale = None
    weight_bias = None
    if qcfg.enable_norm:
        norm_dim = int(qcfg.norm_dim)
        weight_min = weight.amin(dim=norm_dim)
        weight_max = weight.amax(dim=norm_dim)
        scale = (weight_max - weight_min).clamp_min(_ORACLE_EPS)
        weight = (weight - weight_min.unsqueeze(norm_dim)) / scale.unsqueeze(norm_dim)
        weight_scale = scale
        weight_bias = weight_min

    kmeans_weight = None
    if qcfg.kmeans_mode == "hessian":
        kmeans_weight = (
            torch.diagonal(hessian)
            .to(weight.dtype)
            .unsqueeze(0)
            .expand(weight.shape[0], weight.shape[1])
            .contiguous()
        )

    perm = None
    if qcfg.enable_perm:
        perm = torch.argsort(torch.diagonal(hessian), descending=True)
        weight = weight[:, perm]
        hessian = hessian[perm][:, perm]
        if kmeans_weight is not None:
            kmeans_weight = kmeans_weight[:, perm]

    group_bounds = [(0, outlier_size)] + [
        (
            outlier_size + index * group_size,
            min(weight.shape[1], outlier_size + (index + 1) * group_size),
        )
        for index in range(group_num)
    ]

    centroids = {}
    working_centroids = {}
    for cidx, (start, end) in enumerate(group_bounds):
        vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
        num_centroids_for_group = num_centroids[0] if cidx == 0 else num_centroids[1]
        if num_centroids_for_group <= 1 or end <= start:
            centroids[cidx] = None
            working_centroids[cidx] = None
            continue

        vectors, _ = oracle_matrix_to_vectors(weight[:, start:end], vector_len)
        group_weights = None
        if kmeans_weight is not None:
            weight_vectors, _ = oracle_matrix_to_vectors(
                kmeans_weight[:, start:end], vector_len
            )
            group_weights = weight_vectors.mean(dim=1)
        group_centroids = oracle_weighted_kmeans(
            vectors,
            num_centroids_for_group,
            group_weights,
            iters=qcfg.kmeans_iters,
            tol=qcfg.kmeans_tol,
            seed=qcfg.kmeans_seed,
        )
        centroids[cidx] = group_centroids.to(dtype=target_dtype)
        working_centroids[cidx] = centroids[cidx].to(dtype=torch.float32)

    res_centroids = dict.fromkeys(centroids)
    working_res_centroids = dict.fromkeys(centroids)
    enable_residual = num_res_centroids[1] > 1

    quantized, qerror, indices, res_indices = oracle_vptq_pass(
        weight=weight,
        hessian=hessian,
        group_bounds=group_bounds,
        vector_lens=vector_lens,
        centroids=working_centroids,
        res_centroids=None,
        enable_residual=False,
        outlier_size=outlier_size,
        group_size=group_size,
        block_size=max(1, int(qcfg.block_size)),
        percdamp=qcfg.percdamp,
    )

    if enable_residual:
        for cidx, (start, end) in enumerate(group_bounds):
            vector_len = vector_lens[0] if cidx == 0 else vector_lens[1]
            num_res = num_res_centroids[0] if cidx == 0 else num_res_centroids[1]
            if num_res <= 1 or end <= start:
                continue

            residual_vectors, _ = oracle_matrix_to_vectors(
                qerror[:, start:end], vector_len
            )
            group_weights = None
            if kmeans_weight is not None:
                weight_vectors, _ = oracle_matrix_to_vectors(
                    kmeans_weight[:, start:end], vector_len
                )
                group_weights = weight_vectors.mean(dim=1)
            group_res_centroids = oracle_weighted_kmeans(
                residual_vectors,
                num_res,
                group_weights,
                iters=qcfg.kmeans_iters,
                tol=qcfg.kmeans_tol,
                seed=qcfg.kmeans_seed,
            )
            res_centroids[cidx] = group_res_centroids.to(dtype=target_dtype)
            working_res_centroids[cidx] = res_centroids[cidx].to(dtype=torch.float32)

        quantized, qerror, indices, res_indices = oracle_vptq_pass(
            weight=weight,
            hessian=hessian,
            group_bounds=group_bounds,
            vector_lens=vector_lens,
            centroids=working_centroids,
            res_centroids=working_res_centroids,
            enable_residual=True,
            outlier_size=outlier_size,
            group_size=group_size,
            block_size=max(1, int(qcfg.block_size)),
            percdamp=qcfg.percdamp,
        )

    if perm is not None:
        quantized = quantized[:, torch.argsort(perm)]

    if qcfg.enable_norm:
        norm_dim = int(qcfg.norm_dim)
        scale = weight_scale.to(dtype=target_dtype)
        bias = weight_bias.to(dtype=target_dtype)
        quantized = (
            quantized.to(dtype=target_dtype) * scale.unsqueeze(norm_dim)
            + bias.unsqueeze(norm_dim)
        ).to(dtype=torch.float32)

    return {
        "centroids": centroids,
        "indices": indices,
        "res_centroids": res_centroids,
        "res_indices": res_indices,
        "weight_scale": weight_scale,
        "weight_bias": weight_bias,
        "perm": perm,
        "outlier_size": outlier_size,
        "group_size": group_size,
        "group_num": group_num,
        "quantized_weight": quantized.to(dtype=target_dtype),
    }


def build_quantizer(weight, qcfg, dtype=torch.float32, *, use_fallback=False):
    """Attach a VPTQ quantizer to a real `nn.Linear` holding `weight`."""

    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=dtype)
    linear.weight.data.copy_(weight.to(dtype))
    quantizer = VPTQ(module=linear, qcfg=qcfg)
    if not use_fallback:
        # Tests that pass a synthetic Hessian exercise that Hessian directly;
        # the calibration-coverage fallback is tested separately.
        quantizer.fallback = None
    return quantizer


def build_module(
    qcfg,
    in_features,
    out_features,
    dtype=torch.float32,
    register_buffers=False,
    bias=False,
    adapter=None,
):
    """Create a `VPTQLinear` matching `qcfg`."""

    return VPTQLinear(
        bits=qcfg.bits,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
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
        adapter=adapter,
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


@pytest.mark.parametrize(
    ("indices", "res_indices", "index_bits", "res_bits"),
    [
        ([[-1, 0]], [[0, 0]], 2, 2),
        ([[4, 0]], [[0, 0]], 2, 2),
        ([[0, 0]], [[0, 4]], 2, 2),
        ([[0, 0]], [[-1, 0]], 2, 2),
    ],
    ids=[
        "negative-main",
        "main-overflow-with-residual",
        "residual-overflow",
        "negative-residual",
    ],
)
def test_vptq_pack_rejects_out_of_range_index_streams(
    indices, res_indices, index_bits, res_bits
):
    """Each stream must fit its own field instead of spilling into the other."""

    with pytest.raises(ValueError, match="pack_vptq_indices"):
        pack_vptq_indices(
            torch.tensor(indices, dtype=torch.int64),
            index_bits,
            res_indices=torch.tensor(res_indices, dtype=torch.int64),
            res_bits=res_bits,
        )


def test_vptq_pack_empty_and_invalid():
    empty = pack_vptq_indices(torch.zeros((2, 0), dtype=torch.int64), 4)
    assert empty.shape == (2, 0)
    indices, res_indices = unpack_vptq_indices(empty, 4, 0, res_bits=0)
    assert indices.shape == (2, 0) and res_indices is None

    with pytest.raises(ValueError):
        pack_vptq_indices(torch.zeros((1, 2), dtype=torch.int64), 0)
    with pytest.raises(ValueError):
        pack_vptq_indices(torch.zeros((1, 2), dtype=torch.int64), 4, res_bits=2)


def test_vptq_pack_rejects_index_streams_wider_than_a_word():
    """A 32-bit word cannot hold an element that straddles two words."""

    indices = torch.zeros((1, 2), dtype=torch.int64)
    res_indices = torch.zeros((1, 2), dtype=torch.int64)

    with pytest.raises(ValueError):
        pack_vptq_indices(indices, 17, res_indices=res_indices, res_bits=16)
    with pytest.raises(ValueError):
        pack_vptq_indices(indices, 33)
    # exactly 32 bits per element is the widest supported stream
    packed = pack_vptq_indices(indices, 32)
    unpacked, _ = unpack_vptq_indices(packed, 32, 2)
    assert torch.equal(unpacked, indices)
    with pytest.raises(ValueError):
        unpack_vptq_indices(packed, 33, 2)


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


def test_vptq_group_layout_without_outlier_codebook():
    """Without an outlier codebook the last group carries the remainder."""

    # `group_num` divides 64 - 4 = 60 evenly here, so nothing is padded
    assert vptq_group_setting(64, npercent=6.25, group_size=-1, group_num=4, enable_outlier=True) == (
        4,
        15,
        4,
    )
    # the same remainder is moved out of the outlier set when outliers are off
    assert vptq_group_setting(64, npercent=6.25, group_size=-1, group_num=4, enable_outlier=False) == (
        0,
        16,
        4,
    )
    # 65 channels over 4 groups: the last group is one channel short
    outlier_size, group_size, group_num = vptq_group_setting(
        65, npercent=0.0, group_size=-1, group_num=4, enable_outlier=False
    )
    assert (outlier_size, group_size, group_num) == (0, 17, 4)
    assert group_size * group_num >= 65
    assert vptq_group_for_column(64, outlier_size=outlier_size, group_size=group_size) == group_num


def test_vptq_group_num_cannot_exceed_available_channels():
    """An oversized `group_num` would create codebook groups without any channel."""

    # one channel per codebook is the smallest legal grouping
    assert vptq_group_setting(64, npercent=0.0, group_size=-1, group_num=64, enable_outlier=False) == (
        0,
        1,
        64,
    )
    with pytest.raises(ValueError, match="empty codebook groups"):
        vptq_group_setting(64, npercent=0.0, group_size=-1, group_num=65, enable_outlier=False)

    # with an outlier codebook the remainder moves into the outlier set, so the
    # limit applies to the channels left over after the outlier split
    outlier_size, group_size, group_num = vptq_group_setting(
        64, npercent=10.0, group_size=-1, group_num=57, enable_outlier=True
    )
    assert (outlier_size, group_size, group_num) == (7, 1, 57)
    with pytest.raises(ValueError, match="empty codebook groups"):
        vptq_group_setting(64, npercent=10.0, group_size=-1, group_num=58, enable_outlier=True)


def test_vptq_group_num_equal_to_in_features_roundtrip():
    """`group_num == in_features` is legal and packs one codebook per channel."""

    weight, hessian = synthetic_problem(out_features=32, in_features=64, seed=24)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_num=64,
        kmeans_iters=4,
    )
    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)
    assert all(result.centroids[cidx] is not None for cidx in range(1, 65))

    module = build_module(qcfg, 64, 32)
    module.pack(linear=quantizer.layer, vptq_result=result)
    dequantized = module.dequantize()
    assert dequantized.shape == weight.shape
    assert torch.equal(dequantized, result.quantized_weight)

    oversized = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_num=65,
        kmeans_iters=4,
    )
    with pytest.raises(ValueError, match="empty codebook groups"):
        build_module(oversized, 64, 32)
    with pytest.raises(ValueError, match="empty codebook groups"):
        build_quantizer(weight, oversized).quantize_weight(weight, hessian)


def test_vptq_outlier_codebook_resolution():
    """`npercent` alone does not enable the outlier codebook."""

    assert not vptq_outlier_enabled(npercent=0.0, outlier_vector_len=4, outlier_num_centroids=16)
    assert not vptq_outlier_enabled(npercent=10.0, outlier_vector_len=-1, outlier_num_centroids=16)
    assert not vptq_outlier_enabled(npercent=10.0, outlier_vector_len=4, outlier_num_centroids=-1)
    assert not vptq_outlier_enabled(npercent=10.0, outlier_vector_len=4, outlier_num_centroids=1)
    assert vptq_outlier_enabled(npercent=10.0, outlier_vector_len=4, outlier_num_centroids=16)


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


def test_vptq_config_rejects_unusable_codebooks():
    """Configurations the packer or the runtime cannot represent must fail early."""

    # a single-entry codebook needs zero index bits, which cannot be packed
    with pytest.raises(ValueError):
        VPTQConfig(num_centroids=[-1, 1], num_res_centroids=[-1, -1], vector_lens=[-1, 4])
    with pytest.raises(ValueError, match=r"num_res_centroids\[1\].*at least 2"):
        VPTQConfig(num_centroids=[-1, 16], num_res_centroids=[-1, 1], vector_lens=[-1, 4])
    # more than 32 bits per element cannot fit the int32 word stream
    with pytest.raises(ValueError):
        VPTQConfig(num_centroids=[-1, 2**17], num_res_centroids=[-1, 2**16], vector_lens=[-1, 8])
    # the runtime has no outlier residual codebook
    with pytest.raises(ValueError):
        VPTQConfig(num_centroids=[-1, 64], num_res_centroids=[16, 16], vector_lens=[-1, 4])
    # `npercent` needs a real outlier codebook
    with pytest.raises(ValueError):
        VPTQConfig(num_centroids=[-1, -1], num_res_centroids=[-1, -1], vector_lens=[-1, 4], npercent=10.0)
    with pytest.raises(ValueError):
        VPTQConfig(num_centroids=[-1, 64], num_res_centroids=[-1, -1], vector_lens=[4, 8], npercent=10.0)


def test_vptq_config_rejects_one_entry_residual_on_reload():
    """A persisted one-entry residual codebook must not reload as disabled."""

    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
    )
    payload = qcfg.to_dict()
    payload["num_res_centroids"] = [-1, 1]
    with pytest.raises(ValueError, match=r"num_res_centroids\[1\].*at least 2"):
        QuantizeConfig.from_quant_config(payload)


def test_vptq_linear_rejects_one_entry_residual_codebook():
    """Direct runtime construction must not silently disable a requested residual stage."""

    with pytest.raises(ValueError, match=r"num_res_centroids\[1\].*at least 2"):
        VPTQLinear(
            bits=2,
            in_features=16,
            out_features=8,
            vector_lens=(-1, 4),
            num_centroids=(-1, 16),
            num_res_centroids=(-1, 1),
            dtype=torch.float32,
        )


def test_vptq_config_round_trip_keeps_group_size():
    """`group_size` must survive the saved payload so reloads rebuild the layout."""

    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_size=8,
    )
    payload = qcfg.to_dict()
    assert payload["group_size"] == 8

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, VPTQConfig)
    assert restored.group_size == 8
    assert restored.group_num == -1


def test_vptq_module_built_from_config_matches_quantizer_layout():
    """The kernel must be built with the group layout the quantizer used."""

    weight, hessian = synthetic_problem(out_features=32, in_features=64, seed=25)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_size=8,
        kmeans_iters=4,
    )
    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)

    module = VPTQLinear(
        bits=qcfg.bits,
        # `create_quant_module` passes the generic fields explicitly and the
        # method-specific ones from `quant_linear_init_kwargs()`
        group_size=qcfg.group_size,
        desc_act=qcfg.desc_act,
        sym=qcfg.sym,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=False,
        # the quantizer targeted the fp32 layer weight, so the kernel must store
        # its codebooks in the same dtype for a bit-exact round trip
        dtype=torch.float32,
        **qcfg.quant_linear_init_kwargs(),
    )
    assert module.group_size == 8
    assert module.group_num == 8
    module.pack(linear=quantizer.layer, vptq_result=result)
    assert torch.equal(module.dequantize(), result.quantized_weight)


# ---------------------------------------------------------------------------
# quantization -> module round trip
# ---------------------------------------------------------------------------
VPTQ_VARIANTS = [
    ("baseline", {}),
    ("group_num", {"group_num": 4}),
    ("group_size_remainder", {"group_size": 6}),
    ("group_num_remainder", {"group_num": 5}),
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
    assert result.stats["fallback"] is False

    module = build_module(qcfg, weight.shape[1], weight.shape[0], dtype=dtype)
    module.pack(linear=quantizer.layer, vptq_result=result)

    dequantized = module.dequantize()
    assert dequantized.shape == weight.shape
    assert dequantized.dtype == dtype
    # quantizer reconstruction (vectorized) == module unpack+lookup == oracle
    assert torch.equal(dequantized, result.quantized_weight)
    assert torch.equal(dequantized.float(), oracle_reconstruct(result, module))


VPTQ_ORACLE_VARIANTS = [
    ("baseline", {}),
    ("group_size_remainder", {"group_size": 6}),
    ("group_num", {"group_num": 4}),
    ("perm", {"enable_perm": True}),
    ("norm", {"enable_norm": True}),
    ("residual", {"num_res_centroids": [-1, 8]}),
    ("outlier", {"npercent": 12.5, "vector_lens": [4, 8], "num_centroids": [8, 32]}),
    (
        "all",
        {
            "npercent": 12.5,
            "vector_lens": [4, 8],
            "num_centroids": [8, 32],
            "num_res_centroids": [-1, 8],
            "enable_perm": True,
            "enable_norm": True,
            "group_num": 2,
        },
    ),
]


@pytest.mark.parametrize(
    ("variant", "overrides"),
    VPTQ_ORACLE_VARIANTS,
    ids=[item[0] for item in VPTQ_ORACLE_VARIANTS],
)
def test_vptq_quantizer_matches_independent_oracle(variant, overrides):
    """The complete quantizer must match an independent Torch reference exactly."""

    seed = 40 + [item[0] for item in VPTQ_ORACLE_VARIANTS].index(variant)
    weight, hessian = synthetic_problem(out_features=24, in_features=32, seed=seed)
    settings = {
        "num_centroids": [-1, 32],
        "num_res_centroids": [-1, -1],
        "vector_lens": [-1, 4],
        "kmeans_iters": 6,
        "kmeans_tol": 1e-12,
        "block_size": 16,
    }
    settings.update(overrides)
    qcfg = VPTQConfig(**settings)

    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)
    assert result.stats["fallback"] is False

    expected = oracle_vptq_quantize(weight, hessian, qcfg)
    assert (
        result.outlier_size,
        result.group_size,
        result.group_num,
    ) == (
        expected["outlier_size"],
        expected["group_size"],
        expected["group_num"],
    )

    for field in ("centroids", "res_centroids"):
        actual_map = getattr(result, field)
        expected_map = expected[field]
        assert set(actual_map) == set(expected_map)
        for cidx, expected_tensor in expected_map.items():
            actual_tensor = actual_map[cidx]
            if expected_tensor is None:
                assert actual_tensor is None
            else:
                assert actual_tensor is not None
                assert torch.allclose(
                    actual_tensor, expected_tensor, rtol=1e-6, atol=1e-6
                ), f"{field}[{cidx}] drift {float((actual_tensor - expected_tensor).abs().max())}"

    for field in ("indices", "res_indices"):
        actual_map = getattr(result, field)
        expected_map = expected[field]
        assert set(actual_map) == set(expected_map)
        for cidx, expected_tensor in expected_map.items():
            actual_tensor = actual_map[cidx]
            if expected_tensor is None:
                assert actual_tensor is None
            else:
                assert actual_tensor is not None
                assert torch.equal(actual_tensor, expected_tensor), f"{field}[{cidx}] mismatch"

    for field in ("weight_scale", "weight_bias", "perm"):
        actual_tensor = getattr(result, field)
        expected_tensor = expected[field]
        if expected_tensor is None:
            assert actual_tensor is None
        else:
            assert actual_tensor is not None
            assert torch.equal(actual_tensor, expected_tensor), field

    drift = (result.quantized_weight - expected["quantized_weight"]).abs()
    assert torch.allclose(
        result.quantized_weight, expected["quantized_weight"], rtol=1e-6, atol=1e-6
    ), f"quantized weight max drift {float(drift.max())}"


def test_vptq_norm_scale_clamp_matches_independent_oracle():
    """The normalization scale clamp must match the independent reference."""

    weight = torch.full((8, 16), 0.25)
    hessian = torch.eye(16, dtype=torch.float64)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 8],
        num_res_centroids=[-1, -1],
        enable_norm=True,
        kmeans_iters=2,
        kmeans_tol=1e-12,
        block_size=8,
    )

    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)
    assert result.stats["fallback"] is False

    expected = oracle_vptq_quantize(weight, hessian, qcfg)
    assert torch.allclose(result.weight_scale, expected["weight_scale"], rtol=1e-6, atol=1e-6)
    assert torch.allclose(
        result.quantized_weight,
        expected["quantized_weight"],
        rtol=1e-6,
        atol=1e-6,
    )


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


def test_vptq_outlier_indices_beyond_int16_survive_pack():
    """Outlier codes above 32767 must not wrap in the outlier index buffer."""

    in_features, out_features = 8, 4
    qcfg = VPTQConfig(
        vector_lens=[1, 4],
        num_centroids=[65536, 16],
        num_res_centroids=[-1, -1],
        npercent=50.0,
    )
    module = build_module(qcfg, in_features, out_features)
    assert module.enable_outlier

    generator = torch.Generator().manual_seed(13)
    outlier_centroids = torch.randn(65536, 1, generator=generator)
    main_centroids = torch.randn(16, 4, generator=generator)
    outlier_indices = torch.tensor(
        [[40000, 1, 2, 3], [65000, 5, 6, 7], [1, 2, 3, 4], [2, 3, 4, 5]],
        dtype=torch.int64,
    )
    result = VPTQQuantResult(
        centroids={0: outlier_centroids, 1: main_centroids},
        indices={0: outlier_indices, 1: torch.zeros(1, module.group_size, dtype=torch.int64)},
        res_centroids={0: None, 1: None},
        res_indices={0: None, 1: None},
        outlier_size=module.outlier_size,
        group_size=module.group_size,
        group_num=module.group_num,
        index_bits=module.index_bits,
        res_index_bits=0,
    )
    module.pack(vptq_result=result)
    assert module.outlier_indices.dtype == torch.int32

    dequantized = module.dequantize()
    expected_outlier = outlier_centroids[outlier_indices].squeeze(-1)
    assert torch.equal(dequantized[:, : module.outlier_size], expected_outlier)


def test_vptq_group_padding_columns_are_dropped():
    """A partial last group must not widen the dequantized weight."""

    weight, hessian = synthetic_problem(out_features=32, in_features=100, seed=21)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_size=16,
        kmeans_iters=4,
    )
    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)

    module = build_module(qcfg, weight.shape[1], weight.shape[0])
    # 7 groups of 16 channels cover 112 columns, i.e. 12 padding columns
    assert module.group_num * module.group_size == 112
    module.pack(linear=quantizer.layer, vptq_result=result)

    dequantized = module.dequantize()
    assert dequantized.shape == weight.shape
    assert torch.equal(dequantized, result.quantized_weight)
    assert torch.equal(dequantized.float(), oracle_reconstruct(result, module))
    assert module(torch.randn(3, weight.shape[1])).shape == (3, weight.shape[0])


def test_vptq_group_num_remainder_without_outlier_codebook():
    """`group_num` must not move leftover channels into a disabled outlier codebook."""

    weight, hessian = synthetic_problem(out_features=32, in_features=65, seed=22)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_num=4,
        kmeans_iters=4,
    )
    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)

    module = build_module(qcfg, weight.shape[1], weight.shape[0])
    assert module.outlier_size == 0
    # 4 groups of 17 channels cover 68 columns, i.e. 3 padding columns
    assert module.group_num * module.group_size == 68
    module.pack(linear=quantizer.layer, vptq_result=result)

    dequantized = module.dequantize()
    assert dequantized.shape == weight.shape
    assert torch.equal(dequantized, result.quantized_weight)
    assert torch.equal(dequantized.float(), oracle_reconstruct(result, module))


def test_vptq_zero_hessian_uses_hessian_free_fallback():
    """A module with no calibration traffic keeps its weights instead of zeroing them."""

    weight, _hessian = synthetic_problem(out_features=16, in_features=32, seed=23)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        kmeans_iters=8,
    )
    quantizer = build_quantizer(weight, qcfg)
    result, loss = quantizer.quantize_weight(weight, torch.zeros(32, 32, dtype=torch.float64))

    assert result.stats["fallback"] is True
    assert math.isfinite(loss)
    assert result.quantized_weight.abs().max().item() > 0

    # the fallback is plain vector quantization: every sub-vector of the
    # original weight lands on its nearest codebook entry, with no Hessian
    # error compensation between columns
    codebook = result.centroids[1].float()
    vectors, _ = _matrix_to_vectors(weight, 4)
    quantized_vectors, _ = _matrix_to_vectors(result.quantized_weight.float(), 4)
    error = ((quantized_vectors - vectors) ** 2).sum(dim=1)
    nearest = ((vectors.unsqueeze(1) - codebook.unsqueeze(0)) ** 2).sum(dim=-1).min(dim=1).values
    assert torch.allclose(error, nearest, atol=1e-5)

    module = build_module(qcfg, weight.shape[1], weight.shape[0])
    module.pack(linear=quantizer.layer, vptq_result=result)
    assert torch.equal(module.dequantize(), result.quantized_weight)


def test_vptq_hessian_usability_rejects_zero_and_nonfinite():
    """The combined device predicate must preserve the Hessian validity rules."""

    assert VPTQ._hessian_is_usable(torch.eye(4, dtype=torch.float64))
    assert not VPTQ._hessian_is_usable(torch.zeros(4, 4, dtype=torch.float64))

    nan_hessian = torch.eye(4, dtype=torch.float64)
    nan_hessian[0, 0] = float("nan")
    assert not VPTQ._hessian_is_usable(nan_hessian)

    inf_hessian = torch.eye(4, dtype=torch.float64)
    inf_hessian[0, 1] = float("inf")
    assert not VPTQ._hessian_is_usable(inf_hessian)


def test_vptq_weighted_kmeans_defers_inertia_read_until_after_update():
    """The convergence read must not block the independent centroid update."""

    ops = []

    class Recorder(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ops.append(str(func))
            return func(*args, **(kwargs or {}))

    data = torch.randn(32, 4)
    weights = torch.rand(32)
    with Recorder():
        centroids, assignment = _weighted_kmeans(
            data,
            8,
            weights,
            iters=2,
            tol=0.0,
            seed=0,
        )

    assert centroids.shape == (8, 4)
    assert assignment.shape == (32,)
    inertia_read = ops.index("aten._local_scalar_dense.default")
    centroid_update = ops.index("aten.index_add_.default")
    assert inertia_read > centroid_update


def test_vptq_partial_dead_hessian_matches_oracle():
    """Dead channels must be neutralized without changing the oracle result."""

    weight, hessian = synthetic_problem(out_features=16, in_features=32, seed=25)
    dead = torch.tensor([1, 7, 19])
    hessian[dead, :] = 0.0
    hessian[:, dead] = 0.0
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        kmeans_iters=4,
        block_size=8,
    )

    result, loss = build_quantizer(weight, qcfg).quantize_weight(weight, hessian)
    expected = oracle_vptq_quantize(weight, hessian, qcfg)

    assert math.isfinite(loss)
    assert torch.allclose(
        result.quantized_weight,
        expected["quantized_weight"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_vptq_hessian_weighted_error_accumulates_chunks_on_device(monkeypatch):
    """Chunked loss accumulation must match a single independent float64 oracle."""

    generator = torch.Generator().manual_seed(31)
    weight = torch.randn(17, 9, generator=generator, dtype=torch.float64)
    quantized = weight + torch.randn(
        17, 9, generator=generator, dtype=torch.float64
    ) * 0.01
    hessian = torch.randn(9, 9, generator=generator, dtype=torch.float64)
    hessian = hessian @ hessian.mT

    delta = quantized - weight
    expected_error = ((delta @ hessian) * delta).sum()
    expected_weight = ((weight @ hessian) * weight).sum()
    expected = float(expected_error / max(abs(float(expected_weight)), 1e-12))

    monkeypatch.setattr(vptq_module, "_LOSS_CHUNK_BYTES", 9 * 8 * 2)
    actual = VPTQ.hessian_weighted_error(
        weight_original=weight,
        quantized=quantized,
        hessian=hessian,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_vptq_fallback_threshold_switches_to_hessian_free_vq():
    """The configured `fallback` threshold must switch low-coverage modules."""

    weight, hessian = synthetic_problem(out_features=16, in_features=32, seed=24)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        kmeans_iters=8,
    )

    low_sample = build_quantizer(weight, qcfg)
    low_sample.fallback = Fallback(threshold="50%")
    low_sample.expected_nsamples = 10000.0
    low_sample.nsamples = 10
    low_sample_result, _ = low_sample.quantize_weight(weight, hessian)
    assert low_sample_result.stats["fallback"] is True

    covered = build_quantizer(weight, qcfg)
    covered.fallback = Fallback(threshold="50%")
    covered.expected_nsamples = 10000.0
    covered.nsamples = 9000
    covered_result, _ = covered.quantize_weight(weight, hessian)
    assert covered_result.stats["fallback"] is False

    # the Hessian-aware path must win on the Hessian metric it optimizes
    low_sample_loss = VPTQ.hessian_weighted_error(
        weight_original=weight, quantized=low_sample_result.quantized_weight, hessian=hessian
    )
    covered_loss = VPTQ.hessian_weighted_error(
        weight_original=weight, quantized=covered_result.quantized_weight, hessian=hessian
    )
    assert covered_loss < low_sample_loss


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

    base_result, base_loss = build_quantizer(weight, base).quantize_weight(weight, hessian)
    residual_result, residual_loss = build_quantizer(
        weight, with_residual
    ).quantize_weight(weight, hessian)
    assert base_result.stats["fallback"] is False
    assert residual_result.stats["fallback"] is False
    assert residual_loss < base_loss


def test_vptq_nearest_centroid_tie_breaks_to_lower_index():
    """An exact tie must pick the lower codebook index; one ULP must not."""

    codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    tie = torch.tensor([[0.5, 0.0]])
    assignment, _ = _chunked_assign(tie, codebook)
    assert int(assignment[0]) == 0
    assert int(oracle_assign(tie, codebook)[0][0]) == 0

    below = torch.tensor([[0.5 - 1e-6, 0.0]])
    above = torch.tensor([[0.5 + 1e-6, 0.0]])
    assert int(_chunked_assign(below, codebook)[0][0]) == 0
    assert int(_chunked_assign(above, codebook)[0][0]) == 1
    assert int(oracle_assign(below, codebook)[0][0]) == 0
    assert int(oracle_assign(above, codebook)[0][0]) == 1


def test_vptq_chunked_assign_returns_true_squared_distance():
    """The residual must be `||x - c||^2`, not the expanded form without `||x||^2`."""

    generator = torch.Generator().manual_seed(17)
    data = torch.randn(37, 4, generator=generator)
    centroids = torch.randn(6, 4, generator=generator)

    assignment, residual = _chunked_assign(data, centroids)
    expected = ((data - centroids[assignment]) ** 2).sum(dim=1)
    assert torch.allclose(residual, expected, rtol=1e-6, atol=1e-6)
    # the k-means inertia depends on the absolute distance, not a per-row offset
    assert float(residual.min()) >= 0.0

    exact_assignment, exact_residual = _chunked_assign(centroids[:2], centroids)
    assert exact_assignment.tolist() == [0, 1]
    assert torch.allclose(exact_residual, torch.zeros(2), atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [(6,), (2, 6), (2, 3, 6), (2, 3, 5, 6)],
    ids=["rank1", "rank2", "rank3", "rank4"],
)
def test_vptq_quantizer_accumulates_calibration_hessian(shape):
    """`add_batch` must average `x^T x` over the calibration tokens."""

    qcfg = VPTQConfig(num_centroids=[-1, 16], num_res_centroids=[-1, -1], vector_lens=[-1, 4])
    quantizer = build_quantizer(torch.zeros(8, 6), qcfg)

    generator = torch.Generator().manual_seed(3)
    first = torch.randn(*shape, generator=generator, dtype=torch.float64)
    second = torch.randn(*shape, generator=generator, dtype=torch.float64)
    quantizer.add_batch(first)
    quantizer.add_batch(second)

    tokens = torch.cat([first.reshape(-1, 6), second.reshape(-1, 6)], dim=0)
    expected = (tokens.t() @ tokens) / tokens.shape[0]
    hessian = quantizer.materialize_hessian()
    normalized_error = float((hessian - expected).norm() / expected.norm())
    assert torch.isfinite(hessian).all()
    assert normalized_error <= 1e-6
    assert torch.allclose(hessian, expected, rtol=1e-12, atol=1e-12)
    assert quantizer.nsamples == tokens.shape[0]

    with pytest.raises(ValueError, match="at least one dimension"):
        quantizer.add_batch(torch.tensor(1.0))
    with pytest.raises(ValueError, match="expected 6 input features"):
        quantizer.add_batch(torch.randn(5))


def test_vptq_percdamp_is_applied_exactly_once():
    """`percdamp` must be the only damping: `U^T U == (H + percdamp * I)^-1`."""

    raw = torch.diag(torch.linspace(0.5, 4.0, 32, dtype=torch.float64))
    # `quantize_weight` normalizes the Hessian to mean diagonal 1 before the pass
    normalized = raw / raw.diagonal().mean()
    identity = torch.eye(32, dtype=torch.float64)

    inverses = {}
    for percdamp in (0.0, 0.01, 0.25):
        qcfg = VPTQConfig(
            vector_lens=[-1, 4],
            num_centroids=[-1, 16],
            num_res_centroids=[-1, -1],
            kmeans_iters=2,
            percdamp=percdamp,
        )
        quantizer = build_quantizer(torch.zeros(16, 32), qcfg)
        upper = quantizer._inverse_hessian_cholesky(normalized.clone())
        inverse = upper.t().double() @ upper.double()
        oracle = torch.linalg.inv(normalized + percdamp * identity)

        # a float64 oracle keeps this far below the 1e-6 matrix tolerance
        assert torch.allclose(inverse, oracle, rtol=1e-9, atol=1e-9), (
            f"percdamp={percdamp}: max drift {float((inverse - oracle).abs().max())}"
        )
        inverses[percdamp] = inverse

    # percdamp=0 applies no damping at all, and nonzero values change the matrix
    assert torch.allclose(inverses[0.0], torch.linalg.inv(normalized), rtol=1e-9, atol=1e-9)
    assert not torch.allclose(inverses[0.0], inverses[0.01])
    assert not torch.allclose(inverses[0.01], inverses[0.25])


def test_vptq_percdamp_reaches_the_error_propagation_pass():
    """Changing `percdamp` must change the quantized weight, not just a helper."""

    generator = torch.Generator().manual_seed(25)
    weight = torch.randn(24, 32, generator=generator) * 0.05
    # correlated features: the Hessian stays positive definite but the
    # inverse used by the pass is sensitive to the requested damping
    features = torch.randn(32, 32, generator=generator)
    hessian = features.t() @ features
    hessian = (hessian / hessian.diagonal().mean()).double()

    quantized = {}
    for percdamp in (0.0, 0.9):
        qcfg = VPTQConfig(
            vector_lens=[-1, 4],
            num_centroids=[-1, 64],
            num_res_centroids=[-1, -1],
            kmeans_iters=8,
            percdamp=percdamp,
        )
        quantizer = build_quantizer(weight, qcfg)
        # the synthetic Hessian stands in for the calibration Hessian, so the
        # sample-count fallback must not replace it with the identity
        quantizer.fallback = None
        result, _loss = quantizer.quantize_weight(weight, hessian)
        quantized[percdamp] = result.quantized_weight

    assert not torch.allclose(quantized[0.0], quantized[0.9])


def test_vptq_rejects_convolution_modules():
    """The dense-weight runtime cannot preserve convolutional semantics."""

    qcfg = VPTQConfig(vector_lens=[-1, 4], num_centroids=[-1, 16], kmeans_iters=2)
    for layer in (
        torch.nn.Conv1d(8, 6, kernel_size=3, bias=False),
        torch.nn.Conv2d(4, 6, kernel_size=3, bias=False),
    ):
        with pytest.raises(ValueError, match="nn.Linear.*transformers.Conv1D"):
            VPTQ(module=layer, qcfg=qcfg)


def test_vptq_transformers_conv1d_roundtrip():
    """`transformers.Conv1D` stores `(in, out)` and must transpose into the runtime."""

    torch.manual_seed(26)
    conv1d = transformers.Conv1D(16, 32)  # (nf, nx) = (out_features, in_features)
    conv1d.weight.data.copy_(torch.randn(32, 16) * 0.05)
    conv1d.bias.data.copy_(torch.randn(16) * 0.05)

    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 64],
        num_res_centroids=[-1, -1],
        kmeans_iters=8,
    )
    quantizer = VPTQ(module=conv1d, qcfg=qcfg)
    activations = torch.randn(2, 7, 32)
    quantizer.add_batch(activations)
    result, _duration, _loss = quantizer.quantize()

    module = build_module(qcfg, 32, 16, bias=True)
    module.pack(linear=conv1d, vptq_result=result)

    dequantized = module.dequantize()
    assert dequantized.shape == conv1d.weight.t().shape
    assert torch.equal(dequantized, result.quantized_weight)
    assert torch.equal(dequantized.float(), oracle_reconstruct(result, module))

    actual = module(activations)
    expected = conv1d(activations)
    assert actual.shape == expected.shape
    # a transposed or otherwise mis-shaped reconstruction would not stay close
    relative_error = (actual - expected).detach().norm() / expected.detach().norm()
    assert float(relative_error) < 0.5


def test_vptq_linear_applies_lora_adapter():
    """A loaded LoRA adapter must be added to the dequantized linear output."""

    torch.manual_seed(28)
    weight, hessian = synthetic_problem(out_features=16, in_features=32, seed=27)
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        kmeans_iters=4,
    )
    quantizer = build_quantizer(weight, qcfg)
    result, _loss = quantizer.quantize_weight(weight, hessian)

    lora = Lora(
        rank=1,
        # merged-weight layout: (in_features, rank) x (rank, out_features)
        lora_A=torch.randn(32, 1) * 0.1,
        lora_B=torch.randn(1, 16) * 0.1,
    )
    module = build_module(qcfg, 32, 16, adapter=lora)
    module.pack(linear=quantizer.layer, vptq_result=result)

    for activations in (torch.randn(3, 32), torch.randn(2, 5, 32)):
        base = torch.nn.functional.linear(activations, module.dequantize())
        delta = (activations @ module.adapter.lora_A) @ module.adapter.lora_B
        actual = module(activations)
        assert not torch.allclose(actual, base)
        assert torch.allclose(actual, base + delta, atol=1e-6)


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
def test_vptq_processor_participates_in_calibration_coverage_policy():
    """Never-invoked VPTQ modules must be pruned or reported, never quantized blind."""

    from gptqmodel.looper.stage_subset import _resolve_subset_calibration_coverage_policy
    from gptqmodel.looper.vptq_processor import VPTQProcessor

    policy = _resolve_subset_calibration_coverage_policy(object.__new__(VPTQProcessor), None)
    assert policy.validate_input_coverage is True
    assert policy.prune_uncovered_modules is True
    assert policy.record_dynamic_exclusions is True


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


def _build_tiny_gpt2_fixture(model_dir: Path):
    """Save a tiny GPT-2 checkpoint whose projections are `transformers.Conv1D`."""

    config = GPT2Config(
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_positions=32,
        n_ctx=32,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    GPT2LMHeadModel(config).save_pretrained(model_dir)
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


@pytest.mark.cpu
@pytest.mark.slow
def test_tiny_gpt2_vptq_conv1d_quantize_save_reload(tmp_path: Path):
    """Processor replay and feature inference must preserve Conv1D orientation."""

    native_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    native_dir.mkdir()

    _, tokenizer = _build_tiny_gpt2_fixture(native_dir)
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

    expected_modules = {
        "model.transformer.h.0.attn.c_attn": (32, 96),
        "model.transformer.h.0.attn.c_proj": (32, 32),
        "model.transformer.h.0.mlp.c_fc": (32, 128),
        "model.transformer.h.0.mlp.c_proj": (128, 32),
    }
    for name, (in_features, out_features) in expected_modules.items():
        module = dict(model.named_modules())[name]
        assert isinstance(module, VPTQLinear), name
        assert (module.in_features, module.out_features) == (in_features, out_features)
        assert module.dequantize().shape == (out_features, in_features)
        activations = torch.randn(2, 3, in_features)
        assert module(activations).shape == (2, 3, out_features)

    prompt = "tiny vptq calibration sample"
    inputs = tokenizer(prompt, return_tensors="pt")
    model.save(quantized_dir)

    reloaded = GPTQModel.load(str(quantized_dir), device="cpu", dtype=torch.bfloat16)
    reloaded_modules = dict(reloaded.named_modules())
    for name, (in_features, out_features) in expected_modules.items():
        restored = reloaded_modules[name]
        assert isinstance(restored, VPTQLinear), name
        assert (restored.in_features, restored.out_features) == (in_features, out_features)
        assert restored.dequantize().shape == (out_features, in_features)
        assert torch.equal(restored.dequantize(), dict(model.named_modules())[name].dequantize())

    with torch.no_grad():
        logits = reloaded.model(**inputs).logits
        tokens = reloaded.model.generate(**inputs, max_new_tokens=2, do_sample=False)
    assert torch.isfinite(logits).all()
    assert tokens.shape[-1] == inputs["input_ids"].shape[-1] + 2


@pytest.mark.cpu
@pytest.mark.slow
def test_tiny_qwen3_vptq_grouped_checkpoint_roundtrip(tmp_path: Path):
    """A grouped layout with a partial last group must survive save and reload."""

    native_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    native_dir.mkdir()
    _, tokenizer = _build_tiny_qwen3_fixture(native_dir)
    calibration = _build_calibration_dataset(tokenizer)

    # 64 hidden and 128 intermediate channels over groups of 12 leave 8 and 4
    # padding channels in the last group
    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, 8],
        vector_lens=[-1, 8],
        group_size=12,
        kmeans_iters=8,
        device="cpu",
    )
    model = GPTQModel.load(str(native_dir), quantize_config=qcfg, backend=BACKEND.VPTQ_TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.VPTQ_TORCH, calibration_data_min_length=1)

    grouped = {
        name: module for name, module in model.named_modules() if isinstance(module, VPTQLinear)
    }
    assert grouped
    for module in grouped.values():
        assert module.group_size == 12
        assert module.group_num * module.group_size >= module.in_features
        assert module.dequantize().shape == (module.out_features, module.in_features)

    model.save(quantized_dir)
    reloaded = GPTQModel.load(str(quantized_dir), device="cpu", dtype=torch.bfloat16)
    assert reloaded.quantize_config.group_size == 12

    reloaded_modules = dict(reloaded.named_modules())
    for name, module in grouped.items():
        restored = reloaded_modules[name]
        assert isinstance(restored, VPTQLinear)
        assert restored.group_size == module.group_size
        assert restored.group_num == module.group_num
        assert torch.equal(
            restored.dequantize().float(), module.dequantize().float()
        ), name

    inputs = tokenizer("tiny vptq calibration sample", return_tensors="pt")
    with torch.no_grad():
        logits = reloaded.model(**inputs).logits
    assert torch.isfinite(logits).all()


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
