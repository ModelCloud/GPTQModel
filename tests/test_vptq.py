Warning: truncated output (original token count: 23134)
Total output lines: 2382

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
3. the Torch and CUDA inference implementations, checked for activation-dtype
   preservation and accuracy against independent Torch oracles

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
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from gptqmodel.nn_modules.qlinear.vptq_cuda import (
    VPTQCudaLinear,
    _vptq_cuda_fused_max_rows,
)
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
                group_we…13134 tokens truncated…fg, dtype=dtype)
    quantizer.layer.to(device)
    result, loss = quantizer.quantize_weight(
        weight.to(device=device, dtype=dtype), hessian.to(device), target_dtype=dtype
    )
    assert math.isfinite(loss)
    assert result.indices[1].device.type == "cuda"

    module = build_module(
        qcfg, weight.shape[1], weight.shape[0], dtype=dtype, linear_cls=VPTQCudaLinear
    ).to(device)
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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vector_len", [4, 8, 16])
@pytest.mark.parametrize("rows", [1, 2, 3, 8, 16, 17, 64, 127, 128, 129, 1024, 2048])
def test_vptq_cuda_linear_matches_independent_torch_oracle(dtype, rows, vector_len, monkeypatch):
    """Check fused and dense CUDA dispatch around the M crossover against an oracle."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for VPTQ fused inference coverage")

    device = torch.device("cuda:0")
    weight, hessian = synthetic_problem(out_features=64, in_features=128, seed=71)
    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, 8],
        vector_lens=[-1, vector_len],
        kmeans_iters=12,
        enable_norm=True,
        enable_perm=True,
    )
    quantizer = build_quantizer(weight, qcfg, dtype=dtype)
    result, _ = quantizer.quantize_weight(
        weight.to(device=device, dtype=dtype), hessian.to(device), target_dtype=dtype
    )
    module = build_module(
        qcfg, weight.shape[1], weight.shape[0], dtype=dtype, linear_cls=VPTQCudaLinear
    ).to(device)
    module.pack(linear=quantizer.layer, vptq_result=result)

    from gptqmodel.utils.vptq_cuda import (
        vptq_cuda_dequantize as real_vptq_cuda_dequantize,
        vptq_cuda_linear as real_vptq_cuda_linear,
    )

    fused_calls = []
    dequantized_weights = []

    def record_fused_call(*args, **kwargs):
        fused_calls.append(True)
        return real_vptq_cuda_linear(*args, **kwargs)

    def record_dequantize(*args, **kwargs):
        weight = real_vptq_cuda_dequantize(*args, **kwargs)
        dequantized_weights.append(weight)
        return weight

    monkeypatch.setattr("gptqmodel.utils.vptq_cuda.vptq_cuda_linear", record_fused_call)
    monkeypatch.setattr("gptqmodel.utils.vptq_cuda.vptq_cuda_dequantize", record_dequantize)

    generator = torch.Generator().manual_seed(73)
    activations = torch.randn(rows, 128, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    actual = module(activations)
    assert actual.dtype == dtype
    fused_expected = rows <= _vptq_cuda_fused_max_rows(module)
    assert bool(fused_calls) is fused_expected
    assert bool(dequantized_weights) is not fused_expected

    oracle_weight = oracle_reconstruct(result, module).to(device=device)
    oracle_raw = activations.to(torch.float64) @ oracle_weight.to(torch.float64).t()
    oracle_rounded = oracle_raw.to(dtype)
    if fused_expected:
        # The fused kernel accumulates directly from independently
        # reconstructed weights; compare visible output after output rounding.
        torch.testing.assert_close(actual.float(), oracle_rounded.float(), rtol=2e-3, atol=2e-3)
    else:
        # The dense path must preserve native Torch/cuBLAS GEMM behavior once
        # the independently reconstructed dense weights are supplied.
        expected = torch.nn.functional.linear(activations, oracle_weight.to(dtype=dtype))
        assert torch.equal(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 3])
def test_vptq_cuda_outlier_path_matches_independent_torch_oracle(dtype, rows):
    """Both native CUDA paths handle outliers, groups, residuals, permutation and norm."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for VPTQ CUDA outlier coverage")

    device = torch.device("cuda:0")
    weight, hessian = synthetic_problem(out_features=32, in_features=64, seed=83)
    qcfg = VPTQConfig(
        num_centroids=[8, 16],
        num_res_centroids=[-1, 4],
        vector_lens=[4, 4],
        npercent=12.5,
        group_num=2,
        kmeans_iters=8,
        enable_norm=True,
        enable_perm=True,
    )
    quantizer = build_quantizer(weight, qcfg, dtype=dtype)
    result, _ = quantizer.quantize_weight(
        weight.to(device=device, dtype=dtype), hessian.to(device), target_dtype=dtype
    )
    module = build_module(
        qcfg, weight.shape[1], weight.shape[0], dtype=dtype, linear_cls=VPTQCudaLinear
    ).to(device)
    module.pack(linear=quantizer.layer, vptq_result=result)

    activations = torch.randn(rows, weight.shape[1], device=device, dtype=dtype)
    actual = module(activations)
    oracle_weight = oracle_reconstruct(result, module).to(device=device)
    oracle = torch.nn.functional.linear(activations, oracle_weight.to(dtype=dtype))
    assert actual.dtype == dtype
    assert float((actual.float() - oracle.float()).abs().max()) <= 2e-3


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(0, 128), (2, 0, 128)])
def test_vptq_cuda_empty_rows_skip_kernel_and_preserve_adapter(shape, monkeypatch):
    """Empty CUDA inputs preserve Linear shape and adapter behavior without a launch."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for VPTQ empty-input coverage")

    class AddOneAdapter:
        def apply(self, *, x, out):
            return out + 1

    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, 8],
        vector_lens=[-1, 4],
        kmeans_iters=12,
        enable_norm=True,
        enable_perm=True,
    )
    module = build_module(
        qcfg, 128, 64, dtype=torch.float16, linear_cls=VPTQCudaLinear
    ).to("cuda")
    # VPTQ currently rejects adapters during normal backend selection, but its
    # forward path still preserves an attached adapter for shared linear APIs.
    module.adapter = AddOneAdapter()

    def fail_if_kernel_called(*args, **kwargs):
        raise AssertionError("empty input must not dispatch the CUDA kernel")

    monkeypatch.setattr("gptqmodel.utils.vptq_cuda.vptq_cuda_linear", fail_if_kernel_called)
    x = torch.empty(shape, device="cuda", dtype=torch.float16)
    actual = module(x)
    assert actual.shape == shape[:-1] + (64,)
    assert actual.dtype == x.dtype and actual.device == x.device
    assert torch.equal(actual, torch.ones_like(actual))


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
