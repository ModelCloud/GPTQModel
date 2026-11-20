# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exercise native EXL3 launches in a child so a runtime abort is a test failure."""

import struct
import subprocess
import sys
from itertools import combinations
from math import sqrt
from pathlib import Path
from random import Random

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.parametrize("bits", [2.5, 3.0, 3.5])
def test_q4nx_beam_reproduces_a_legal_tile_and_packs_its_bytes(bits):
    from tests.accelerator_gates import requires_nvidia_cuda

    if not requires_nvidia_cuda():
        pytest.skip("requires NVIDIA CUDA")

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import pack_trellis, quantize_tiles

    kv = int(4 * bits)
    rng = Random(671)
    suffixes = [rng.randrange(1 << kv) for _ in range(64)]
    stream = "".join(f"{suffix:0{kv}b}" for suffix in suffixes)
    words_in_stream_order = [int(stream[i:i + 16], 2) for i in range(0, len(stream), 16)]
    packed_oracle = torch.tensor(words_in_stream_order).reshape(-1, 2).flip(-1).flatten().to(torch.int16)

    state_bits = 32 - kv
    prefix_stream = "0" * state_bits + stream
    values = []
    for step in range(1, 65):
        end = state_bits + step * kv
        state = int(prefix_stream[end - state_bits:end], 2)
        product = (state * 0x83DCD12D) % (1 << 32)
        for byte in product.to_bytes(4, "little"):
            values.append(((byte // 16) + (byte % 16) - 15) / sqrt(42.5))
    source = torch.tensor([values], dtype=torch.float32, device="cuda")

    for scale_search in (False, True):
        args = {"K": bits, "q4nx": True, "q4nx_scale_search": scale_search}
        reconstructed, codes = quantize_tiles(source, args)
        torch.testing.assert_close(reconstructed, source, atol=2e-7, rtol=1e-7)
        assert codes[0, :64].tolist() == suffixes
        torch.testing.assert_close(
            pack_trellis(codes.view(1, 1, 256), args).flatten().cpu(), packed_oracle, rtol=0, atol=0,
        )


@pytest.mark.cuda
def test_q4nx_w35_matches_exhaustive_viterbi():
    from tests.accelerator_gates import requires_nvidia_cuda

    if not requires_nvidia_cuda():
        pytest.skip("requires NVIDIA CUDA")

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles

    # The 18-bit state overlaps by four bits when a step appends its 14-bit
    # suffix. An independent FP64 recurrence enumerates all 2**18 states at
    # every step and retains the true minimum for each successor class.
    device = torch.device("cuda", torch.cuda.current_device())
    source = torch.randn((1, 256), generator=torch.Generator().manual_seed(516)).to(device)
    state = torch.arange(1 << 18, device=device)
    product = (state[:, None] * 0x83DCD12D) >> torch.arange(0, 32, 8, device=device)
    values = (((product >> 4) & 15) + (product & 15) - 15).double() / sqrt(42.5)
    prior = torch.full((16,), torch.inf, dtype=torch.float64, device=device)
    prior[0] = 0
    history = []
    successors = torch.arange(16, device=device)
    for target in source.reshape(64, 4).double():
        costs = (values - target).square().sum(-1) + prior[state >> 14]
        prior, best = costs.view(-1, 16).min(dim=0)
        history.append(best * 16 + successors)
    current = int(prior.argmin())
    suffixes = []
    for line in reversed(torch.stack(history).cpu()):
        state = int(line[current])
        suffixes.append(state & ((1 << 14) - 1))
        current = state >> 14

    recovered, codes = quantize_tiles(source, {"K": 3.5, "q4nx": True})
    assert codes[0, :64].tolist() == suffixes[::-1]
    torch.testing.assert_close((recovered.double() - source).square().sum(), prior.min(), atol=1e-6, rtol=1e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("bits", [1.5, 2.5, 3.5])
def test_native_mul1_half_rate_packer_matches_alternating_ring(bits):
    if (
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(0)[0] < 8
    ):
        pytest.skip("requires NVIDIA Ampere or newer")

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import clear_tile_workspace, pack_trellis, quantize_tiles
    from gptqmodel.utils.kernel import KernelConfig, configure_kernel, get_kernel_config

    source = torch.linspace(-3, 3, 256, dtype=torch.float32, device="cuda").roll(7).reshape(1, 256)
    previous = get_kernel_config()
    configure_kernel(KernelConfig(exl3_tile_batch=1))
    try:
        args = {"K": bits, "mul1": True}
        quantized, encoded = quantize_tiles(source, args)
        packed = pack_trellis(encoded.view(1, 1, 256), args).cpu()
        encoded = encoded.cpu().long().flatten() & 0xFFFF
        quantized = quantized.cpu().flatten()
    finally:
        clear_tile_workspace()
        configure_kernel(previous)

    widths = [int(bits) + index % 2 for index in range(256)]
    ring = "".join(f"{int(state) & ((1 << width) - 1):0{width}b}" for state, width in zip(encoded, widths))
    expected_words = torch.tensor([int(ring[start:start + 16], 2) for start in range(0, len(ring), 16)])
    expected_words = expected_words.view(-1, 2).flip(-1).reshape(1, 1, int(16 * bits)).to(torch.int16)
    assert torch.equal(packed, expected_words)
    tail_biting_ring = ring[-16:] + ring
    end = 0
    for index, width in enumerate(widths):
        end += width
        assert encoded[index] == int(tail_biting_ring[end:end + 16], 2)
    assert encoded.unique().numel() > 20
    assert torch.equal(quantized, _codebook("mul1")[encoded])


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize(("bits", "codebook"), [(2.5, "mul1"), (3.5, "mul1"), (2.5, "q4nx"), (3, "q4nx"), (3.5, "q4nx")])
def test_exl3_tiny_llama_quantize_and_reload(tmp_path, bits, codebook):
    if (
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(0)[0] < 8
    ):
        pytest.skip("requires NVIDIA Ampere or newer")

    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
    from gptqmodel.quantization.config import EXL3Config
    from gptqmodel.utils.kernel import KernelConfig
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    source = tmp_path / "source"
    quantized = tmp_path / "quantized"
    config = LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
        bos_token_id=0, eos_token_id=1, pad_token_id=3,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        LlamaForCausalLM(config).eval().half().save_pretrained(source)
    build_wordlevel_tokenizer(128).save_pretrained(source)
    projection = "model.layers.0.self_attn.q_proj"
    control = "model.layers.0.self_attn.k_proj"
    mixed = codebook == "q4nx"
    include = [f"^{projection}$", f"^{control}$"] if mixed else [f"^{projection}$"]
    cfg = EXL3Config(
        bits=3 if mixed else bits,
        codebook="mcg" if mixed else codebook,
        dynamic={projection: {"bits": bits, "codebook": codebook}} if mixed else None,
        offload_to_disk=False,
        module_include=include,
        kernel=KernelConfig(exl3_tile_batch=4),
    )
    model = GPTQModel.load(str(source), quantize_config=cfg, device="cuda:0", dtype=torch.float16)
    tokens = torch.randint(4, 128, (8, 32), generator=torch.Generator().manual_seed(119)).tolist()
    dataset = [{"input_ids": row, "attention_mask": [1] * len(row)} for row in tokens]
    model.quantize(CalibrationConfig(dataset=dataset, batch_size=2))
    packed = model.model.get_submodule(projection)
    assert isinstance(packed, ExllamaV3Linear)
    assert packed.kernel.exl3_tile_batch == 4
    assert packed.trellis.dtype == torch.int16 and packed.trellis.shape == (8, 8, int(16 * bits))
    assert packed.tensor_storage_entry()["bits_per_weight"] == bits
    assert getattr(packed, codebook).dtype == torch.int32
    x = torch.linspace(-0.5, 0.5, 128, device="cuda", dtype=torch.float16).unsqueeze(0)
    packed.to(x.device)
    with torch.inference_mode():
        expected = packed(x)
    model.save(str(quantized))
    tensors = load_file(quantized / "model.safetensors")
    for name, value in packed.state_dict().items():
        assert torch.equal(tensors[f"{projection}.{name}"], value.cpu())

    loaded_policy = KernelConfig(exl3_tile_batch=2)
    restored = GPTQModel.load(
        str(quantized), device="cuda:0", backend=BACKEND.EXL3_EXLLAMA_V3, dtype=torch.float16,
        kernel=loaded_policy,
    )
    reloaded = restored.model.get_submodule(projection)
    assert isinstance(reloaded, ExllamaV3Linear)
    assert reloaded.kernel.exl3_tile_batch == 2
    assert packed.kernel.exl3_tile_batch == 4
    assert restored.quantize_config.runtime_bits == cfg.runtime_bits
    assert restored.quantize_config.codebook == cfg.codebook
    assert reloaded.tensor_storage_entry()["bits_per_weight"] == bits
    for name, value in reloaded.state_dict().items():
        assert torch.equal(tensors[f"{projection}.{name}"], value.cpu())
    with torch.inference_mode():
        assert torch.equal(reloaded(x), expected)
    if mixed:
        assert restored.quantize_config.dynamic_get(projection, "codebook") == "q4nx"
        assert restored.model.get_submodule(control).mcg.dtype == torch.int32
        assert reloaded.mcg is None
        portable = GPTQModel.load(
            str(quantized), device="cpu", backend=BACKEND.EXL3_TORCH, dtype=torch.float16,
        )
        assert portable.model.get_submodule(projection).q4nx.item() == 0x83DCD12D - (1 << 32)
        with torch.inference_mode():
            probe = torch.tensor([tokens[0][:8]])
            cpu_logits = portable.model(probe, use_cache=False).logits
            cuda_logits = restored.model(probe.cuda(), use_cache=False).logits.cpu()
        torch.testing.assert_close(cuda_logits, cpu_logits, rtol=2e-3, atol=2e-3)


def _codebook(codebook):
    states = torch.arange(65536, dtype=torch.int64)
    if codebook == "mul1":
        raw = (states * 0x83DCD12D) & 0xFFFFFFFF
        byte_sum = sum((raw >> shift) & 255 for shift in (0, 8, 16, 24))
        accumulator = (byte_sum + 0x6400).to(torch.int16).view(torch.float16).double()
        scale = struct.unpack("<e", struct.pack("<H", 0x1EEE))[0]
        bias = struct.unpack("<e", struct.pack("<H", 0xC931))[0]
        return (accumulator * scale + bias).half().float()
    raw = states * (89226354 if codebook == "3inst" else 0xCBAC1FED)
    if codebook == "3inst":
        raw += 64248484
    raw = (raw & 0x8FFF8FFF) ^ 0x3B603B60
    halves = torch.stack((raw & 65535, (raw >> 16) & 65535), dim=-1)
    return halves.to(torch.int16).view(torch.float16).double().sum(-1).half().float()


def _torch_oracle(source, bits, codebook):
    """Independent arithmetic for the native FP32 search with FP16 cost storage.

    Preserve native tail-state selection from the penultimate cost buffer and
    the strict-less-than shuffle tree's tie order; neither is changed by the
    compact LDS layout. FP64 multiply/add followed by FP32 models its FMA.
    """
    weights = source.cpu().half().float()
    count = len(weights)
    edges = 65536 >> bits
    states = torch.arange(65536).reshape(1 << bits, edges)
    incoming = states >> bits
    lut = _codebook(codebook)
    decoded = lut[states]
    history = torch.empty((count, 256, edges), dtype=torch.int16)

    def forward(roll, fixed=None):
        costs = None
        previous = None
        for step in range(256):
            ri = (step + roll) % 256
            delta = (decoded[None] - weights[:, ri, None, None]).double()
            candidates = delta * delta
            if costs is not None:
                candidates += costs[:, incoming].double()
            elif fixed is not None:
                candidates.masked_fill_(incoming[None] != fixed[:, None, None], torch.inf)
            best, symbols = candidates.float().min(dim=1)
            history[:, ri] = incoming.gather(0, symbols).to(torch.int16)
            previous, costs = costs, best.half()
        return previous

    previous = forward(128)
    # Each thread first scans its strided edges, then two descending-offset
    # shuffle trees reduce lanes and warps. Ties retain the left operand.
    local_costs, groups = previous.reshape(count, -1, 1024).min(dim=1)
    order = torch.tensor([int(f"{i:010b}"[::-1], 2) for i in range(1024)])
    thread = order[local_costs[:, order].argmin(dim=1)]
    edge = groups.gather(1, thread[:, None]).squeeze(1) * 1024 + thread
    for step in range(255, -1, -1):
        ri = (step + 128) % 256
        edge = history[:, ri].gather(1, edge[:, None]).squeeze(1).long()
        if ri == 0:
            break
    initial = edge
    forward(0, initial)
    indices = torch.empty_like(weights, dtype=torch.int64)
    for ri in range(255, -1, -1):
        previous = history[:, ri].gather(1, edge[:, None]).squeeze(1).long()
        indices[:, ri] = (previous << bits) | edge
        edge = previous
    return lut[indices], indices.to(torch.int16)


def _native_checks(codebook):
    from gptqmodel.exllamav3.ext import exllamav3_ext

    torch.set_num_threads(2)
    props = torch.cuda.get_device_properties(0)
    limit = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    lut = _codebook(codebook)
    # FP16 conversion midpoint and its FP32 neighbors, signed zero, codebook
    # endpoints, and an exact codebook-distance tie with its FP16 neighbors.
    midpoint = torch.tensor(1.0 + 2**-11)
    candidates = lut.reshape(4, -1)
    for left, right in combinations(range(4), 2):
        mid = (candidates[left] + candidates[right]) / 2
        distance = (candidates - mid).abs()
        tied = (
            (mid.half().float() == mid)
            & (candidates[left] != candidates[right])
            & (distance[left] == distance.min(dim=0).values)
        )
        if tied.any():
            tie = mid[tied][0].half()
            break
    else:
        raise AssertionError(f"no representable codebook tie for {codebook}")
    values = torch.stack(
        [
            torch.nextafter(midpoint, torch.tensor(-torch.inf)),
            midpoint,
            torch.nextafter(midpoint, torch.tensor(torch.inf)),
            torch.nextafter(tie, torch.tensor(-torch.inf, dtype=torch.float16)).float(),
            tie.float(),
            torch.nextafter(tie, torch.tensor(torch.inf, dtype=torch.float16)).float(),
            torch.tensor(-0.0),
            torch.tensor(0.0),
            lut.min(),
            lut.max(),
        ]
    )
    source = torch.randn((2, 256), generator=torch.Generator().manual_seed(3528))
    source[1] = values.repeat(26)[:256]
    source = source.cuda()

    def launch(bits, slots):
        edges = 65536 >> bits
        output = torch.full_like(source, 123.0)
        indices = torch.full_like(source, -123, dtype=torch.int16)
        costs = torch.empty((slots, 2, edges), device="cuda", dtype=torch.float16)
        history = torch.empty((slots, 256, edges), device="cuda", dtype=torch.int16)
        required = 4 * edges + 704
        if bits == 2 and 65536 <= limit < required:
            required = 65536
        if required > limit:
            with pytest.raises(RuntimeError, match=f"needs {required} B.*K={bits}.*{limit} B"):
                exllamav3_ext.quantize_tiles(
                    source,
                    output,
                    indices,
                    costs,
                    history,
                    bits,
                    codebook == "mcg",
                    codebook == "mul1",
                )
            torch.cuda.synchronize()
            assert torch.all(output == 123)
            assert torch.all(indices == -123)
            return None
        exllamav3_ext.quantize_tiles(
            source,
            output,
            indices,
            costs,
            history,
            bits,
            codebook == "mcg",
            codebook == "mul1",
        )
        torch.cuda.synchronize()
        return output.cpu(), indices.cpu()

    # Exercise rejection on RDNA4/Ada before a supported call in the same process.
    launch(1, 1)
    for bits in (2, 3):
        expected = _torch_oracle(source, bits, codebook)
        for slots in (1, 2, 1):
            actual = launch(bits, slots)
            if actual is not None:
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
def test_native_quantize_tiles_memory_layout(codebook):
    if not torch.cuda.is_available():
        pytest.skip("EXL3 native quantization requires CUDA/HIP")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), codebook],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _native_checks(sys.argv[1])
