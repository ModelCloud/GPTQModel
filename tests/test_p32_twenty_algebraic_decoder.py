"""Exact CPU state algebra checks; no GPU or model-quality claims."""

import json
import os
import random
from pathlib import Path

import pytest

from scripts.p32_twenty.algebraic_decoder import (
    MASK,
    Affine,
    affine_scan,
    bitsliced,
    compose,
    decode_planar_states,
    gf2_scan,
    hierarchical_all_start,
    super_symbol,
)


def serial(edges, width):
    # Independent chronological recurrence: one lap resolves tail biting.
    state = 0
    for edge in edges:
        state = ((state << width) | edge) & MASK
    initial = state
    result = []
    for edge in edges:
        state = ((state << width) | edge) & MASK
        result.append(state)
    assert state == initial
    return result


@pytest.mark.parametrize('width', range(2, 8))
@pytest.mark.parametrize('pattern', ['random', 'zero', 'ones', 'impulse', 'alternating'])
def test_candidates_against_serial(width, pattern):
    rng = random.Random(7)
    edge_mask = (1 << width) - 1
    edges = {
        'random': [rng.randrange(1 << width) for _ in range(128)],
        'zero': [0] * 128,
        'ones': [edge_mask] * 128,
        'impulse': [0] * 127 + [edge_mask],
        'alternating': [0, edge_mask] * 64,
    }[pattern]
    before = edges[:]
    reference = serial(edges, width)
    assert affine_scan(edges, width) == reference
    assert gf2_scan(edges, width) == reference
    assert bitsliced([edges], width) == [reference]
    for block in (1, 2, 6, 8, 16, 31, 32, 64, 127, 128):
        assert hierarchical_all_start(edges, width, block) == reference
    for steps in (2, 4, 6, 8):
        assert super_symbol(edges, width, steps) == reference
    assert before == edges


def test_compact_composition_all_65536_states():
    # Direction-sensitive, partial shifts and saturation, including arbitrary
    # XOR suffixes (the public affine primitive is more general than edges).
    for a, b in ((Affine(), Affine(3, 5)), (Affine(3, 7), Affine(5, 19)),
                 (Affine(7, 65001), Affine(13, 17000)), (Affine(16, MASK), Affine(2, 2))):
        combined = compose(a, b)
        for state in range(65536):
            assert combined(state) == b(a(state))
    rng = random.Random(21)
    for _ in range(100):
        a, b, c = [Affine(rng.randrange(17), rng.randrange(65536)) for _ in range(3)]
        assert compose(compose(a, b), c) == compose(a, compose(b, c))
        assert compose(a, Affine()) == compose(Affine(), a) == a


@pytest.mark.parametrize('count', [0, 1, 16, 31, 32, 33, 65])
@pytest.mark.parametrize('width', range(2, 8))
def test_bitsliced_stream_boundaries(count, width):
    rng = random.Random(25)
    streams = [[rng.randrange(1 << width) for _ in range(128)] for _ in range(count)]
    assert bitsliced(streams, width) == [serial(row, width) for row in streams]


@pytest.mark.parametrize('function', [affine_scan, hierarchical_all_start, super_symbol, gf2_scan])
def test_invalid_streams(function):
    for edges, width in (([0] * 127, 2), ([0] * 129, 2), ([0] * 128, 1), ([0] * 128, 8),
                         ([0] * 128, True), ([-1] * 128, 2), ([4] * 128, 2)):
        with pytest.raises(ValueError):
            function(edges, width)
    with pytest.raises(TypeError):
        function([0.5] * 128, 2)


def test_invalid_parameters():
    for shift, suffix in ((-1, 0), (17, 0), (0, -1), (0, 65536)):
        with pytest.raises(ValueError):
            Affine(shift, suffix)
    for block in (0, 129, True, 2.5):
        with pytest.raises(ValueError):
            hierarchical_all_start([0] * 128, 2, block)
    for steps in (0, 3, 9, True):
        with pytest.raises(ValueError):
            super_symbol([0] * 128, 2, steps)
    with pytest.raises(ValueError):
        bitsliced([], 8)
    with pytest.raises(ValueError):
        bitsliced([[0] * 127], 2)


@pytest.mark.parametrize('bits', [1, 1.5, 2, 2.5, 3, 3.5])
def test_planar_adapter_matches_canonical_and_direct_window(bits):
    import torch

    from gptqmodel.quantization.qvq import (
        repack_p32_planar_to_window,
        unpack_p32_window_states,
        unpack_trellis_states,
    )

    gen = torch.Generator().manual_seed(7)
    storage = torch.randint(-(1 << 31), 1 << 31, (2, 3, int(bits * 16)), dtype=torch.int32, generator=gen)
    words = storage[..., ::2]  # signed packed fields, noncontiguous and multidimensional
    before = words.clone()
    canonical = unpack_trellis_states(words, bits=bits)
    window = unpack_p32_window_states(repack_p32_planar_to_window(words, bits=bits), bits=bits)
    assert torch.equal(canonical, window)
    for experiment in (21, 22, 24, 25, 26):
        assert torch.equal(decode_planar_states(words, bits=bits, experiment=experiment), canonical)
        assert torch.equal(words, before)
    empty = torch.empty((0, int(bits * 8)), dtype=torch.int32)
    assert decode_planar_states(empty, bits=bits, experiment=21).shape == (0, 128)


def test_adapter_validation():
    import torch

    words = torch.zeros((2, 16), dtype=torch.int32)
    for payload, rate, experiment, kwargs in (
        (words, 4, 21, {}), (words.float(), 2, 21, {}), (words[:, :-1], 2, 21, {}),
        (words, 2, 23, {}), (words, 2, 25, {'steps': 4}),
        (torch.zeros((), dtype=torch.int32), 2, 21, {}),
        (torch.empty((2, 16), dtype=torch.int32, device='meta'), 2, 21, {}),
    ):
        with pytest.raises(ValueError):
            decode_planar_states(payload, bits=rate, experiment=experiment, **kwargs)
    for experiment, kwargs in ((22, {'block_steps': 6}), (24, {'steps': 6})):
        assert not decode_planar_states(words, bits=2, experiment=experiment, **kwargs).any()


@pytest.mark.skipif(not os.environ.get('P32_ALGEBRAIC_SNAPSHOT'), reason='opt-in read-only real snapshot check')
def test_real_snapshot_states_and_banked_values():
    """First/middle/last tiles of every P32 module; no calibration or inference."""
    import torch
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import (
        unpack_qvq_binary_bank_ids,
        unpack_trellis_states,
    )
    from gptqmodel.quantization.qvq_codecs import pgc16_decode_states_v2_banked

    root = Path(os.environ['P32_ALGEBRAIC_SNAPSHOT'])
    weight_map = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    prefixes = sorted(name[:-8] for name in weight_map
                      if name.endswith('.trellis') and name[:-8] + '.bank_alt_id' in weight_map)
    assert prefixes, 'snapshot contains no P32 modules'
    checked = 0
    for prefix in prefixes:
        def read(suffix, prefix=prefix):
            key = prefix + suffix
            with safe_open(str(root / weight_map[key]), framework='pt', device='cpu') as handle:
                return handle.get_tensor(key)

        # Read slices from mapped shards, avoiding full module payload copies.
        key = prefix + '.trellis'
        with safe_open(str(root / weight_map[key]), framework='pt', device='cpu') as handle:
            view = handle.get_slice(key)
            shape = view.get_shape()
            assert len(shape) == 2
            selected = sorted({0, shape[0] // 2, shape[0] - 1})
            words = torch.cat([view[i:i + 1] for i in selected], dim=0)
        bits = shape[-1] / 8
        banks = unpack_qvq_binary_bank_ids(read('.bank_ids'), shape[0] * 8).reshape(-1, 8)[selected]
        banks = banks.repeat_interleave(16, dim=1) * int(read('.bank_alt_id').item())
        canonical = unpack_trellis_states(words, bits=bits)
        values = pgc16_decode_states_v2_banked(canonical, banks, bits=bits)
        for experiment in (21, 22, 24, 25, 26):
            actual = decode_planar_states(words, bits=bits, experiment=experiment)
            assert torch.equal(actual, canonical), (prefix, experiment)
            assert torch.equal(pgc16_decode_states_v2_banked(actual, banks, bits=bits), values), (prefix, experiment)
        checked += len(selected)
    print(f'CPU snapshot check: {len(prefixes)} modules, {checked} sampled tiles, five state candidates')
