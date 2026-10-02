# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.models.w4a_nvfp4_weight_qad import _project_master_codes


def test_cuda_memory_usage_keeps_live_and_peak_counters_separate(monkeypatch):
    from tests.models.w4a_nvfp4_weight_qad import _cuda_memory_usage

    for name, value in (("memory_allocated", 11), ("memory_reserved", 22),
                        ("max_memory_allocated", 33), ("max_memory_reserved", 44)):
        monkeypatch.setattr(torch.cuda, name, lambda value=value: value)
    assert _cuda_memory_usage() == {
        "allocated_bytes": 11, "reserved_bytes": 22,
        "peak_allocated_bytes": 33, "peak_reserved_bytes": 44,
    }


def test_token_accounting_reports_actual_variable_lengths():
    from tests.models.w4a_nvfp4_weight_qad import _token_length_summary

    samples = [torch.arange(319), torch.arange(1024), torch.arange(796)]
    assert _token_length_summary(samples) == {"rows": 3, "tokens": 2139,
                                              "minimum": 319, "maximum": 1024}
    for invalid in [[], [torch.empty(0, dtype=torch.long)]]:
        with pytest.raises(ValueError, match="nonempty samples"):
            _token_length_summary(invalid)


def test_gradient_coverage_does_not_accept_stale_or_other_lane_gradients():
    from tests.models.w4a_nvfp4_weight_qad import WeightGradientCoverage

    a = torch.nn.Parameter(torch.ones(3))
    b = torch.nn.Parameter(torch.ones(3))
    coverage = WeightGradientCoverage({"a": a, "b": b})
    try:
        with coverage.lane("a4"):
            (a.sum() + b.sum()).backward()
        assert coverage.counts == {"a4": 1, "a16": 0}
        # b.grad exists from A4, but A16 did not traverse b's graph.
        with pytest.raises(RuntimeError, match="Missing a16.*b"):
            with coverage.lane("a16"):
                (a.sum() + b.detach().sum()).backward()
        # An actual zero gradient is valid and distinct from a detached branch.
        with coverage.lane("a16"):
            (0 * a.sum() + 0 * b.sum()).backward()
        assert coverage.counts == {"a4": 1, "a16": 1}
        with pytest.raises(RuntimeError, match="one A4 or A16"):
            with coverage.lane("invalid"):
                pass
        assert coverage.active is None and not coverage.seen
    finally:
        coverage.close()
    assert not coverage.handles


def test_changed_codes_by_module_reports_only_serialized_code_changes():
    from tests.models.w4a_nvfp4_weight_qad import (
        _changed_code_count, _changed_codes_by_module, _install_trainable_gptq_codes,
    )

    core = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.Linear(128, 128))
    codes = {"0": torch.zeros((256, 128), dtype=torch.int8),
             "1": torch.zeros((128, 128), dtype=torch.int8)}
    scales = {"0": torch.full((2, 128), .125), "1": torch.full((1, 128), .25)}
    _, modules = _install_trainable_gptq_codes(core, codes, scales)
    assert _changed_codes_by_module(modules) == {"0": 0, "1": 0}
    with torch.no_grad():
        # Distinct K groups/scales and sub-threshold movement exercise the
        # training layout independently of the checkpoint's [K, N] layout.
        core[0]._w4a_qad_latent_weight[3, 0, 4] = .125 * .51
        core[0]._w4a_qad_latent_weight[7, 1, 8] = .125 * -.51
        core[1]._w4a_qad_latent_weight[2, 0, 5] = .25 * .49
    assert _changed_codes_by_module(modules) == {"0": 2, "1": 0}
    assert _changed_code_count(modules) == 2
    assert _changed_codes_by_module({}) == {}


def test_diagnostic_candidate_is_selected_only_by_improving_finite_a4_loss():
    from tests.models.w4a_nvfp4_weight_qad import DiagnosticCandidate

    tracker = DiagnosticCandidate(.2)
    calls = []
    def snapshot():
        calls.append(1)
        return {"codes": torch.tensor([len(calls)], dtype=torch.int8)}

    assert not tracker.consider(1, {"a4": .2, "a16": 0., "accepted": False}, snapshot)
    assert tracker.codes is None and not calls
    # A diagnostic may violate the proxy gate. The accepted flag stays false.
    row = {"a4": .19, "a16": .07, "accepted": False}
    assert tracker.consider(2, row, snapshot)
    assert tracker.step == 2 and tracker.validation["accepted"] is False
    row["a4"] = 99
    assert tracker.validation["a4"] == .19
    assert not tracker.consider(3, {"a4": .195, "a16": 0.}, snapshot)
    assert len(calls) == 1
    assert tracker.consider(4, {"a4": .18, "a16": .02}, snapshot)
    assert tracker.step == 4 and tracker.codes['codes'].item() == 2
    for bad in ({"a4": float('nan'), "a16": 0.}, {"a4": .17, "a16": float('inf')}):
        with pytest.raises(ValueError, match="finite losses"):
            tracker.consider(5, bad, snapshot)
    assert tracker.step == 4 and len(calls) == 2
    with pytest.raises(ValueError, match="finite losses"):
        DiagnosticCandidate(float('nan'))


def test_diagnostic_candidate_restores_previous_state_when_snapshot_fails():
    from tests.models.w4a_nvfp4_weight_qad import DiagnosticCandidate

    tracker = DiagnosticCandidate(.2)
    tracker.consider(1, {"a4": .19, "a16": .1}, lambda: {'original': True})
    def fail():
        raise RuntimeError('snapshot failure')
    with pytest.raises(RuntimeError, match='snapshot failure'):
        tracker.consider(2, {"a4": .18, "a16": .1}, fail)
    assert tracker.loss == .19 and tracker.step == 1 and tracker.codes == {'original': True}


def test_strict_and_diagnostic_native_exports_are_independent(tmp_path):
    from safetensors.torch import load_file, save_file
    from tests.models.w4a_nvfp4_weight_qad import _export_code_candidate

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'quantize_config.json').write_text('{"bits":4,"pack_dtype":"int32"}')
    state = {
        'linear.qweight': torch.full((16, 8), -2004318072, dtype=torch.int32),  # 0x88888888
        'linear.qzeros': torch.full((1, 1), 0x77777777, dtype=torch.int32),
        'linear.g_idx': torch.zeros(128, dtype=torch.int32),
        'linear.scales': torch.linspace(.01, .1, 8).half()[None],
        'model.embed_tokens.weight': torch.arange(128).bfloat16(),
    }
    save_file(state, str(source / 'model.safetensors'))
    original_bytes = (source / 'model.safetensors').read_bytes()
    base = torch.zeros((128, 8), dtype=torch.int8)
    strict, diagnostic = tmp_path / 'strict', tmp_path / 'diagnostic'
    count, immutable = _export_code_candidate(source, strict, None, {'linear': base})
    assert count == 0
    strict_bytes = (strict / 'model.safetensors').read_bytes()
    candidate = base.clone()
    candidate[7, 3], candidate[8, 4] = -8, 7
    count, diagnostic_immutable = _export_code_candidate(source, diagnostic, None, {'linear': candidate})
    assert count == 2 and diagnostic_immutable == immutable
    restored = load_file(str(diagnostic / 'model.safetensors'))
    expected = torch.empty((16, 8), dtype=torch.int32)
    for row in range(16):
        for column in range(8):
            value = sum((int(candidate[row * 8 + nibble, column]) + 8) << (4 * nibble)
                        for nibble in range(8))
            expected[row, column] = value - 2**32 if value >= 2**31 else value
    assert torch.equal(restored['linear.qweight'], expected)
    for key in state:
        if key != 'linear.qweight':
            assert restored[key].dtype == state[key].dtype and torch.equal(restored[key], state[key])
    assert (source / 'model.safetensors').read_bytes() == original_bytes
    assert (strict / 'model.safetensors').read_bytes() == strict_bytes
    assert not (strict / 'model.safetensors').samefile(diagnostic / 'model.safetensors')
    with pytest.raises(FileExistsError):
        _export_code_candidate(source, strict, None, {'linear': candidate})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequantize_replay_preserves_rope_and_other_non_linear_buffers(dtype):
    from types import SimpleNamespace
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from tests.models.w4a_nvfp4_norm_qat import _dequantize_for_replay

    core = torch.nn.Module()
    core.config = SimpleNamespace(quantization_config={})
    core.proj = TorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                            in_features=128, out_features=128, bias=False)
    core.proj.qweight.fill_(0x77777777)
    core.proj.qzeros.fill_(0x77777777)
    core.proj.scales.fill_(.01)
    core.rotary_emb = torch.nn.Module()
    frequencies = torch.tensor([1., .1234567, .01023456, .00012345], dtype=torch.float32)
    core.rotary_emb.register_buffer("inv_freq", frequencies.clone())
    core.register_buffer("extra_fp64", torch.tensor([1.123456789], dtype=torch.float64))
    original_frequency = core.rotary_emb.inv_freq
    original_extra = core.extra_fp64.clone()
    expected_weight = core.proj.dequantize_weight().T.half().to(dtype)
    restored = _dequantize_for_replay(core, device="cpu", dtype=dtype)
    assert restored is core and isinstance(core.proj, torch.nn.Linear)
    assert core.proj.weight.dtype == dtype
    torch.testing.assert_close(core.proj.weight, expected_weight, rtol=0, atol=0)
    assert core.rotary_emb.inv_freq is original_frequency
    assert core.rotary_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(core.rotary_emb.inv_freq, frequencies, rtol=0, atol=0)
    torch.testing.assert_close(core.extra_fp64, original_extra, rtol=0, atol=0)
    assert not torch.equal(frequencies, frequencies.to(dtype).float())


def test_master_projection_preserves_gptq_codes_and_direction():
    base = torch.tensor([-8, -3, 0, 4, 7], dtype=torch.int8)
    master = torch.tensor([-20.0, -2.7, 0.0, 4.8, 20.0])

    projected = _project_master_codes(master, base)

    assert torch.equal(projected.round().clamp(-8, 7).to(torch.int8), base)
    assert torch.all((projected - base.float()).abs() <= 0.25)
    assert torch.equal(torch.sign(projected - base), torch.sign(master - base))


def test_master_projection_is_monotone_within_a_code_cell():
    base = torch.zeros(7)
    master = torch.tensor([-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0])

    projected = _project_master_codes(master, base)

    assert torch.all(projected[1:] >= projected[:-1])
    assert projected[0] > -0.25
    assert projected[-1] < 0.25


@pytest.mark.parametrize("radius", [0.01, 0.25, 0.49])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_master_cell_radius_matches_independent_double_oracle(radius, device):
    import math

    # Every native code, both saturation ends, near-zero offsets, and large
    # master outliers. Expected values use Python double scalar arithmetic.
    offsets = [-100., -2., -.5, -.01, 0., .01, .5, 2., 100.]
    pairs = [(base, offset) for base in range(-8, 8) for offset in offsets]
    base = torch.tensor([b for b, _ in pairs], dtype=torch.int8, device=device)
    master = torch.tensor([b + d for b, d in pairs], dtype=torch.float32, device=device)
    projected = _project_master_codes(master, base, radius=radius)
    expected = torch.tensor([
        int(b) + radius * math.tanh((float(m) - int(b)) / radius)
        for m, b in zip(master.cpu(), base.cpu(), strict=True)
    ], dtype=torch.float64, device=device)
    torch.testing.assert_close(projected.double(), expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(projected.round().clamp(-8, 7).to(torch.int8), base)
    assert bool(((projected - base.float()).abs() < .5).all())


@pytest.mark.parametrize("radius", [True, 0., -.1, .5, float('inf'), float('nan')])
def test_master_cell_radius_rejects_invalid_values(radius):
    from tests.models.w4a_nvfp4_weight_qad import _install_trainable_gptq_codes

    with pytest.raises(ValueError, match="Master cell radius"):
        _project_master_codes(torch.zeros(1), torch.zeros(1), radius=radius)
    with pytest.raises(ValueError, match="Master cell radius"):
        _install_trainable_gptq_codes(torch.nn.Module(), {}, {}, master_cell_radius=radius)


@pytest.mark.parametrize("parameterization", ["physical_weight", "code_cell"])
@pytest.mark.parametrize("radius", [.25, .49])
def test_master_cell_radius_installer_keeps_original_snapshot(parameterization, radius):
    from tests.models.w4a_nvfp4_weight_qad import (
        _install_trainable_gptq_codes, _latent_codes, _snapshot_codes,
    )

    core = torch.nn.Sequential(torch.nn.Linear(128, 8, bias=False))
    codes = (torch.arange(1024).reshape(128, 8) % 16 - 8).to(torch.int8)
    scales = torch.tensor([[.0003, .001, .002, .004, .008, .016, .032, .125]]).half().float()
    offsets = torch.linspace(-2, 2, 1024).reshape(8, 1, 128)
    master = (codes.T.reshape(8, 1, 128).float() + offsets) * scales.T[..., None]
    _, modules = _install_trainable_gptq_codes(
        core, {"0": codes}, {"0": scales}, {"0": master.reshape(8, 128)},
        parameterization=parameterization, master_cell_radius=radius,
    )
    expected = codes.T.reshape(8, 1, 128).double() + radius * torch.tanh(offsets.double() / radius)
    torch.testing.assert_close(_latent_codes(core[0]).double(), expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(_snapshot_codes(modules)["0"], codes)


def test_larger_master_radius_allows_an_earlier_discrete_transition():
    base, master = torch.tensor([0], dtype=torch.int8), torch.tensor([2.])
    narrow = _project_master_codes(master, base, radius=.25)
    wide = _project_master_codes(master, base, radius=.49)
    assert torch.equal(narrow.round().to(torch.int8), base)
    assert torch.equal(wide.round().to(torch.int8), base)
    # This establishes only the intended initialization mechanism. It is not
    # evidence that wider cells improve an actual model's training or quality.
    assert (narrow + .02).round().item() == 0
    assert (wide + .02).round().item() == 1


def test_master_rotation_recovery_is_independent_of_rng():
    from tests.models.w4a_nvfp4_weight_qad import _recover_hadamard_rotation

    generator = torch.Generator().manual_seed(9007)
    width = 128
    indices = torch.arange(width, dtype=torch.int64)
    bit_parity = torch.zeros((width, width), dtype=torch.int64)
    for bit in range(7):
        bit_parity ^= ((indices[:, None] >> bit) & 1) * ((indices[None, :] >> bit) & 1)
    # Independent Walsh matrix construction, without the recovery butterfly.
    hadamard = (1 - 2 * bit_parity).double() / width**0.5
    signs = torch.randint(0, 2, (width,), generator=generator).double() * 2 - 1
    expected = signs[:, None] * hadamard
    embeddings = torch.randn((32, width), dtype=torch.float64, generator=generator)
    saved = embeddings @ expected
    torch.manual_seed(8172)
    recovered, report = _recover_hadamard_rotation(embeddings, saved)
    torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
    assert report['relative_embedding_error'] < 1e-12

    recovered_bf16, report = _recover_hadamard_rotation(embeddings.bfloat16(),
                                                      (embeddings.bfloat16().double() @ expected).bfloat16())
    torch.testing.assert_close(recovered_bf16, expected, rtol=0, atol=0)
    assert report['relative_embedding_error'] < 0.002


def test_master_rotation_recovery_rejects_unrelated_embeddings():
    import pytest
    from tests.models.w4a_nvfp4_weight_qad import _recover_hadamard_rotation

    generator = torch.Generator().manual_seed(9008)
    a = torch.randn((32, 128), generator=generator)
    b = torch.randn((32, 128), generator=generator)
    with pytest.raises(ValueError, match='do not match'):
        _recover_hadamard_rotation(a, b)


def test_qad_replay_uses_the_exported_checkpoint_policy(tmp_path):
    import json
    from tests.models.w4a_nvfp4_norm_qat import _activation_replay_config

    for version, recipe in [(2, 'four_six'), (3, 'least_squares'), (4, 'least_squares_grid')]:
        config = {'rotation': 'hadamard', 'activation': {
            'version': version, 'mode': 'w4a_nvfp4', 'recipe': recipe}}
        (tmp_path / 'quantize_config.json').write_text(json.dumps(config))
        policy = _activation_replay_config(tmp_path)
        assert policy.activation_version == version
        assert policy.activation_recipe == recipe
        assert policy.dynamic_get(layer_name='model.layers.0.self_attn.q_proj') is None


def test_qad_rejects_norm_training_that_breaks_v4_contract(tmp_path):
    import json
    import pytest
    from tests.models.w4a_nvfp4_norm_qat import _activation_replay_config

    (tmp_path / 'quantize_config.json').write_text(json.dumps({
        'rotation': 'hadamard', 'activation': {'version': 4, 'mode': 'w4a_nvfp4'}}))
    with pytest.raises(ValueError, match='unit fused norms'):
        _activation_replay_config(tmp_path, train_norms=True)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trainable", ["weights", "code_cells", "scales"])
def test_trainable_gptq_forward_matches_groupwise_fp32_oracle(dtype, trainable, device):
    from tests.models.w4a_nvfp4_weight_qad import _install_trainable_gptq_codes
    from tests.models.w4a_nvfp4_scale_qad import _install_trainable_gptq_scales

    generator = torch.Generator(device=device).manual_seed(9010)
    core = torch.nn.Sequential(torch.nn.Linear(256, 128, bias=True, dtype=dtype, device=device))
    core[0].bias.data.copy_(torch.linspace(-0.1, 0.1, 128, dtype=dtype, device=device))
    codes = torch.randint(-8, 8, (256, 128), generator=generator, dtype=torch.int8, device=device)
    scales = (0.01 + 0.19 * torch.rand((2, 128), generator=generator, device=device)).half().float()
    inputs = torch.randn((7, 256), dtype=dtype, generator=generator, device=device)
    bias = core[0].bias.detach().clone()
    installer = _install_trainable_gptq_scales if trainable == "scales" else _install_trainable_gptq_codes
    kwargs = {"parameterization": "code_cell"} if trainable == "code_cells" else {}
    parameters, _ = installer(core, {'0': codes}, {'0': scales}, **kwargs)
    actual = core(inputs)
    reference = torch.zeros((7, 128), dtype=torch.float64, device=device)
    for group in range(2):
        sl = slice(group * 128, (group + 1) * 128)
        reference += (inputs[:, sl].double() @ codes[sl].double()) * scales[group].double()
    reference += bias.double()
    if device == "cuda":
        torch.cuda.synchronize()
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference.to(dtype), rtol=2e-3, atol=2e-3)
    actual.float().square().mean().backward()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_code_cell_gradient_clamp_regularizer_and_snapshot_independent_oracle():
    from tests.models.w4a_nvfp4_weight_qad import (
        _clamp_latent_codes, _install_trainable_gptq_codes, _snapshot_codes, _weight_regularizer,
    )

    core = torch.nn.Sequential(torch.nn.Linear(256, 128, bias=False))
    codes = torch.zeros((256, 128), dtype=torch.int8)
    codes[0, 0], codes[128, 1] = -8, 7
    scales = torch.stack([torch.full((128,), 2.**-12), torch.full((128,), .25)])
    params, modules = _install_trainable_gptq_codes(
        core, {"0": codes}, {"0": scales}, parameterization="code_cell",
    )
    inputs = torch.linspace(-1, 1, 768).reshape(3, 256)
    core(inputs).sum().backward()
    expected_gradient = torch.stack([
        inputs[:, g*128:(g+1)*128].double().sum(0)[None, :] * scales[g].double()[:, None]
        for g in range(2)
    ], dim=1)
    torch.testing.assert_close(params[0].grad.double(), expected_gradient, rtol=1e-6, atol=1e-6)
    assert torch.equal(_snapshot_codes(modules)["0"], codes)
    assert _weight_regularizer(params, modules).item() == 0
    with torch.no_grad():
        params[0][0, 0, 0] = -20
        params[0][1, 1, 0] = 20
        params[0][2, 0, 3] = 10
        params[0][3, 1, 4] = -10
    _clamp_latent_codes(modules, 1)
    expected = codes.clone()
    expected[3, 2], expected[132, 3] = 1, -1
    assert torch.equal(_snapshot_codes(modules)["0"], expected)
    assert _weight_regularizer(params, modules).item() == 2 / codes.numel()
    with pytest.raises(ValueError, match="parameterization"):
        _install_trainable_gptq_codes(core, {}, {}, parameterization="invalid")


def test_teacher_disk_cache_is_lossless_and_preserves_hidden_order(tmp_path):
    from tests.models.w4a_nvfp4_weight_qad import DiskTeacherTargets

    store = DiskTeacherTargets(tmp_path / 'targets')
    target = {'logits': torch.arange(35).reshape(1, 5, 7).bfloat16(),
              'hidden_states': tuple(torch.full((1, 5, 3), float(i)).half() for i in range(12))}
    store.append(target)
    store.append({'logits': target['logits'] * 2, 'hidden_states': ()})
    assert len(store) == 2 and store.bytes_on_disk > 0
    restored = store[0]
    torch.testing.assert_close(restored['logits'], target['logits'], rtol=0, atol=0)
    for actual, expected in zip(restored['hidden_states'], target['hidden_states'], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target['logits'].zero_()
    assert torch.count_nonzero(store[0]['logits']) == 34
    assert store[1]['hidden_states'] == ()
    with pytest.raises(IndexError):
        store[2]
    with pytest.raises(FileExistsError):
        DiskTeacherTargets(tmp_path / 'targets')


def test_teacher_disk_cache_syncs_before_release_and_returns_owned_tensors(tmp_path, monkeypatch):
    import os
    from tests.models import w4a_nvfp4_weight_qad as qad

    if not hasattr(os, "posix_fadvise"):
        pytest.skip("POSIX file-cache advice required")
    events = []
    real_sync, real_advice = os.fsync, os.posix_fadvise

    def sync(fd):
        events.append("sync")
        real_sync(fd)

    def advise(fd, offset, length, advice):
        assert offset == length == 0 and advice == os.POSIX_FADV_DONTNEED
        events.append("release")
        real_advice(fd, offset, length, advice)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "posix_fadvise", advise)
    store = qad.DiskTeacherTargets(tmp_path / "targets")
    target = {"logits": torch.arange(12).reshape(1, 3, 4).bfloat16(), "hidden_states": ()}
    store.append(target)
    assert events == ["sync", "release"]
    original_file = store.paths[0].read_bytes()
    mapped = qad.load_file(str(store.paths[0]), device="cpu")
    monkeypatch.setattr(qad, "load_file", lambda *_args, **_kwargs: mapped)
    restored = store[0]
    assert events == ["sync", "release", "release"]
    assert restored["logits"].data_ptr() != mapped["logits"].data_ptr()
    torch.testing.assert_close(restored["logits"], target["logits"], rtol=0, atol=0)
    restored["logits"].zero_()
    torch.testing.assert_close(mapped["logits"], target["logits"], rtol=0, atol=0)
    assert store.paths[0].read_bytes() == original_file


def test_teacher_disk_cache_without_posix_advice_remains_lossless(tmp_path, monkeypatch):
    import os
    from tests.models.w4a_nvfp4_weight_qad import DiskTeacherTargets

    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    store = DiskTeacherTargets(tmp_path / "targets")
    target = {"logits": torch.arange(6).reshape(1, 2, 3).half(), "hidden_states": ()}
    store.append(target)
    torch.testing.assert_close(store[0]["logits"], target["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_nonreentrant_checkpointing_preserves_replay_outputs_and_latent_gradients(monkeypatch, dtype, rounded, device):
    from types import SimpleNamespace
    from transformers import LlamaConfig, LlamaForCausalLM
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from tests.models.w4a_nvfp4_weight_qad import _install_trainable_gptq_codes

    torch.manual_seed(9501)
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    core = LlamaForCausalLM(config).to(device=device, dtype=dtype).train()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    codes, scales = {}, {}
    for name, module in core.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            codes[name] = torch.randint(-8, 8, (module.in_features, module.out_features), dtype=torch.int8)
            scales[name] = torch.full((module.in_features // 128, module.out_features), .002).half().float()
    parameters, _ = _install_trainable_gptq_codes(core, codes, scales)
    qcfg = SimpleNamespace(activation_mode="w4a_nvfp4", activation_recipe="least_squares",
                           activation_version=4, activation_global_scales=None,
                           dynamic_get=lambda **_kwargs: None)
    replay.install_w4a_llama_replay(core, qcfg)
    original_round = replay.round_w4a_activation
    def straight_through(x, *args):
        if device == "cuda":
            from tests.models.w4a_nvfp4_weight_qad import straight_through_activation_round
            return straight_through_activation_round(x, *args)
        value = original_round(x.detach(), *args)
        return x + (value - x).detach()
    monkeypatch.setattr(replay, "_round", straight_through)
    replay.set_w4a_replay_enabled(core, rounded)
    ids = torch.arange(9, device=device)[None]
    reference = core(input_ids=ids, use_cache=False).logits
    reference.float().square().mean().backward()
    gradients = [parameter.grad.detach().clone() for parameter in parameters]
    for parameter in parameters:
        parameter.grad = None
    core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    actual = core(input_ids=ids, use_cache=False).logits
    actual.float().square().mean().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for parameter, expected in zip(parameters, gradients, strict=True):
        assert torch.isfinite(parameter.grad).all() and bool(parameter.grad.abs().sum() > 0)
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
