# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ scale search must not duplicate host copies, concatenations, or scales.

The AWQ hooks fire once per quantizable projection, but q/k/v and gate/up read the
same activation object. Copying it to the host once per module is redundant D2H
traffic and host RAM, and the resulting host tensors are then concatenated and
scaled once per module as well. The FP reference output must likewise stay on the
replay device, because the scale grid moves every chunk back there anyway.

These tests pin the sharing contract:

* one host copy per unique source tensor, and none across distinct sources;
* one ``torch.cat`` per identical capture list;
* one in-place scale division per tensor, even though several module names
  reference it, with an independent closed-form oracle for the expected value;
* the chunked FP reference stays on the device the replay produced it on;
* the activation magnitude is materialized per chunk, so the reference and every
  candidate replay run without an activation-sized ``abs()`` alive (#5765).
"""

import threading
import types
import weakref

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from gptqmodel import CalibrationConfig
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.config import AWQConfig


def _processor():
    """A processor with only the state the capture helpers touch."""

    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._mask_tls = threading.local()
    processor.tasks = {}
    processor._capture_host_cache = {}
    processor._capture_host_batch = None
    processor._module_forward_kwargs = {}
    processor._feature_task_names = set()
    processor._feature_stats = {}
    processor._awq_feature_kwargs = {}
    processor._scale_feature_by_module = {}
    processor._nsamples_total = 1
    processor.gptq_model = None
    processor.current_batch_index = lambda: 0
    return processor


def _replay_processor(max_chunk_memory, quant_batch_size=1):
    """A processor with only the state the replay-sizing helper touches."""

    processor = AWQProcessor.__new__(AWQProcessor)
    processor.max_chunk_memory = max_chunk_memory
    processor._quant_batch_size = quant_batch_size
    return processor


def test_scale_search_replay_batch_size_accounts_for_the_sequence_dimension():
    """A 3-D capture is [batches, seq, hidden], not [batches, hidden].

    Equal-length calibration windows are concatenated to ``[128, 2048, 4096]``,
    so one leading-dimension row costs ``2048 * 4096`` elements. Sizing a row as
    ``hidden`` alone undercounts by ``seq`` and returned ``0`` ("one forward over
    everything"), which is a 6.4 GiB chunk for Llama-3.1-8B.
    """

    budget = 1024 * 1024 * 1024
    x = torch.empty(128, 2048, 4096, dtype=torch.float16)
    batch_size = _replay_processor(budget)._scale_search_replay_batch_size(x)

    assert batch_size is not None and batch_size > 0
    per_row_bytes = (x.numel() // x.shape[0]) * x.element_size() * 3
    # Independent arithmetic: the budget bounds the replayed chunk.
    assert batch_size * per_row_bytes <= budget
    assert batch_size < x.shape[0]


def test_scale_search_replay_batch_size_keeps_the_2d_row_size():
    """A 2-D ``[rows, hidden]`` capture has one hidden vector per row."""

    budget = 1024 * 1024 * 1024
    x = torch.empty(262144, 4096, dtype=torch.float16)
    batch_size = _replay_processor(budget)._scale_search_replay_batch_size(x)
    per_row_bytes = (x.numel() // x.shape[0]) * x.element_size() * 3

    assert batch_size == budget // per_row_bytes
    assert batch_size * per_row_bytes <= budget


def test_scale_search_replay_batch_size_handles_empty_and_1d_inputs():
    processor = _replay_processor(1024 * 1024 * 1024)

    # No leading-dimension rows, and no elements at all: one forward over nothing
    # rather than a division by an empty row.
    assert processor._scale_search_replay_batch_size(torch.empty(0)) is None
    assert processor._scale_search_replay_batch_size(torch.empty(4, 0, 8)) is None
    # A 1-D capture has one element per leading-dimension row, so it always fits.
    assert processor._scale_search_replay_batch_size(torch.empty(1024, dtype=torch.float16)) == 0


def _record(processor, name, tensor):
    processor._record_input_feature(name, tensor)


def test_siblings_share_one_host_copy():
    processor = _processor()
    source = torch.randn(2, 6, 8)

    _record(processor, "self_attn.q_proj", source)
    _record(processor, "self_attn.k_proj", source)
    _record(processor, "self_attn.v_proj", source)

    q = processor.tasks["self_attn.q_proj"]["inputs"][0]
    k = processor.tasks["self_attn.k_proj"]["inputs"][0]
    v = processor.tasks["self_attn.v_proj"]["inputs"][0]

    assert q is k is v
    # Values still match an independent host copy of the capture.
    assert torch.equal(q, source.detach())


def test_inference_tensor_captures_share_one_host_copy():
    """Calibration captures are inference tensors, which have no ``_version``.

    The sibling dedup used to key on ``feature._version`` and silently return
    ``None`` for every real capture. Count the host materializations so a
    regression that disables sharing again fails here rather than only showing
    up as extra D2H traffic.
    """

    processor = _processor()
    materialized = []
    original = processor._apply_active_keep_mask_to_feature
    processor._apply_active_keep_mask_to_feature = lambda feature: (
        materialized.append(feature) or original(feature)
    )

    with torch.inference_mode():
        source = torch.randn(2, 6, 8)
        with pytest.raises(RuntimeError):
            # The version counter is what made the old key unusable.
            int(source._version)
        _record(processor, "self_attn.q_proj", source)
        _record(processor, "self_attn.k_proj", source)
        _record(processor, "self_attn.v_proj", source)

    q = processor.tasks["self_attn.q_proj"]["inputs"][0]
    k = processor.tasks["self_attn.k_proj"]["inputs"][0]
    v = processor.tasks["self_attn.v_proj"]["inputs"][0]

    assert q is k is v
    assert len(materialized) == 1, "the second and third captures must hit the cache"
    assert torch.equal(q, source.detach())


def test_distinct_sources_are_not_shared():
    processor = _processor()
    first = torch.randn(2, 6, 8)
    second = torch.randn(2, 6, 8)

    _record(processor, "self_attn.q_proj", first)
    _record(processor, "self_attn.k_proj", second)

    q = processor.tasks["self_attn.q_proj"]["inputs"][0]
    k = processor.tasks["self_attn.k_proj"]["inputs"][0]
    assert q is not k
    assert torch.equal(q, first.detach())
    assert torch.equal(k, second.detach())


def test_mutated_source_is_not_reused():
    processor = _processor()
    source = torch.randn(2, 6, 8)

    _record(processor, "self_attn.q_proj", source)
    source.add_(1.0)  # bumps ``_version``; the cached copy is stale now
    _record(processor, "self_attn.k_proj", source)

    q = processor.tasks["self_attn.q_proj"]["inputs"][0]
    k = processor.tasks["self_attn.k_proj"]["inputs"][0]
    assert q is not k
    assert torch.equal(k, source.detach())


def test_capture_cache_is_per_batch():
    processor = _processor()
    batch = [0]
    processor.current_batch_index = lambda: batch[0]
    source = torch.randn(2, 6, 8)

    _record(processor, "a", source)
    batch[0] = 1
    _record(processor, "b", source)

    assert processor.tasks["a"]["inputs"][0] is not processor.tasks["b"]["inputs"][0]
    assert processor._capture_host_batch == 1


def test_layer_input_features_concatenates_identical_lists_once():
    processor = _processor()
    batch = [0]
    processor.current_batch_index = lambda: batch[0]
    first = torch.randn(2, 6, 8)
    second = torch.randn(2, 6, 8)

    for batch_index, source in enumerate((first, second)):
        batch[0] = batch_index
        _record(processor, "q", source)
        _record(processor, "k", source)

    state = types.SimpleNamespace(modules={"q": None, "k": None})
    features = processor._layer_input_features(state)

    # Both siblings captured the same two batches, so one concatenation serves both.
    assert features["q"] is features["k"]
    expected = torch.cat([first.detach(), second.detach()], dim=0)
    assert torch.equal(features["q"], expected)


def _scale_module():
    module = nn.Module()
    module.prev = nn.Linear(8, 4, bias=False)
    module.fc_a = nn.Linear(4, 4, bias=False)
    module.fc_b = nn.Linear(4, 4, bias=False)
    return module


def test_apply_scale_divides_shared_feature_once():
    torch.manual_seed(0)
    module = _scale_module()
    scales = torch.rand(4) + 0.5

    shared = torch.randn(3, 4)
    original = shared.clone()
    feature_dict = {"fc_a": shared, "fc_b": shared}
    scales_list = [("prev", ["fc_a", "fc_b"], scales, 0.0)]

    apply_scale(module, scales_list, input_feat_dict=feature_dict)

    # Independent oracle: exactly one division of the captured activation.
    expected = original / scales.view(1, -1)
    assert torch.equal(feature_dict["fc_a"], expected)
    assert feature_dict["fc_a"] is feature_dict["fc_b"]


def test_apply_scale_restores_shared_feature_for_a_second_group():
    torch.manual_seed(1)
    module = _scale_module()
    first_scales = torch.rand(4) + 0.5
    second_scales = torch.rand(4) + 0.5

    shared = torch.randn(3, 4)
    original = shared.clone()
    feature_dict = {"fc_a": shared, "fc_b": shared}
    scales_list = [
        ("prev", ["fc_a"], first_scales, 0.0),
        ("prev", ["fc_b"], second_scales, 0.0),
    ]

    apply_scale(module, scales_list, input_feat_dict=feature_dict)

    # The second group must start from the pristine capture, not from the value
    # the first group already divided.
    assert torch.equal(shared, original / second_scales.view(1, -1))


def test_apply_scale_leaves_unshared_features_per_name():
    torch.manual_seed(2)
    module = _scale_module()
    scales = torch.rand(4) + 0.5

    first = torch.randn(3, 4)
    second = torch.randn(3, 4)
    first_original = first.clone()
    second_original = second.clone()
    feature_dict = {"fc_a": first, "fc_b": second}
    scales_list = [("prev", ["fc_a", "fc_b"], scales, 0.0)]

    apply_scale(module, scales_list, input_feat_dict=feature_dict)

    assert torch.equal(feature_dict["fc_a"], first_original / scales.view(1, -1))
    assert torch.equal(feature_dict["fc_b"], second_original / scales.view(1, -1))


def test_chunked_reference_outputs_stay_on_the_replay_device():
    processor = _processor()
    processor.qcfg = types.SimpleNamespace(scale_search_chunked_activations=True)
    reference = [torch.randn(2, 4, 6), torch.randn(1, 4, 6)]

    def _iter(prepared_x, module, kwargs, batch_size=None):
        assert prepared_x is sentinel
        assert batch_size == 3
        yield from reference

    sentinel = object()
    processor._iter_prepared_forward_outputs = _iter
    outputs = processor._build_chunked_reference_outputs(sentinel, nn.Identity(), {}, 3)

    assert len(outputs) == 2
    for chunk, expected in zip(outputs, reference):
        # No host round-trip: the scale grid receives device-resident chunks.
        assert chunk.device == expected.device
        assert torch.equal(chunk, expected)


ACTIVATION_SHAPE = (32, 64, 512)
CHUNK_BYTES = 64 << 10


def _search_fixture(x):
    """A real processor over one norm + linear group."""

    width = x.shape[-1]
    layer = nn.Sequential(
        nn.LayerNorm(width, dtype=x.dtype),
        nn.Linear(width, 32, bias=False, dtype=x.dtype),
    ).eval()
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=32, sym=True),
        calibration=CalibrationConfig(),
        gptq_model=types.SimpleNamespace(model=layer, rotary_embedding=None),
        model=layer,
    )
    processor.max_chunk_memory = CHUNK_BYTES
    processor._quant_batch_size = 1
    return processor, layer


def _search(processor, layer, x, captured=None, x_mean_override=None):
    """Run the production scale search, optionally scoring with a given mean."""

    original = processor._compute_best_scale

    def scoring(x_arg, w_mean, x_mean, *args, **kwargs):
        if captured is not None:
            captured["x_mean"] = x_mean.detach().clone()
        if x_mean_override is not None:
            x_mean = x_mean_override
        return original(x_arg, w_mean, x_mean, *args, **kwargs)

    processor._compute_best_scale = scoring
    try:
        return processor._search_best_scale(layer, layer[0], [layer[1]], x)
    finally:
        del processor._compute_best_scale


def _full_magnitude_mean(x, chunk_bytes):
    """The pre-fix reduction: flatten the whole ``|x|``, then reduce it in chunks."""

    flat = x.abs().view(-1, x.shape[-1])
    num_elements, num_channels = flat.shape
    chunk_size = max(1, min(chunk_bytes // (4 * num_channels), num_elements))
    total = torch.zeros(num_channels, dtype=torch.float32, device=x.device)
    for start in range(0, num_elements, chunk_size):
        total += flat[start : start + chunk_size].to(torch.float32).sum(dim=0)
    return (total / num_elements).to(x.dtype)


class _MagnitudeProbe(TorchDispatchMode):
    """Track the live magnitude storage of the source without retaining it.

    The probe weak-references the storage of every ``|x|`` result, so an alias
    such as ``inp.abs().view(...)`` keeps it observably alive while the probe
    never extends its lifetime. The old implementation kept the whole ``|x|``
    alive through the grid, and a strong reference here would hide that (#5765).
    """

    def __init__(self, source):
        super().__init__()
        self._source_ptr = source.untyped_storage().data_ptr()
        self._refs = {}
        self.peak_bytes = 0

    def live_bytes(self):
        total = 0
        dead = []
        for ptr, (ref, nbytes) in self._refs.items():
            if ref() is None:
                dead.append(ptr)
            else:
                total += nbytes
        for ptr in dead:
            del self._refs[ptr]
        return total

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.abs.default:
            operand = args[0]
            if operand.untyped_storage().data_ptr() == self._source_ptr:
                storage = result.untyped_storage()
                self._refs[storage.data_ptr()] = (weakref.ref(storage), storage.nbytes())
        self.peak_bytes = max(self.peak_bytes, self.live_bytes())
        return result


def test_scale_search_does_not_retain_the_activation_magnitude_5765():
    """The whole ``|x|`` must not outlive the chunk reduction.

    ``_search_best_scale`` used to bind ``inp.abs().view(...)`` for the chunked
    mean and then keep it referenced by the frame through the FP reference and
    all 20 candidate replays. Only the FP32 conversion was chunked, so a
    ``[32, 64, 512]`` BF16 capture left a 2 MiB buffer live for each of the 672
    one-batch forwards. The magnitude is now built per chunk and released before
    the replays; the selected scales must not move.
    """

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        x = torch.randn(*ACTIVATION_SHAPE, dtype=torch.bfloat16)
        processor, layer = _search_fixture(x)
        captured = {}

        probe = _MagnitudeProbe(x)
        forward_rows = []
        live_magnitude = []

        def observe(_module, inputs):
            forward_rows.append(inputs[0].shape[0])
            live_magnitude.append(probe.live_bytes())

        handle = layer[1].register_forward_pre_hook(observe)
        try:
            with probe:
                result = _search(processor, layer, x, captured=captured)
        finally:
            handle.remove()

        best_scales, loss = result[2:]

        # 2048 token rows / one-row micro-batches * 20 ratios: the chunked grid
        # still runs the same number of forwards as before the fix.
        assert len(forward_rows) == 672
        assert max(forward_rows) == 1
        # No activation-sized magnitude is alive while any replay runs, and the
        # reduction never holds more than one chunk.
        assert max(live_magnitude) == 0
        assert probe.peak_bytes <= CHUNK_BYTES

        # The only value the fix changes is the mean the grid scores against.
        oracle_mean = _full_magnitude_mean(x, CHUNK_BYTES)
        assert torch.equal(captured["x_mean"], oracle_mean)
        # Scoring the same grid with the independently computed pre-fix mean
        # leaves the chosen scales and the loss bitwise unchanged.
        reference_scales, reference_loss = _search(processor, layer, x, x_mean_override=oracle_mean)[2:]
        assert torch.equal(best_scales, reference_scales)
        assert loss == reference_loss
        assert best_scales.isfinite().all()
    finally:
        torch.set_num_threads(previous_threads)


SMALL_CASES = [
    pytest.param((8, 16, 64), torch.bfloat16, False, id="bf16"),
    pytest.param((8, 16, 64), torch.float16, False, id="fp16"),
    pytest.param((8, 16, 64), torch.float32, False, id="fp32"),
    pytest.param((100, 64), torch.bfloat16, False, id="2d-partial-final-chunk"),
    pytest.param((8, 16, 64), torch.bfloat16, True, id="noncontiguous"),
]


@pytest.mark.parametrize("shape,dtype,noncontiguous", SMALL_CASES)
def test_chunked_activation_mean_matches_the_full_magnitude_reduction_5765(shape, dtype, noncontiguous):
    """Per-chunk ``abs()`` must reproduce the pre-fix reduction bit for bit.

    Seven-row chunks over 128 (and 100) token rows leave a final partial chunk,
    and the non-contiguous case cannot use the flattening view at all. Every
    case must still match the whole-magnitude reduction exactly, so the chosen
    scales stay bitwise identical.
    """

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generator = torch.Generator().manual_seed(5765)
        if noncontiguous:
            base = torch.randn(shape[0], shape[1] * 2, shape[2], dtype=dtype, generator=generator)
            x = base[:, ::2]
            assert not x.is_contiguous()
        else:
            x = torch.randn(*shape, dtype=dtype, generator=generator)

        rows = x.numel() // x.shape[-1]
        budget = 4 * x.shape[-1] * 7
        chunk_rows = max(1, min(budget // (4 * x.shape[-1]), rows))
        # The fixture must cross several chunks and end on a partial one.
        assert chunk_rows < rows
        assert rows % chunk_rows != 0

        processor, layer = _search_fixture(x)
        processor.max_chunk_memory = budget

        captured = {}
        _search(processor, layer, x, captured=captured)
        assert torch.equal(captured["x_mean"], _full_magnitude_mean(x, budget))
    finally:
        torch.set_num_threads(previous_threads)
