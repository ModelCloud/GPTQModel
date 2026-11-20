# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Auto calibration batching: the largest power-of-two batch that fits.

Measured on a 0.5B model (32 rows, ~660 tokens each): batch 1 -> 8 -> 16 took
180 s -> 46 s -> 36 s, while NLL against three held-out slices stayed inside the
corpus noise and two runs at the same batch produced identical weights. These
tests pin the selection arithmetic, not the speedup.
"""

import types

import pytest
import torch

from gptqmodel.models.base import (
    BaseQModel,
    _calibration_row_count,
    _calibration_tokens_per_row,
)
from gptqmodel.quantization.config import CalibrationConfig
from gptqmodel.utils.calibration import (
    AUTO_BATCH_MAX_ROWS,
    AUTO_BATCH_SIZE,
    auto_batch_size,
)


def _pick(**overrides):
    base = {
        "rows": 32, "tokens_per_row": 660, "hidden_size": 896, "num_layers": 24,
        "dtype_bytes": 2, "free_bytes": 20 * 1024 ** 3,
    }
    base.update(overrides)
    return auto_batch_size(**base)


def test_ample_memory_picks_the_row_cap():
    assert _pick() == AUTO_BATCH_MAX_ROWS


def test_memory_binds_before_the_row_cap():
    # 660 * 896 * 24 * 2 bytes * 3 ~= 85 MB per row; 1 GiB * 0.25 / 85 MB = 3 -> 2
    assert _pick(free_bytes=1024 ** 3) == 2


def test_never_more_rows_than_the_dataset_has():
    assert _pick(rows=3) == 2
    assert _pick(rows=1) == 1


def test_never_below_one():
    assert _pick(free_bytes=0) == 1
    assert _pick(rows=0) == 1


def test_unmeasurable_device_falls_back_to_one():
    """Guessing a batch on an unknown device is how a run OOMs halfway through."""

    assert _pick(free_bytes=None) == 1


def test_batch_is_a_power_of_two():
    for free_gib in (0.5, 1, 2, 4, 8, 16, 64):
        batch = _pick(free_bytes=int(free_gib * 1024 ** 3))
        assert batch & (batch - 1) == 0


def test_bigger_rows_allow_a_smaller_batch():
    small_rows = _pick(tokens_per_row=128, free_bytes=2 * 1024 ** 3)
    big_rows = _pick(tokens_per_row=2048, free_bytes=2 * 1024 ** 3)
    assert big_rows <= small_rows


def test_config_accepts_the_auto_sentinel():
    config = CalibrationConfig(dataset=["a"], batch_size=AUTO_BATCH_SIZE)
    assert config.batch_size == "auto"


@pytest.mark.parametrize("value", ["AUTO", "autos", "", "16", 0, -1, 1.5, True])
def test_config_rejects_other_values(value):
    with pytest.raises((TypeError, ValueError)):
        CalibrationConfig(dataset=["a"], batch_size=value)


def _stub_qmodel(*, device="cpu", hidden=64, layers=2):
    parameters = [torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.bfloat16)]
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers),
        parameters=lambda: iter(parameters),
    )
    return types.SimpleNamespace(model=model, quantize_config=types.SimpleNamespace(device=device))


def test_cpu_runs_stay_at_one_row():
    stub = _stub_qmodel()
    calibration = CalibrationConfig(dataset=["x"] * 16, batch_size=AUTO_BATCH_SIZE)
    assert BaseQModel._auto_calibration_batch_size(stub, calibration) == 1


def test_row_count_and_token_estimate_come_from_the_dataset():
    stub = _stub_qmodel()
    calibration = CalibrationConfig(dataset=["x"] * 7, batch_size=AUTO_BATCH_SIZE, concat_size=1024)
    assert _calibration_row_count(calibration) == 7
    # A declared concat size is the row length; no tokenizer is needed.
    assert _calibration_tokens_per_row(stub, calibration) == 1024

class _WordTokenizer:
    """One token per whitespace-separated word."""

    def __call__(self, text):
        return {"input_ids": list(range(len(text.split())))}


def _tokenizer_qmodel():
    return types.SimpleNamespace(tokenizer=_WordTokenizer())


TEXTS = [" ".join(["w"] * (index + 1)) for index in range(64)]
TOKEN_IDS = [list(range(index + 1)) for index in range(64)]
EXPECTED_ROWS = 64
EXPECTED_TOKENS = 64


def _containers():
    pandas = pytest.importorskip("pandas")
    return {
        "list_of_str": TEXTS,
        "list_of_dict_text": [{"text": text} for text in TEXTS],
        "columnar_text": {"text": TEXTS},
        "dataframe": pandas.DataFrame({"text": TEXTS}),
        "list_of_tensor_rows": [torch.tensor([ids]) for ids in TOKEN_IDS],
        "list_of_dict_ids": [
            {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
            for ids in TOKEN_IDS
        ],
        "columnar_ids": {
            "input_ids": TOKEN_IDS,
            "attention_mask": [[1] * len(ids) for ids in TOKEN_IDS],
        },
        "short_rows_first": ["w"] * 4 + TEXTS,
    }


@pytest.mark.parametrize("name", list(_containers()))
def test_auto_batch_sizes_every_supported_input_form(name):
    """The estimate must read the rows, not the container (#5505).

    A ``DataFrame`` used to report one token per row (iterating it yields the
    column label), a columnar mapping reported its column count as the row
    count, a ``[1, S]`` tensor row reported ``shape[-2] == 1`` and a bare tensor
    row raised from ``Tensor.values()``.
    """

    dataset = _containers()[name]
    calibration = CalibrationConfig(dataset=dataset, batch_size=AUTO_BATCH_SIZE)
    expected_rows = EXPECTED_ROWS + (4 if name == "short_rows_first" else 0)

    assert _calibration_row_count(calibration) == expected_rows
    assert _calibration_tokens_per_row(_tokenizer_qmodel(), calibration) == EXPECTED_TOKENS


def test_auto_batch_reads_an_hf_dataset_columnwise():
    pytest.importorskip("datasets")
    from datasets import Dataset

    calibration = CalibrationConfig(
        dataset=Dataset.from_dict({"text": TEXTS}), batch_size=AUTO_BATCH_SIZE
    )

    assert _calibration_row_count(calibration) == EXPECTED_ROWS
    assert _calibration_tokens_per_row(_tokenizer_qmodel(), calibration) == EXPECTED_TOKENS


def test_auto_batch_ignores_a_single_row_mapping():
    calibration = CalibrationConfig(
        dataset={"input_ids": [1, 2, 3]}, batch_size=AUTO_BATCH_SIZE
    )

    assert _calibration_row_count(calibration) == 1
    assert _calibration_tokens_per_row(_tokenizer_qmodel(), calibration) == 3



def _materialize(config):
    """Apply the #5567 fix through a runtime lookup.

    Kept out of the module-level import so this file still collects on the
    pre-fix tree and only the regression nodes fail.
    """

    from gptqmodel.models.base import _materialized_calibration_config

    return _materialized_calibration_config(config)


def _one_shot_factories():
    return (
        lambda: (text for text in TEXTS),
        lambda: iter(TEXTS),
        lambda: map(str, TEXTS),
    )


def test_auto_batch_materialises_a_one_shot_dataset():
    """Sizing must not consume a generator/iter/map before preparation (#5567)."""

    for factory in _one_shot_factories():
        config = CalibrationConfig(dataset=factory(), batch_size="auto")
        bound = _materialize(config)

        assert bound is not config
        assert isinstance(bound.dataset, list)
        assert _calibration_row_count(bound) == 64
        # The EoRA pass sizes again: a materialised input keeps the rows.
        assert _calibration_row_count(bound) == 64
        assert list(bound.dataset) == TEXTS


def test_auto_batch_materialises_a_one_shot_adapter_dataset():
    config = CalibrationConfig(
        dataset=TEXTS,
        adapter_dataset=(text for text in TEXTS),
        batch_size="auto",
    )
    bound = _materialize(config)

    assert bound.dataset is TEXTS
    assert isinstance(bound.adapter_dataset, list)
    assert list(bound.adapter_dataset) == TEXTS


def test_materialisation_keeps_sized_containers_unchanged():
    for dataset in (TEXTS, tuple(TEXTS), {"text": TEXTS}):
        config = CalibrationConfig(dataset=dataset, batch_size="auto")
        assert _materialize(config) is config


def test_materialisation_keeps_a_bound_preparer():
    def preparer(**kwargs):
        return kwargs
    config = CalibrationConfig(
        dataset=(text for text in TEXTS), batch_size="auto"
    ).bind(preparer)

    bound = _materialize(config)

    assert bound.prepare_dataset is preparer
    assert isinstance(bound.dataset, list)


class _CompositeTextConfig:
    def __init__(self, hidden_size, num_hidden_layers):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class _CompositeConfig:
    """A VLM/omni config: decoder dims live on ``text_config``, top-level is None."""

    hidden_size = None
    num_hidden_layers = None

    def __init__(self, text_config):
        self.text_config = text_config

    def get_text_config(self, *args, **kwargs):
        return self.text_config


def _composite_qmodel(text_config, *, device="cuda:0"):
    parameters = [torch.zeros(1, dtype=torch.bfloat16)]
    model = types.SimpleNamespace(
        config=_CompositeConfig(text_config),
        parameters=lambda: iter(parameters),
    )
    return types.SimpleNamespace(model=model, quantize_config=types.SimpleNamespace(device=device))


def test_composite_config_reads_the_text_config_dims(monkeypatch):
    """The guard must use the decoder dims a composite config hides (#5568)."""

    free = 40 * 1024 ** 3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (free, 80 * 1024 ** 3))
    stub = _composite_qmodel(_CompositeTextConfig(2048, 28))
    calibration = CalibrationConfig(dataset=["x"] * 64, batch_size="auto", concat_size=8192)

    batch = BaseQModel._auto_calibration_batch_size(stub, calibration)
    expected = auto_batch_size(
        rows=64,
        tokens_per_row=8192,
        hidden_size=2048,
        num_layers=28,
        dtype_bytes=2,
        free_bytes=free,
    )

    assert batch == expected
    assert batch < AUTO_BATCH_MAX_ROWS


def test_missing_config_dims_fall_back_to_one_row(monkeypatch):
    free = 40 * 1024 ** 3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (free, 80 * 1024 ** 3))
    stub = _composite_qmodel(_CompositeTextConfig(None, None))
    calibration = CalibrationConfig(dataset=["x"] * 64, batch_size="auto", concat_size=8192)

    assert BaseQModel._auto_calibration_batch_size(stub, calibration) == 1


class _ChatTokenizer:
    """A tokenizer whose template renders the conversation text."""

    chat_template = "chat"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs
    ):
        text = " ".join(message["content"] for message in messages)
        if tools:
            text += " " + " ".join(["schema"] * (10 * len(tools)))
        return text

    def __call__(self, text):
        return {"input_ids": list(range(len(text.split())))}


def test_messages_rows_are_measured_through_the_chat_template():
    """Chat rows used to keep the 512 default whatever their real length (#5568)."""

    row = {"messages": [{"role": "user", "content": " ".join(["w"] * 900)}]}
    calibration = CalibrationConfig(dataset=[row], batch_size="auto")
    qmodel = types.SimpleNamespace(tokenizer=_ChatTokenizer())

    assert _calibration_tokens_per_row(qmodel, calibration) == 900


def test_messages_rows_keep_the_tools_block_the_template_renders():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    plain = {"messages": messages}
    with_tools = {"messages": messages, "tools": tools}
    qmodel = types.SimpleNamespace(tokenizer=_ChatTokenizer())

    plain_len = _calibration_tokens_per_row(
        qmodel, CalibrationConfig(dataset=[plain], batch_size="auto")
    )
    with_tools_len = _calibration_tokens_per_row(
        qmodel, CalibrationConfig(dataset=[plain, with_tools], batch_size="auto")
    )

    assert with_tools_len > plain_len


def test_messages_rows_without_a_template_fall_back_to_the_text_column():
    row = {"messages": [{"role": "user", "content": "x"}], "text": " ".join(["w"] * 40)}
    calibration = CalibrationConfig(dataset=[row], batch_size="auto")

    class _NoTemplate:
        def __call__(self, text):
            return {"input_ids": list(range(len(text.split())))}

    qmodel = types.SimpleNamespace(tokenizer=_NoTemplate())

    assert _calibration_tokens_per_row(qmodel, calibration) == 40
