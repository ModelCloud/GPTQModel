# GPU=-1
from __future__ import annotations

from types import SimpleNamespace

from tests import eval as eval_module


def test_build_evalution_suite_defaults_mmlu_stream_true() -> None:
    recorded_kwargs = {}

    def fake_mmlu(**kwargs):
        recorded_kwargs.update(kwargs)
        return kwargs

    eval_module._build_evalution_suite(
        evalution=SimpleNamespace(benchmarks=SimpleNamespace(mmlu=fake_mmlu)),
        task_name="mmlu_stem",
        apply_chat_template=False,
        batch_size=64,
        generation_settings={},
        suite_kwargs={},
    )

    assert recorded_kwargs["subsets"] == "stem"
    assert recorded_kwargs["batch_size"] == 64
    assert recorded_kwargs["stream"] is True


def test_build_evalution_suite_preserves_explicit_stream_override() -> None:
    recorded_kwargs = {}

    def fake_mmlu(**kwargs):
        recorded_kwargs.update(kwargs)
        return kwargs

    eval_module._build_evalution_suite(
        evalution=SimpleNamespace(benchmarks=SimpleNamespace(mmlu=fake_mmlu)),
        task_name="mmlu",
        apply_chat_template=False,
        batch_size=32,
        generation_settings={},
        suite_kwargs={"stream": False},
    )

    assert recorded_kwargs["batch_size"] == 32
    assert recorded_kwargs["stream"] is False
