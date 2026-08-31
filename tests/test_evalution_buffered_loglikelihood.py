# GPU=-1
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from tests import eval as eval_module


@dataclass
class _Output:
    logprob: float
    is_greedy: bool
    token_count: int
    metadata: dict


def test_evalution_loglikelihood_reductions_are_buffered(monkeypatch) -> None:
    class FakeSession:
        _score_chunks = lambda self, chunks, *, batch_size: []

    fake_transformers_common = SimpleNamespace(
        BaseTransformerSession=FakeSession,
        LoglikelihoodOutput=_Output,
        loglikelihood_progress_title=lambda _metadata: None,
        manual_progress=lambda *args, **kwargs: None,
    )
    import_module = eval_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "evalution.engines.transformers_common":
            return fake_transformers_common
        return import_module(name)

    monkeypatch.setattr(eval_module.importlib, "import_module", fake_import_module)
    eval_module._install_buffered_loglikelihood_d2h(object())
    scorer = FakeSession._score_chunks
    assert getattr(scorer, "_qvq_buffered_d2h", False)

    class Model:
        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            logits = torch.full((batch, length, 16), -4.0)
            for row in range(batch):
                for position in range(length - 1):
                    logits[row, position, int(input_ids[row, position + 1])] = 4.0
            return SimpleNamespace(logits=logits)

    session = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        input_device=torch.device("cpu"),
        model=Model(),
        _scoring_attention_context=lambda: nullcontext(),
    )
    chunks = [
        SimpleNamespace(
            input_ids=[index + 1, index + 2],
            score_start=1,
            score_count=1,
            metadata={"_evalution_disable_loglikelihood_chunk_progress": True},
        )
        for index in range(3)
    ]

    outputs = scorer(session, chunks, batch_size=2)
    assert len(outputs) == len(chunks)
    assert all(output.is_greedy for output in outputs)
    assert all(output.token_count == 1 for output in outputs)
    assert outputs[0].logprob == outputs[1].logprob == outputs[2].logprob
