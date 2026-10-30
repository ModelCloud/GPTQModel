# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the OpenAI-compatible server.

These cover the three defects verified against the previous implementation:

1. unsafe defaults (``0.0.0.0:80``) with no authentication,
2. any request carrying ``stop`` returning HTTP 500, and
3. chat completions serialized with the legacy ``text_completion`` schema.
"""

import asyncio
import inspect
import time
import warnings

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.generation_result import GenerationResult
from gptqmodel.utils.openai_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    AuthenticationMiddleware,
    OpenAiServer,
    _bearer_token_matches,
    _token_digests,
)


pytestmark = [pytest.mark.cpu]


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    # Byte-level BPE over printable ASCII so `StopStringCriteria` can build its vocab.
    # IDs must be contiguous for the embedding lookup in the stub Llama model.
    vocab = {chr(32 + offset): offset for offset in range(95)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="~", pad_token=" ")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}assistant:"
    )
    return tokenizer


class _StubConfig:
    name_or_path = "stub-model"


# Printable ids from `_tiny_tokenizer`; the last one is EOS ("~").
_GENERATED_TEXT_IDS = [65, 66, 67, 68, 69, 70, 71]
_GENERATED_EOS_ID = 94


class _StubModel:
    """Deterministic stand-in for ``GPTQModel``.

    Generation is scripted instead of sampled from randomly initialised
    weights, so finish reasons and token counts are a property of the script
    rather than of RNG state or test ordering. ``_HFStubModel`` keeps real
    Transformers generation for the cases that must exercise ``stop_strings``.
    """

    def __init__(self, tokenizer, generated_ids=None, ends_with_eos: bool = True):
        self.tokenizer = tokenizer
        self.config = _StubConfig()
        self.device = "cpu"
        self.calls = []
        if generated_ids is None:
            generated_ids = list(_GENERATED_TEXT_IDS)
            if ends_with_eos:
                generated_ids.append(_GENERATED_EOS_ID)
        self.generated_ids = list(generated_ids)

    def scripted_row(self, max_new_tokens: int) -> list:
        # Truncating never re-adds EOS, so a truncated row really was cut short.
        return self.generated_ids[: max(0, max_new_tokens)]

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["input_ids"]
        sequences = kwargs.get("num_return_sequences") or 1
        row = self.scripted_row(kwargs.get("max_new_tokens", len(self.generated_ids)))
        generated = torch.tensor([row] * sequences, dtype=prompt.dtype)
        return torch.cat([prompt.repeat(sequences, 1), generated], dim=1)


class _HFStubModel(_StubModel):
    """Real Transformers generation over a tiny random model.

    Used only where the test must prove the request survives the real
    ``stop_strings`` machinery instead of being served by the script.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        self._hf = LlamaForCausalLM(config).eval()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._hf.generate(**kwargs)


@pytest.fixture(scope="module")
def tokenizer():
    return _tiny_tokenizer()


@pytest.fixture()
def model(tokenizer):
    return _StubModel(tokenizer)


@pytest.fixture()
def client(model):
    server = OpenAiServer(model=model)
    yield TestClient(server.app)
    server.shutdown()


def _chat(client, headers=None, **payload):
    body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}]}
    body.update(payload)
    return client.post("/v1/chat/completions", json=body, headers=headers)


def test_defaults_are_left_to_the_caller(model):
    # The bind address is a deployment decision, so the library keeps the
    # historical defaults instead of silently changing where the server listens.
    assert DEFAULT_HOST == "0.0.0.0"
    assert DEFAULT_PORT == 80

    defaults = inspect.signature(BaseQModel.serve).parameters
    assert defaults["host"].default == "0.0.0.0"
    assert defaults["port"].default == 80
    # Authentication is opt-in, and the key is part of the public API.
    assert "api_key" in defaults

    # A list of keys is accepted so one deployment can rotate keys.
    server = OpenAiServer(model=model, api_key=["k1", "k2"])
    assert server.api_keys == ["k1", "k2"]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.13.153"])
def test_exposed_bind_without_key_warns(model, host):
    server = OpenAiServer(model=model)
    with pytest.warns(UserWarning, match="without an API key"):
        server._warn_if_exposed(host, server.api_keys)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_bind_does_not_warn(model, host):
    server = OpenAiServer(model=model)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        server._warn_if_exposed(host, server.api_keys)


def test_exposed_bind_with_key_does_not_warn(model):
    server = OpenAiServer(model=model, api_key="s3cret")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        server._warn_if_exposed("0.0.0.0", server.api_keys)


def test_chat_completion_uses_chat_schema(client):
    # Budget must exceed the scripted sequence, otherwise it would end on `length`.
    response = _chat(client, max_tokens=len(_GENERATED_TEXT_IDS) + 1)
    assert response.status_code == 200

    body = response.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert "text" not in choice
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_stop_parameter_stops_generation_instead_of_failing(tokenizer):
    # Real Transformers generation: this is the regression for `stop_strings`
    # without a tokenizer, so the stub must not stand in for the backend here.
    model = _HFStubModel(tokenizer)
    client = TestClient(OpenAiServer(model=model).app)

    # Any in-vocabulary string exercises the same preprocessing path; deriving it
    # from sampled output would make the test depend on random weights.
    response = _chat(client, temperature=0, max_tokens=8, stop=["abc"])
    assert response.status_code == 200, response.text

    call = model.calls[-1]
    # Transformers requires the tokenizer next to `stop_strings`; without it generate() raises.
    assert call["stop_strings"] == ["abc"]
    assert call["tokenizer"] is model.tokenizer
    assert isinstance(response.json()["choices"][0]["message"]["content"], str)


def test_stop_accepts_a_plain_string(client, model):
    response = _chat(client, temperature=0, stop="abc")
    assert response.status_code == 200, response.text
    assert model.calls[-1]["stop_strings"] == ["abc"]


def test_blank_stop_entries_are_ignored(client, model):
    response = _chat(client, temperature=0, stop=["", None])
    assert response.status_code == 200, response.text
    assert "stop_strings" not in model.calls[-1]


def test_blank_scalar_stop_is_ignored_like_a_blank_list_entry(client, model):
    # `stop=""` used to be forwarded as [""], which either fails stop-string
    # preprocessing (500) or matches at position zero and drops the whole answer.
    response = _chat(client, temperature=0, stop="")
    assert response.status_code == 200, response.text
    assert "stop_strings" not in model.calls[-1]

    content = response.json()["choices"][0]["message"]["content"]
    assert content.strip(), "an empty stop string must not truncate the completion"


def test_blank_scalar_stop_is_ignored_for_runtime_adapters(model):
    # Runtime adapters receive the OpenAI-style `stop` instead of `stop_strings`.
    class _RuntimeStub(_StubModel):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self._runtime_generate = object()

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return "alpha beta"

    runtime_model = _RuntimeStub(model.tokenizer)
    client = TestClient(OpenAiServer(model=runtime_model).app)

    response = _chat(client, temperature=0, stop="")
    assert response.status_code == 200, response.text
    assert "stop" not in runtime_model.calls[-1]
    assert response.json()["choices"][0]["message"]["content"] == "alpha beta"


def test_greedy_decoding_does_not_request_multiple_sequences(client, model):
    # `num_return_sequences` without sampling makes Hugging Face raise, so n=1 must omit it.
    assert _chat(client, temperature=0).status_code == 200
    assert "num_return_sequences" not in model.calls[-1]

    assert _chat(client, temperature=0.7, n=2).status_code == 200
    assert model.calls[-1]["num_return_sequences"] == 2


def test_greedy_with_multiple_sequences_is_rejected_not_a_server_error(client, model):
    # `n > 1` without sampling makes the backend raise; that must be a 400, not a 500.
    response = _chat(client, temperature=0, n=2)
    assert response.status_code == 400, response.text
    assert "temperature" in response.json()["detail"]
    assert model.calls == [] or "num_return_sequences" not in model.calls[-1]


def test_finish_reason_is_length_when_the_token_budget_runs_out(client):
    # One token is well under the stub's stop condition, so generation is truncated.
    body = _chat(client, temperature=0, max_tokens=1).json()
    assert body["choices"][0]["finish_reason"] == "length"


def test_finish_reason_is_stop_when_a_stop_string_truncates(client):
    # The script emits seven characters, so a five character stop string leaves a
    # prefix: the completion is cut, and the reason is promoted from `length`.
    stop = "bcdef"
    body = _chat(client, temperature=0, max_tokens=len(_GENERATED_TEXT_IDS) + 1, stop=[stop]).json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == "a"


def test_usage_sums_tokens_across_returned_sequences(client, model):
    # 3 sequences of 6 tokens each; the script is fixed so the sum is exact.
    response = _chat(client, temperature=0.9, n=3, max_tokens=6)
    assert response.status_code == 200, response.text

    body = response.json()
    assert len(body["choices"]) == 3
    # Usage covers every returned sequence, so it cannot equal a single sequence's length.
    assert [choice["finish_reason"] for choice in body["choices"]] == ["length"] * 3
    assert body["usage"]["completion_tokens"] == 3 * 6
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 18


def test_finish_reason_and_usage_when_pad_and_eos_share_an_id(tokenizer):
    # Qwen and Mistral reuse one id for padding and EOS. Masking every pad id
    # also deleted the terminating EOS, which reported a real stop as `length`
    # and under-counted the sequence.
    eos = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id

    first, second = 65, 66

    class _Fixed(_StubModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            prompt = kwargs["input_ids"]
            # Row 0 ends on EOS; row 1 was cut by the budget. Hugging Face pads
            # the finished row with `pad_token_id`, which is the same EOS id here.
            rows = torch.tensor([[first, second, eos, eos], [first, second, 67, 68]])
            return torch.cat([prompt.repeat(len(rows), 1), rows], dim=1)

    client = TestClient(OpenAiServer(model=_Fixed(tokenizer)).app)
    body = _chat(client, temperature=0.7, n=2, max_tokens=4).json()

    assert [choice["finish_reason"] for choice in body["choices"]] == ["stop", "length"]
    assert [choice["message"]["content"] for choice in body["choices"]] == ["ab", "abcd"]
    # 3 tokens for the EOS row (including the EOS) plus 4 for the truncated row.
    assert body["usage"]["completion_tokens"] == 7


def test_runtime_adapter_text_does_not_fabricate_usage_or_finish_reason(model):
    class _TextAdapter(_StubModel):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self._runtime_generate = object()

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return "alpha beta"

    client = TestClient(OpenAiServer(model=_TextAdapter(model.tokenizer)).app)
    body = _chat(client, temperature=0).json()

    assert body["choices"][0]["message"]["content"] == "alpha beta"
    # Reported as unknown rather than as a fabricated zero or a constant "stop".
    assert body["choices"][0]["finish_reason"] is None
    assert body["usage"]["completion_tokens"] is None
    assert body["usage"]["total_tokens"] is None
    assert body["usage"]["prompt_tokens"] > 0


def test_runtime_adapter_can_report_its_own_usage_and_finish_reason(model):
    class _ReportingAdapter(_StubModel):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self._runtime_generate = object()

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [
                GenerationResult("alpha", completion_tokens=5, finish_reason="stop"),
                GenerationResult("beta", completion_tokens=7, finish_reason="length"),
            ]

    client = TestClient(OpenAiServer(model=_ReportingAdapter(model.tokenizer)).app)
    body = _chat(client, temperature=0.7, n=2).json()

    assert [choice["message"]["content"] for choice in body["choices"]] == ["alpha", "beta"]
    assert [choice["finish_reason"] for choice in body["choices"]] == ["stop", "length"]
    assert body["usage"]["completion_tokens"] == 12
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 12


def test_runtime_adapter_mappings_are_not_iterated_as_sequences(model):
    class _MappingAdapter(_StubModel):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self._runtime_generate = object()

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return {"text": "alpha", "meta": "ignored"}

    client = TestClient(OpenAiServer(model=_MappingAdapter(model.tokenizer)).app)
    body = _chat(client, temperature=0, max_tokens=4).json()

    # A mapping is one result; iterating it would emit one choice per key.
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["content"] == "alpha"


def test_runtime_adapter_unusable_result_is_an_error_not_silent_text(model):
    class _NoneAdapter(_StubModel):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self._runtime_generate = object()

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return None

    client = TestClient(OpenAiServer(model=_NoneAdapter(model.tokenizer)).app, raise_server_exceptions=False)
    response = _chat(client, temperature=0)

    assert response.status_code == 500
    assert "NoneType" in response.json()["detail"]


def test_generation_does_not_block_the_event_loop(model):
    # A synchronous handler stalls every other task for the whole generation,
    # so a liveness probe times out and a healthy service gets restarted.
    # A ticker measures the largest gap in the loop instead of the latency of a
    # request issued after the block, which cannot distinguish the two cases.
    class _Slow(_StubModel):
        def generate(self, **kwargs):
            time.sleep(0.4)
            return "alpha"

    server = OpenAiServer(model=_Slow(model.tokenizer))
    body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    gaps = []

    async def scenario():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            async def ticker():
                last = time.perf_counter()
                while True:
                    await asyncio.sleep(0.01)
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now

            tick = asyncio.create_task(ticker())
            response = await client.post("/v1/chat/completions", json=body)
            tick.cancel()
            return response.status_code

    assert asyncio.run(scenario()) == 200
    # A blocking handler keeps the loop from ticking at all, so an empty sample
    # is the worst outcome, not a reason to skip the check.
    assert gaps, "the ticker never ran: the loop was blocked for the whole generation"
    assert max(gaps) < 0.2, f"event loop stalled for {max(gaps):.2f}s during generation"


def test_concurrent_generations_are_serialised(model):
    # The endpoint runs in a threadpool now, so the model itself must stay
    # single-entry: one device cannot serve two generations at once.
    active = []
    overlap = []

    class _Tracking(_StubModel):
        def generate(self, **kwargs):
            active.append(1)
            if len(active) > 1:
                overlap.append(len(active))
            time.sleep(0.2)
            active.pop()
            return "alpha"

    server = OpenAiServer(model=_Tracking(model.tokenizer))
    body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}

    async def scenario():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            responses = await asyncio.gather(
                *[client.post("/v1/chat/completions", json=body) for _ in range(3)]
            )
            return [response.status_code for response in responses]

    assert asyncio.run(scenario()) == [200, 200, 200]
    assert overlap == [], f"{len(overlap)} concurrent generations reached the model"


def test_rejects_empty_messages(client):
    response = client.post("/v1/chat/completions", json={"model": "stub-model", "messages": []})
    assert response.status_code == 400


def test_no_authentication_required_when_no_key_configured(client):
    assert _chat(client).status_code == 200


def test_token_comparison_compares_every_configured_key(model):
    # Every configured key is compared, so the work does not reveal which one
    # matched, and the digests are fixed length so it cannot reveal key length.
    api_tokens = _token_digests(["k1", "k2"])
    assert len({len(token) for token in api_tokens}) == 1

    assert _bearer_token_matches("Bearer k1", api_tokens) is True
    assert _bearer_token_matches("Bearer k2", api_tokens) is True
    assert _bearer_token_matches("Bearer k", api_tokens) is False
    assert _bearer_token_matches("Bearer ", api_tokens) is False
    assert _bearer_token_matches("", api_tokens) is False
    assert _bearer_token_matches("Bearer anything", []) is False


def test_middleware_verify_token_accepts_any_configured_key(model):
    middleware = AuthenticationMiddleware(app=None, tokens=["k1", "k2"])

    def headers_of(value):
        raw = [(b"authorization", value)] if value else []
        return Headers(scope={"type": "http", "headers": raw})

    assert middleware.verify_token(headers_of(b"Bearer k1")) is True
    assert middleware.verify_token(headers_of(b"Bearer k2")) is True
    assert middleware.verify_token(headers_of(b"Bearer nope")) is False
    assert middleware.verify_token(headers_of(b"k1")) is False
    assert middleware.verify_token(headers_of(None)) is False


def test_configured_api_key_is_enforced(model):
    client = TestClient(OpenAiServer(model=model, api_key="s3cret").app)

    assert _chat(client).status_code == 401
    assert _chat(client, headers={"Authorization": "Bearer wrong"}).status_code == 401
    # A non-bearer scheme must not be accepted.
    assert _chat(client, headers={"Authorization": "s3cret"}).status_code == 401
    assert _chat(client, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_api_key_can_be_supplied_by_environment(model, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_OPENAI_API_KEY", "from-env")
    client = TestClient(OpenAiServer(model=model).app)

    assert _chat(client).status_code == 401
    assert _chat(client, headers={"Authorization": "Bearer from-env"}).status_code == 200


def test_explicit_api_key_argument_beats_environment(model, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_OPENAI_API_KEY", "from-env")
    client = TestClient(OpenAiServer(model=model, api_key="explicit").app)

    # Explicit keys replace the environment fallback instead of being merged into it.
    assert _chat(client, headers={"Authorization": "Bearer from-env"}).status_code == 401
    assert _chat(client, headers={"Authorization": "Bearer explicit"}).status_code == 200


@pytest.mark.parametrize("explicit", [[], (), ["", ""], ""])
def test_explicitly_empty_api_key_disables_environment_fallback(model, monkeypatch, explicit):
    # Only `None` may consult the environment; any explicit value is authoritative,
    # otherwise `api_key=[]` would silently re-enable an inherited key.
    monkeypatch.setenv("GPTQMODEL_OPENAI_API_KEY", "from-env")
    client = TestClient(OpenAiServer(model=model, api_key=explicit).app)

    assert _chat(client).status_code == 200
    assert _chat(client, headers={"Authorization": "Bearer from-env"}).status_code == 200


def test_multiple_api_keys_are_all_accepted(model):
    client = TestClient(OpenAiServer(model=model, api_key=["k1", "k2"]).app)

    assert _chat(client).status_code == 401
    assert _chat(client, headers={"Authorization": "Bearer k1"}).status_code == 200
    assert _chat(client, headers={"Authorization": "Bearer k2"}).status_code == 200
    assert _chat(client, headers={"Authorization": "Bearer k3"}).status_code == 401


def test_blank_env_key_counts_as_no_key(model, monkeypatch):
    # A blank key is filtered out, so it means "no key configured".
    monkeypatch.setenv("GPTQMODEL_OPENAI_API_KEY", "")
    assert _chat(TestClient(OpenAiServer(model=model).app)).status_code == 200


def test_blank_entries_do_not_weaken_a_configured_key(model):
    client = TestClient(OpenAiServer(model=model, api_key=["", "k1"]).app)

    assert _chat(client).status_code == 401
    assert _chat(client, headers={"Authorization": "Bearer "}).status_code == 401
    assert _chat(client, headers={"Authorization": "Bearer k1"}).status_code == 200


def test_health_endpoint_is_not_guarded(model):
    client = TestClient(OpenAiServer(model=model, api_key="s3cret").app)

    assert client.get("/v1/models").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_options_preflight_bypasses_authentication(model):
    client = TestClient(OpenAiServer(model=model, api_key="s3cret").app)

    response = client.options("/v1/chat/completions")
    assert response.status_code != 401


def test_shutdown_allowed_from_loopback(model):
    server = OpenAiServer(model=model)
    called = []
    server.shutdown = lambda: called.append(True)
    client = TestClient(server.app, client=("127.0.0.1", 50000))

    assert client.get("/shutdown").status_code == 200
    assert called == [True]


def test_shutdown_requires_the_key_from_loopback_when_one_is_configured(model):
    # A page on the server host can issue this cross-site GET, so loopback must
    # not substitute for the configured key.
    server = OpenAiServer(model=model, api_key="s3cret")
    called = []
    server.shutdown = lambda: called.append(True)
    client = TestClient(server.app, client=("127.0.0.1", 50000))

    assert client.get("/shutdown").status_code == 403
    assert called == []

    assert client.get("/shutdown", headers={"Authorization": "Bearer wrong"}).status_code == 403
    assert called == []

    assert client.get("/shutdown", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert called == [True]


def test_shutdown_rejected_from_remote_caller_without_key(model):
    server = OpenAiServer(model=model)
    called = []
    server.shutdown = lambda: called.append(True)
    client = TestClient(server.app, client=("203.0.113.9", 50000))

    assert client.get("/shutdown").status_code == 403
    assert called == []


def test_shutdown_allowed_from_remote_caller_with_key(model):
    server = OpenAiServer(model=model, api_key="s3cret")
    called = []
    server.shutdown = lambda: called.append(True)
    client = TestClient(server.app, client=("203.0.113.9", 50000))

    assert client.get("/shutdown", headers={"Authorization": "Bearer wrong"}).status_code == 403
    assert called == []

    assert client.get("/shutdown", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert called == [True]


def test_shutdown_is_not_reachable_without_valid_token_when_client_host_is_unparseable(model):
    # TestClient's default peer ("testclient") is not an IP, so it must not count as loopback.
    server = OpenAiServer(model=model)
    called = []
    server.shutdown = lambda: called.append(True)

    assert TestClient(server.app).get("/shutdown").status_code == 403
    assert called == []
