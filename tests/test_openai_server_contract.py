# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the OpenAI-compatible server.

These cover the three defects verified against the previous implementation:

1. unsafe defaults (``0.0.0.0:80``) with no authentication,
2. any request carrying ``stop`` returning HTTP 500, and
3. chat completions serialized with the legacy ``text_completion`` schema.
"""

import inspect
import warnings

import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.openai_server import DEFAULT_HOST, DEFAULT_PORT, OpenAiServer


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


class _StubModel:
    """Minimal stand-in for ``GPTQModel``: exercises the real routes, not real weights."""

    def __init__(self, tokenizer, max_new_tokens: int = 3):
        self.tokenizer = tokenizer
        self.config = _StubConfig()
        self.device = "cpu"
        self.max_new_tokens = max_new_tokens
        self.calls = []

        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        self._model = LlamaForCausalLM(config).eval()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        return self._model.generate(**kwargs)


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
    response = _chat(client)
    assert response.status_code == 200

    body = response.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert "text" not in choice
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] >= 0


def test_stop_parameter_stops_generation_instead_of_failing(client, model):
    # Baseline greedy text so the stop string is guaranteed to occur in the output.
    baseline = _chat(client, temperature=0).json()["choices"][0]["message"]["content"]
    assert baseline, "stub generation produced no text to truncate"

    response = _chat(client, temperature=0, stop=[baseline])
    assert response.status_code == 200, response.text

    call = model.calls[-1]
    # Transformers requires the tokenizer next to `stop_strings`; without it generate() raises.
    assert call["stop_strings"] == [baseline]
    assert call["tokenizer"] is model.tokenizer

    # OpenAI omits the stop sequence from the returned content.
    assert response.json()["choices"][0]["message"]["content"] == ""


def test_stop_accepts_a_plain_string(client, model):
    baseline = _chat(client, temperature=0).json()["choices"][0]["message"]["content"]
    response = _chat(client, temperature=0, stop=baseline)
    assert response.status_code == 200, response.text
    assert model.calls[-1]["stop_strings"] == [baseline]


def test_blank_stop_entries_are_ignored(client, model):
    response = _chat(client, temperature=0, stop=["", None])
    assert response.status_code == 200, response.text
    assert "stop_strings" not in model.calls[-1]


def test_greedy_decoding_does_not_request_multiple_sequences(client, model):
    # `num_return_sequences` without sampling makes Hugging Face raise, so n=1 must omit it.
    assert _chat(client, temperature=0).status_code == 200
    assert "num_return_sequences" not in model.calls[-1]

    assert _chat(client, temperature=0.7, n=2).status_code == 200
    assert model.calls[-1]["num_return_sequences"] == 2


def test_rejects_empty_messages(client):
    response = client.post("/v1/chat/completions", json={"model": "stub-model", "messages": []})
    assert response.status_code == 400


def test_no_authentication_required_when_no_key_configured(client):
    assert _chat(client).status_code == 200


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
