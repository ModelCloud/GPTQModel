# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import hashlib
import ipaddress
import os
import secrets
import threading
import time
import uuid
import warnings

import torch

from .generation_result import GenerationResult


try:
    import uvicorn
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from starlette.datastructures import Headers
except ModuleNotFoundError as exception:
    raise type(exception)(
        "GPT-QModel OpenAi serve required dependencies are not installed.",
        "Please install via `pip install gptqmodel[openai] --no-build-isolation`.",
    )

# Environment fallback so the token does not have to be passed on the command line.
API_KEY_ENV = "GPTQMODEL_OPENAI_API_KEY"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 80

# Only the inference API is authenticated. Liveness/root routes stay open so
# probes and browsers do not need the key.
GUARDED_PREFIX = ("/v1", "/v2", "/inference")


def _resolve_api_keys(api_key) -> list:
    """Resolve configured keys.

    Only ``None`` falls back to the environment. Any explicit value is
    authoritative, so ``api_key=[]`` means "no authentication" instead of
    silently re-enabling the environment key.
    """
    if api_key is None:
        env_key = os.environ.get(API_KEY_ENV)
        keys = [env_key] if env_key else []
    elif isinstance(api_key, str):
        keys = [api_key]
    else:
        keys = list(api_key)
    # Blank entries must not silently disable authentication.
    return [key for key in keys if key]


def _token_digests(tokens: list) -> list:
    # Digesting first gives the comparison a fixed-length operand, so it cannot
    # expose the configured key length.
    return [hashlib.sha256(token.encode("utf-8")).digest() for token in tokens]


def _bearer_token_matches(authorization: str, api_tokens: list) -> bool:
    if not authorization:
        return False
    scheme, _, param = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return False
    param_digest = hashlib.sha256(param.encode("utf-8")).digest()
    matched = False
    for token_digest in api_tokens:
        # `|=` instead of `break` keeps the work constant regardless of which key matches.
        matched |= secrets.compare_digest(param_digest, token_digest)
    return matched


class AuthenticationMiddleware:
    """Pure ASGI bearer-token authentication.

    Skipped in two cases:
        1. The HTTP method is OPTIONS (CORS preflight).
        2. The request path does not start with one of ``GUARDED_PREFIX``.
    """

    def __init__(self, app, tokens: list) -> None:
        self.app = app
        self.api_tokens = _token_digests(tokens)

    def verify_token(self, headers: Headers) -> bool:
        return _bearer_token_matches(headers.get("Authorization"), self.api_tokens)

    def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("method") == "OPTIONS":
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        if url_path.startswith(GUARDED_PREFIX):
            headers = Headers(scope=scope)
            if not self.verify_token(headers):
                response = JSONResponse(
                    content={"error": "Unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                return response(scope, receive, send)
        return self.app(scope, receive, send)


def _sequence_end(row, eos_token_id, pad_token_id):
    """Return ``(token_count, finish_reason)`` for one generated row.

    Counting runs to the first EOS, inclusive, and stops there. Masking every
    ``pad_token_id`` occurrence would also delete the EOS whenever a tokenizer
    reuses one id for both (Qwen and Mistral do), which both under-counts the
    sequence and turns a real ``stop`` into ``length``. Anything after the EOS
    is padding, including a second EOS written as the pad value.
    """
    ids = [int(token) for token in row.tolist()]
    if eos_token_id is not None and eos_token_id in ids:
        return ids.index(eos_token_id) + 1, "stop"
    if pad_token_id is not None:
        # No EOS: a row that only ends in padding was cut short by the budget.
        while ids and ids[-1] == pad_token_id:
            ids.pop()
    return len(ids), "length"


def _coerce_generation_results(outputs) -> list:
    """Normalise whatever a runtime adapter returned into ``GenerationResult``s.

    A bare string is one result (iterating it would emit one choice per
    character). Mappings and objects are never iterated: only an explicit
    ``{"text": ...}`` mapping or a ``.texts`` sequence is understood, and
    anything else raises instead of silently producing `str()` of an object.
    """
    from collections.abc import Mapping as _Mapping

    if isinstance(outputs, GenerationResult):
        return [outputs]
    if isinstance(outputs, str):
        return [GenerationResult(outputs)]
    if isinstance(outputs, _Mapping):
        if isinstance(outputs.get("text"), str):
            return [
                GenerationResult(
                    outputs["text"],
                    completion_tokens=outputs.get("completion_tokens"),
                    finish_reason=outputs.get("finish_reason"),
                )
            ]
        raise TypeError(
            "Runtime adapter returned a mapping without a `text` string; "
            f"got keys {sorted(map(str, outputs))}."
        )
    texts = getattr(outputs, "texts", None)
    if isinstance(texts, (list, tuple)):
        outputs = texts
    if isinstance(outputs, (list, tuple)):
        results = []
        for item in outputs:
            if isinstance(item, GenerationResult):
                results.append(item)
            elif isinstance(item, str):
                results.append(GenerationResult(item))
            elif isinstance(item, _Mapping) and isinstance(item.get("text"), str):
                results.append(
                    GenerationResult(
                        item["text"],
                        completion_tokens=item.get("completion_tokens"),
                        finish_reason=item.get("finish_reason"),
                    )
                )
            else:
                raise TypeError(
                    "Runtime adapter result entries must be strings or describe a sequence; "
                    f"got {type(item).__name__}."
                )
        return results
    raise TypeError(
        "Runtime adapter results must be text, a sequence of text, or a value exposing "
        f"`texts`; got {type(outputs).__name__}."
    )


class OpenAiServer:
    def __init__(self, model, api_key: str = None):
        self.uvicorn_server = None
        self.app = FastAPI()
        self.model = model
        self.tokenizer = model.tokenizer
        self.model_id_or_path = model.config.name_or_path
        self.api_keys = _resolve_api_keys(api_key)
        self.api_tokens = _token_digests(self.api_keys)
        # One generation at a time: the endpoint runs in the threadpool, and a
        # single device cannot serve two concurrent generations.
        self._generation_lock = threading.Lock()
        if self.api_keys:
            self.app.add_middleware(AuthenticationMiddleware, tokens=self.api_keys)
        self.setup_routes()

    def _token_matches(self, authorization: str) -> bool:
        return _bearer_token_matches(authorization, self.api_tokens)

    @staticmethod
    def _is_loopback(request: Request) -> bool:
        client = request.client
        if client is None or not client.host:
            return False
        try:
            return ipaddress.ip_address(client.host).is_loopback
        except ValueError:
            return False

    def setup_routes(self):
        class OpenAiRequest(BaseModel):
            model: str = ""
            messages: list = []
            max_tokens: int = 256
            temperature: float = 0.0
            top_p: float = 1.0
            n: int = 1
            stop: list | str | None = None

        class OpenAiResponseMessage(BaseModel):
            role: str = "assistant"
            content: str

        class OpenAiResponseChoice(BaseModel):
            message: OpenAiResponseMessage
            index: int = 0
            finish_reason: str | None = None

        class OpenAiUsage(BaseModel):
            prompt_tokens: int = 0
            # `None` means "not reported by the backend". A fabricated 0 would
            # understate usage for every text-returning runtime adapter.
            completion_tokens: int | None = None
            total_tokens: int | None = None

        class OpenAiResponse(BaseModel):
            id: str = ""
            object: str = "chat.completion"
            created: int
            model: str
            choices: list[OpenAiResponseChoice]
            usage: OpenAiUsage | None = None

        def _stop_list(stop) -> list:
            if stop is None:
                return []
            if isinstance(stop, str):
                # An empty scalar must be ignored exactly like an empty list
                # entry: it matches at position zero and would drop the answer.
                return [stop] if stop else []
            return [value for value in stop if isinstance(value, str) and value]

        def _strip_stop(text: str, stop_strings: list):
            """Trim at the earliest stop sequence and report whether one was hit."""
            earliest = None
            for stop_string in stop_strings:
                index = text.find(stop_string)
                if index != -1 and (earliest is None or index < earliest):
                    earliest = index
            if earliest is None:
                return text, False
            return text[:earliest], True

        @self.app.post("/v1/chat/completions", response_model=OpenAiResponse)
        def create_completion(request: OpenAiRequest):
            # A plain `def` endpoint runs in the threadpool, so a long generation
            # cannot stall the event loop and starve `/health` or `/shutdown`.
            # `_generation_lock` keeps the model itself single-entry: one GPU
            # cannot run two generations concurrently, and letting the threadpool
            # call into it in parallel would raise VRAM use until it OOMs.
            # Authentication is enforced by AuthenticationMiddleware for /v1 paths.
            if not request.messages:
                raise HTTPException(status_code=400, detail="`messages` must not be empty.")

            try:
                model_inputs = self.tokenizer.apply_chat_template(
                    request.messages,
                    add_generation_prompt=True,
                    return_tensors='pt',
                )

                if isinstance(model_inputs, torch.Tensor):
                    model_inputs = model_inputs.to(self.model.device)
                    generate_inputs = {"inputs": model_inputs}
                    prompt_length = model_inputs.size(-1)
                else:
                    model_inputs = model_inputs.to(self.model.device)
                    input_ids = model_inputs["input_ids"]
                    generate_inputs = dict(model_inputs)
                    prompt_length = input_ids.size(-1)

                stop_strings = _stop_list(request.stop)
                do_sample = request.temperature != 0.0

                generate_kwargs = {
                    "max_new_tokens": request.max_tokens,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "do_sample": do_sample,
                }
                if do_sample:
                    generate_kwargs["temperature"] = request.temperature
                    generate_kwargs["top_p"] = request.top_p
                if request.n != 1:
                    if not do_sample:
                        # Transformers refuses multiple greedy sequences; reject the
                        # request up front instead of surfacing it as a server error.
                        raise HTTPException(
                            status_code=400,
                            detail="`n` must be 1 when `temperature` is 0 (greedy decoding). "
                            "Set `temperature` > 0 to request multiple sequences.",
                        )
                    generate_kwargs["num_return_sequences"] = request.n

                if stop_strings:
                    if getattr(self.model, "_runtime_generate", None) is None:
                        # Transformers 5.x raises unless the tokenizer is passed alongside `stop_strings`.
                        generate_kwargs["stop_strings"] = stop_strings
                        generate_kwargs["tokenizer"] = self.tokenizer
                    else:
                        # External runtime adapters accept the OpenAI-style `stop` argument.
                        generate_kwargs["stop"] = stop_strings

                with self._generation_lock, torch.inference_mode():
                    outputs = self.model.generate(**generate_kwargs, **generate_inputs)

                pad_token_id = self.tokenizer.pad_token_id
                eos_token_id = self.tokenizer.eos_token_id

                if torch.is_tensor(outputs):
                    generated_ids = outputs[:, prompt_length:]
                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    # Count each sequence separately: usage aggregates them, while a
                    # single row length is only the max across the batch.
                    token_counts = []
                    finish_reasons = []
                    for row in generated_ids:
                        count, reason = _sequence_end(row, eos_token_id, pad_token_id)
                        token_counts.append(count)
                        finish_reasons.append(reason)
                else:
                    results = _coerce_generation_results(outputs)
                    generated_texts = [result.text for result in results]
                    # An adapter that reports its own counts keeps them; one that
                    # returns only text yields `None`, which is reported as unknown
                    # rather than as a fabricated zero.
                    token_counts = [result.completion_tokens for result in results]
                    finish_reasons = [result.finish_reason for result in results]

                # Unknown counts stay unknown instead of being summed as zero.
                known_counts = [count for count in token_counts if count is not None]
                completion_tokens = sum(known_counts) if known_counts else None

                trimmed = [_strip_stop(text, stop_strings) for text in generated_texts]
                generated_texts = [text for text, _ in trimmed]
                # A sequence cut by a stop string ends on that stop, not on the budget.
                finish_reasons = [
                    "stop" if matched else reason for reason, (_, matched) in zip(finish_reasons, trimmed)
                ]

                choices = [
                    OpenAiResponseChoice(
                        message=OpenAiResponseMessage(content=gen_text),
                        index=i,
                        finish_reason=finish_reasons[i],
                    )
                    for i, gen_text in enumerate(generated_texts)
                ]

                usage = OpenAiUsage(prompt_tokens=prompt_length)
                if completion_tokens is not None:
                    usage.completion_tokens = completion_tokens
                    usage.total_tokens = prompt_length + completion_tokens

                return OpenAiResponse(
                    id=f"{uuid.uuid4()}",
                    created=int(time.time()),
                    model=self.model_id_or_path,
                    choices=choices,
                    usage=usage,
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/")
        def read_root():
            return {"message": "GPT-QModel OpenAI Compatible Server is running."}

        @self.app.get("/health")
        def health():
            # Unauthenticated liveness probe; deliberately outside GUARDED_PREFIX.
            return {"status": "ok"}

        @self.app.get("/shutdown")
        def shutdown(request: Request, authorization: str | None = Header(default=None)):
            # Stopping the service is privileged. When a key is configured it is
            # the only accepted credential: a loopback peer is not authorization,
            # because any page open on the host can issue a cross-site GET here.
            # Loopback is only sufficient while the server has no key at all.
            if self.api_keys:
                allowed = self._token_matches(authorization)
                detail = "`/shutdown` requires a valid API key."
            else:
                allowed = self._is_loopback(request)
                detail = "`/shutdown` requires a loopback caller when no API key is configured."
            if not allowed:
                raise HTTPException(status_code=403, detail=detail)
            self.shutdown()
            return {"message": "Server is shutting down..."}

    @staticmethod
    def _warn_if_exposed(host: str, api_keys: list) -> None:
        """Warn when an unauthenticated endpoint is reachable from other machines.

        The bind address is the caller's deployment decision, so this never
        refuses to start; it only makes the consequence explicit.
        """
        if api_keys:
            return
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            # A hostname such as "localhost" is loopback; anything else is effectively exposed.
            loopback = host in {"localhost", "localhost.localdomain"}
        if loopback:
            return
        warnings.warn(
            f"GPT-QModel OpenAI server is listening on {host} without an API key, so any "
            "machine that can reach that address can run inference on this model and stop "
            "the server. Pass `api_key=` (or set GPTQMODEL_OPENAI_API_KEY) before exposing "
            "it beyond localhost, or bind to 127.0.0.1 and use an SSH tunnel.",
            UserWarning,
            stacklevel=3,
        )

    def start(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, async_mode: bool = True):
        self._warn_if_exposed(host, self.api_keys)
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        self.uvicorn_server = uvicorn.Server(config)

        def run_server():
            self.uvicorn_server.run()

        if async_mode:
            thread = threading.Thread(target=run_server, daemon=False)
            thread.start()
            print(f"GPT-QModel OpenAi Server has started asynchronously at http://{host}:{port}.")
        else:
            run_server()
            print(f"GPT-QModel OpenAi Server has started synchronously at http://{host}:{port}.")

    def shutdown(self):
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True
            print("GPT-QModel OpenAi Server is shutting down...")

    def wait_until_ready(self, timeout: int = 30, check_interval: float = 0.1):
        if self.uvicorn_server is None:
            raise RuntimeError("GPT-QModel OpenAi server has not been started yet.")
        start_time = time.time()
        while not self.uvicorn_server.started:
            if time.time() - start_time > timeout:
                raise TimeoutError("GPT-QModel OpenAi server failed to start within the specified time.")
            time.sleep(check_interval)
