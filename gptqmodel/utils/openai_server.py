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
    """Resolve configured keys: explicit keys win, the environment is a fallback."""
    if isinstance(api_key, str):
        keys = [api_key]
    else:
        keys = list(api_key or [])
    if not keys:
        env_key = os.environ.get(API_KEY_ENV)
        keys = [env_key] if env_key else []
    # Blank entries must not silently disable authentication.
    return [key for key in keys if key]


def _token_hashes(keys: list) -> list:
    # Hashing first means the comparison never leaks the configured key length.
    return [hashlib.sha256(key.encode("utf-8")).digest() for key in keys]


def _bearer_token_matches(authorization: str, token_hashes: list) -> bool:
    if not authorization:
        return False
    scheme, _, param = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return False
    param_hash = hashlib.sha256(param.encode("utf-8")).digest()
    matched = False
    for token_hash in token_hashes:
        # `|=` instead of `break` keeps the work constant regardless of which key matches.
        matched |= secrets.compare_digest(param_hash, token_hash)
    return matched


class AuthenticationMiddleware:
    """Pure ASGI bearer-token authentication.

    Skipped in two cases:
        1. The HTTP method is OPTIONS (CORS preflight).
        2. The request path does not start with one of ``GUARDED_PREFIX``.
    """

    def __init__(self, app, tokens: list) -> None:
        self.app = app
        self.token_hashes = _token_hashes(tokens)

    def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("method") == "OPTIONS":
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        if url_path.startswith(GUARDED_PREFIX):
            headers = Headers(scope=scope)
            if not _bearer_token_matches(headers.get("Authorization"), self.token_hashes):
                response = JSONResponse(
                    content={"error": "Unauthorized"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                return response(scope, receive, send)
        return self.app(scope, receive, send)


class OpenAiServer:
    def __init__(self, model, api_key: str = None):
        self.uvicorn_server = None
        self.app = FastAPI()
        self.model = model
        self.tokenizer = model.tokenizer
        self.model_id_or_path = model.config.name_or_path
        self.api_keys = _resolve_api_keys(api_key)
        self.token_hashes = _token_hashes(self.api_keys)
        if self.api_keys:
            self.app.add_middleware(AuthenticationMiddleware, tokens=self.api_keys)
        self.setup_routes()

    def _token_matches(self, authorization: str) -> bool:
        return _bearer_token_matches(authorization, self.token_hashes)

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
            completion_tokens: int = 0
            total_tokens: int = 0

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
                return [stop]
            return [value for value in stop if isinstance(value, str) and value]

        def _strip_stop(text: str, stop_strings: list) -> str:
            # OpenAI returns the text without the stop sequence that ended generation.
            for stop_string in stop_strings:
                index = text.find(stop_string)
                if index != -1:
                    text = text[:index]
            return text

        @self.app.post("/v1/chat/completions", response_model=OpenAiResponse)
        async def create_completion(request: OpenAiRequest):
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
                    generate_kwargs["num_return_sequences"] = request.n

                if stop_strings:
                    if getattr(self.model, "_runtime_generate", None) is None:
                        # Transformers 5.x raises unless the tokenizer is passed alongside `stop_strings`.
                        generate_kwargs["stop_strings"] = stop_strings
                        generate_kwargs["tokenizer"] = self.tokenizer
                    else:
                        # External runtime adapters accept the OpenAI-style `stop` argument.
                        generate_kwargs["stop"] = stop_strings

                with torch.inference_mode():
                    outputs = self.model.generate(**generate_kwargs, **generate_inputs)

                if torch.is_tensor(outputs):
                    generated_texts = self.tokenizer.batch_decode(
                        outputs[:, prompt_length:],
                        skip_special_tokens=True,
                    )
                    completion_tokens = max(int(outputs.shape[-1]) - prompt_length, 0)
                else:
                    generated_texts = [str(text) for text in outputs]
                    completion_tokens = 0

                generated_texts = [_strip_stop(text, stop_strings) for text in generated_texts]

                choices = [
                    OpenAiResponseChoice(
                        message=OpenAiResponseMessage(content=gen_text),
                        index=i,
                        finish_reason="stop",
                    )
                    for i, gen_text in enumerate(generated_texts)
                ]

                return OpenAiResponse(
                    id=f"{uuid.uuid4()}",
                    created=int(time.time()),
                    model=self.model_id_or_path,
                    choices=choices,
                    usage=OpenAiUsage(
                        prompt_tokens=prompt_length,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_length + completion_tokens,
                    ),
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
            # Stopping the service is privileged: loopback callers, or a valid API key.
            if not self._is_loopback(request) and not self._token_matches(authorization):
                raise HTTPException(status_code=403, detail="`/shutdown` requires localhost or an API key.")
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
