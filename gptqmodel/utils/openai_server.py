# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
import uuid

import torch

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ModuleNotFoundError as exception:
    raise type(exception)(
        "GPTQModel OpenAi serve required dependencies are not installed.",
        "Please install via `pip install gptqmodel[openai] --no-build-isolation`.",
    )

class OpenAiServer:
    def __init__(self, model):
        self.uvicorn_server = None
        self.app = FastAPI()
        self.model = model
        self.tokenizer = model.tokenizer
        self.model_id_or_path = model.config.name_or_path
        self.setup_routes()

    def setup_routes(self):
        class OpenAiRequest(BaseModel):
            model: str
            messages: list = []
            max_tokens: int = 256
            temperature: float = 0.0
            top_p: float = 1.0
            n: int = 1
            stop: list = None

        class OpenAiResponseChoice(BaseModel):
            text: str
            index: int = 0

        class OpenAiResponse(BaseModel):
            id: str = ""
            object: str = "text_completion"
            created: int
            model: str
            choices: list[OpenAiResponseChoice]

        @self.app.post("/v1/chat/completions", response_model=OpenAiResponse)
        async def create_completion(request: OpenAiRequest):
            try:
                inputs_tensor = self.tokenizer.apply_chat_template(
                    request.messages,
                    add_generation_prompt=True,
                    return_tensors='pt').to(self.model.device)

                do_sample = True if request.temperature != 0.0 else False
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs_tensor,
                        max_length=inputs_tensor.shape[0] + request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        num_return_sequences=request.n,
                        eos_token_id=self.tokenizer.eos_token_id,
                        stop_strings=request.stop,
                        do_sample=do_sample
                    )

                generated_texts = self.tokenizer.batch_decode(
                    outputs[:, inputs_tensor.size(-1):],
                    skip_special_tokens=True,
                )

                choices = [
                    OpenAiResponseChoice(
                        text=gen_text,
                        index=i,
                    )
                    for i, gen_text in enumerate(generated_texts)
                ]

                response = OpenAiResponse(
                    id=f"{uuid.uuid4()}",
                    created=int(time.time()),
                    model=self.model_id_or_path,
                    choices=choices
                )
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/")
        def read_root():
            return {"message": "GPTQModel OpenAI Compatible Server is running."}

        @self.app.get("/shutdown")
        def shutdown():
            self.shutdown()
            return {"message": "Server is shutting down..."}

    def start(self, host: str = "0.0.0.0", port: int = 80, async_mode: bool = True):
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        self.uvicorn_server = uvicorn.Server(config)

        def run_server():
            self.uvicorn_server.run()

        if async_mode:
            thread = threading.Thread(target=run_server, daemon=False)
            thread.start()
            print(f"GPTQModel OpenAi Server has started asynchronously at http://{host}:{port}.")
        else:
            run_server()
            print(f"GPTQModel OpenAi Server has started synchronously at http://{host}:{port}.")

    def shutdown(self):
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True
            print("GPTQModel OpenAi Server is shutting down...")

    def wait_until_ready(self, timeout: int = 30, check_interval: float = 0.1):
        start_time = time.time()
        while not self.uvicorn_server.started:
            if time.time() - start_time > timeout:
                raise TimeoutError("GPTQModel OpenAi server failed to start within the specified time.")
            time.sleep(check_interval)
