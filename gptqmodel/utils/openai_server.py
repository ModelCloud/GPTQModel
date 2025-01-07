import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import time
import uuid

class OpenAIServer:
    def __init__(self, model):
        self.app = FastAPI()
        self.model = model
        self.tokenizer = model.tokenizer
        self.model_id_or_path = model.model.config.name_or_path
        self._setup_routes()

    def _setup_routes(self):
        class OpenAIRequest(BaseModel):
            model: str
            prompt: str
            max_tokens: int = 256
            temperature: float = 1.0
            top_p: float = 1.0
            n: int = 1
            stop: list = []

        class OpenAIResponseChoice(BaseModel):
            text: str
            index: int = 0

        class OpenAIResponse(BaseModel):
            id: str = ""
            object: str = "text_completion"
            created: int
            model: str
            choices: list[OpenAIResponseChoice]

        @self.app.post("/v1/completions", response_model=OpenAIResponse)
        async def create_completion(request: OpenAIRequest):
            try:
                input_ids = self.tokenizer.encode(request.prompt, return_tensors='pt')
                do_sample = True if request.temperature != 1.0 else False
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        num_return_sequences=request.n,
                        eos_token_id=self.tokenizer.eos_token_id,
                        stop_strings=request.stop,
                        do_sample=do_sample
                    )

                generated_texts = [
                    self.tokenizer.decode(output, skip_special_tokens=True)[len(request.prompt):]
                    for output in outputs
                ]

                choices = [
                    OpenAIResponseChoice(
                        text=gen_text,
                        index=i,
                    )
                    for i, gen_text in enumerate(generated_texts)
                ]

                response_id = f"gptqmodel-{uuid.uuid4()}"

                response = OpenAIResponse(
                    id=response_id,
                    created=int(time.time()),
                    model=self.model_id_or_path,
                    choices=choices
                )
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/")
        def read_root():
            return {"message": "OpenAI Compatible Server is running."}

    def start(self, host: str = "0.0.0.0", port: int = 80, async_mode: bool = True):
        if async_mode:
            thread = threading.Thread(
                target=uvicorn.run,
                args=(self.app,),
                kwargs={"host": host, "port": port, "log_level": "info"},
                daemon=True
            )
            thread.start()
            print(f"Server has started asynchronously at http://{host}:{port}.")
        else:
            uvicorn.run(self.app, host=host, port=port, log_level="info")
            print(f"Server is starting in synchronous mode at http://{host}:{port}.")

