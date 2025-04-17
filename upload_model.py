import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi
from logbar import LogBar

api = HfApi()

log = LogBar.shared()

MODEL_PATH="YOUR_QUANTIZED_MODEL_PATH" # /root/mymodel/4bit_gptq
HF_REPO_PATH="YOUR_FULL_HF_MODEL_REPO_PATH" # ModelCloud/NewModel

# upload
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=HF_REPO_PATH,
    repo_type="model",
)