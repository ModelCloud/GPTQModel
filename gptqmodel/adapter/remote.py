import os
from urllib.parse import urlparse

from ..utils.logger import setup_logger

log = setup_logger()

def parse_url(url: str):
    parsed_url = urlparse(url)

    if parsed_url.netloc.endswith("huggingface.co") or parsed_url.netloc.endswith("hf.co"):
        parts = parsed_url.path.strip("/").split("/")

        if "blob" in parts:
            idx = parts.index("blob")
            repo_id = "/".join(parts[:idx])
            rev = parts[idx + 1]
            filename = parts[idx + 2].split("?")[0]  # remove ?download=true
            return [repo_id, rev, filename]
    else:
        return [url]
    return []

def resolve_path(path: str, filename: str) -> str: # return a valid file path to read
    if os.path.isdir(path):
        resolved_path = f"{path.removesuffix('/')}/{filename}"
        log.info(f"Resolver: Local path: `{resolved_path}`")
        if not os.path.isfile(resolved_path):
            raise ValueError(f"Resolver: Cannot find file in path: `{resolved_path}`")

        return resolved_path
    elif path.startswith("http"):
        from huggingface_hub import hf_hub_download

        result = parse_url(path)
        if len(result) == 3:
            log.info(
                f"Resolver: Downloading file from HF repo: `{result[0]}` revision: `{result[1]}` file: `{result[2]}`")
            resolved_path = hf_hub_download(repo_id=result[0], revision=result[1], filename=result[2])
            return resolved_path
        else:
            raise ValueError(f"Resolver: We only support local file path or HF repo id; actual = path: `{path}`, filename = `{filename}`")
            # logger.info(f"Adapter: Downloading adapter weights from uri = `{self.path}`")
            # import requests
            # response = requests.get(self.path, stream=True)
            # lora_path = HF_ADAPTER_FILE_NAME
            # with open(lora_path, "wb") as f:
            #     for chunk in response.iter_content(chunk_size=8192):
            #         f.write(chunk)
    elif not path.startswith("/"):
        path = path.rstrip("/")
        subfolder = None

        # fix HF subfoler path like: sliuau/llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc
        if path.count("/") > 1:
            path_split = path.split("/")
            path = f"{path_split[0]}/{path_split[1]}"
            subfolder = "/".join(path_split[2:])

        from huggingface_hub import HfApi, hf_hub_download

        # _ = HfApi().list_repo_files(path)

        resolved_path = hf_hub_download(repo_id=path, filename=filename, subfolder=subfolder)
        return resolved_path
            # print(f"Adapter tensors loaded from `{self.path}`")
    else:
        raise ValueError(
            f"Resolver: We only support local file path or HF repo id; actual = path: `{path}`, filename = `{filename}`")