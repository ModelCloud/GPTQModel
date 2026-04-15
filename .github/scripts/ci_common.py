import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def fetch_text(url: str, *, timeout: float, suppress_error: bool = False) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if suppress_error:
            print(f"Request failed for {url}: {exc}")
            return ""
        raise


def fetch_with_retry(
    url: str,
    *,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_text(url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)
    if last_error is not None:
        print(f"Request failed after retries: {last_error}")
    return ""


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(value: str | None, *, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def append_github_file_var(file_env_name: str, name: str, value: str) -> None:
    target = os.environ.get(file_env_name)
    if not target:
        return
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def append_github_env(name: str, value: str) -> None:
    append_github_file_var("GITHUB_ENV", name, value)


def append_github_output(name: str, value: str) -> None:
    append_github_file_var("GITHUB_OUTPUT", name, value)


def run_command(
    command: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> int:
    print(f"+ {' '.join(command)}")
    completed = subprocess.run(command, check=False, env=env)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode
