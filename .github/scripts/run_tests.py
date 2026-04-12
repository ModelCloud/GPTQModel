import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def append_github_env(name: str, value: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def fetch_text(url: str, *, timeout: float, suppress_error: bool = False) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if suppress_error:
            print(f"Request failed for {url}: {exc}")
            return ""
        raise


def kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        # Process (or process group) is already gone; nothing to do.
        pass


def start_keepalive_monitor(
        *,
        proc: subprocess.Popen[str],
        keep_alive_url: str,
        interval_sec: int,
) -> tuple[threading.Thread, threading.Event, dict[str, int]]:
    stop_event = threading.Event()
    state = {"forced_exit_code": 0}

    def worker() -> None:
        print(f"start to keep alive... {keep_alive_url}")
        while not stop_event.wait(interval_sec):
            resp = fetch_text(keep_alive_url, timeout=10, suppress_error=True)
            # if resp.strip() == "-1":
            if int(resp.strip()) < 0:
                print(f"Server returned {resp.strip()}, terminating job...")
                state["forced_exit_code"] = 3
                kill_process_group(proc)
                stop_event.set()
                return
            print("gpu is kept alive...")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, stop_event, state


def stream_process_output(proc: subprocess.Popen[str], log_file: Path) -> int:
    assert proc.stdout is not None
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as fh:
        for line in proc.stdout:
            print(line, end="")
            fh.write(line)
    return proc.wait()


def maybe_uninstall_vllm() -> None:
    uninstall_cmd = ["uv", "pip", "uninstall", "vllm", "-y"]
    print(f"+ {' '.join(uninstall_cmd)}")
    subprocess.run(uninstall_cmd, check=False)

    list_cmd = ["uv", "pip", "list"]
    print(f"+ {' '.join(list_cmd)}")
    subprocess.run(list_cmd, check=False)


def log_vram(base_url: str, run_id: str, gpu_id: str, execution_time: int, test: str) -> None:
    encoded_test = urllib.parse.quote(test, safe="")
    url = (
        f"{base_url}/gpu/logVram?runid={run_id}&gpu={gpu_id}"
        f"&range={execution_time}&unit=second&test={encoded_test}"
    )
    try:
        print(fetch_text(url, timeout=30, suppress_error=True))
    except Exception:
        # Logging VRAM usage is best-effort; failures must not affect the main test flow.
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test-script", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--model-test-mode")
    parser.add_argument("--clear-cuda", action="store_true")
    parser.add_argument("--xpu-mode", action="store_true")
    parser.add_argument("--monitor-interval-sec", type=int, default=60)
    args = parser.parse_args()

    env = os.environ.copy()
    if args.clear_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
        print("CUDA_VISIBLE_DEVICES=")

    if args.xpu_mode:
        maybe_uninstall_vllm()

    if args.model_test_mode is not None:
        env["GPTQMODEL_MODEL_TEST_MODE"] = args.model_test_mode
        print(f"GPTQMODEL_MODEL_TEST_MODE={args.model_test_mode}")

    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")

    log_dir = Path(f"/opt/dist/GPTQModel/{args.run_id}/logs")
    log_file = log_dir / f"{args.test_script}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    pytest_cmd = ["pytest", "--durations=0", f"tests/{args.test_script}.py"]
    print(f"+ {' '.join(pytest_cmd)}")

    proc = subprocess.Popen(
        pytest_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )

    encoded_test = urllib.parse.quote(args.test_script, safe="")
    encoded_runner = urllib.parse.quote(args.runner, safe="")
    keep_alive_url = (
        f"{args.base_url}/gpu/keepalive?runid={args.run_id}&test={encoded_test}"
        f"&runner={encoded_runner}&timestamp={int(time.time())}&gpu={env.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    monitor_thread = None
    monitor_stop = None
    monitor_state = {"forced_exit_code": 0}
    if env.get("CUDA_VISIBLE_DEVICES", ""):
        monitor_thread, monitor_stop, monitor_state = start_keepalive_monitor(
            proc=proc,
            keep_alive_url=keep_alive_url,
            interval_sec=args.monitor_interval_sec,
        )

    start_time = time.time()
    try:
        return_code = stream_process_output(proc, log_file)
    finally:
        if monitor_stop is not None:
            print("trap cleanup EXIT...")
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)

    if monitor_state["forced_exit_code"]:
        append_github_env("ERROR", "22")
        return 22

    if return_code != 0:
        append_github_env("ERROR", "22")
        print(f"pipe status wrong: {return_code}")
        return 22

    execution_time = int(time.time() - start_time)
    print(f"{execution_time // 60}m {execution_time % 60}s")

    try:
        for entry in sorted(log_dir.iterdir()):
            stat = entry.stat()
            size = stat.st_size
            print(f"{size:>10} {entry.name}")
    except OSError as exc:
        print(f"Failed to list log dir: {exc}")

    gpu_id = args.gpu_id or env.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_id:
        log_vram(args.base_url, args.run_id, gpu_id, execution_time, args.test_script)

    return 0


if __name__ == "__main__":
    sys.exit(main())
