import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from ci_common import append_github_env, fetch_text

LOG_ERROR_PATTERN = re.compile(
    r"nvcc fatal|error:|fatal error|ModuleNotFoundError|ImportError|AssertionError|"
    r"Exception|is the correct path|No such file or directory|Repo id must be in"
)


def kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
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
            response = fetch_text(keep_alive_url, timeout=10, suppress_error=True)
            if int(response.strip()) < 0:
                print(f"Server returned {response.strip()}, terminating job...")
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
    with log_file.open("w", encoding="utf-8") as handle:
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
    return proc.wait()


def maybe_uninstall_vllm() -> None:
    uninstall_command = ["uv", "pip", "uninstall", "vllm", "-y"]
    print(f"+ {' '.join(uninstall_command)}")
    subprocess.run(uninstall_command, check=False)

    list_command = ["uv", "pip", "list"]
    print(f"+ {' '.join(list_command)}")
    subprocess.run(list_command, check=False)


def log_vram(base_url: str, run_id: str, gpu_id: str, execution_time: int, test_name: str) -> None:
    encoded_test = urllib.parse.quote(test_name, safe="")
    url = (
        f"{base_url}/gpu/logVram?runid={run_id}&gpu={gpu_id}"
        f"&range={execution_time}&unit=second&test={encoded_test}"
    )
    try:
        print(fetch_text(url, timeout=30, suppress_error=True))
    except Exception:
        pass


def command_run(args: argparse.Namespace) -> int:
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

    pytest_command = ["pytest", "--durations=0", f"tests/{args.test_script}.py"]
    print(f"+ {' '.join(pytest_command)}")

    proc = subprocess.Popen(
        pytest_command,
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
            size = entry.stat().st_size
            print(f"{size:>10} {entry.name}")
    except OSError as exc:
        print(f"Failed to list log dir: {exc}")

    gpu_id = args.gpu_id or env.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_id:
        log_vram(args.base_url, args.run_id, gpu_id, execution_time, args.test_script)

    return 0


def command_check_log(args: argparse.Namespace) -> int:
    log_file = Path(f"/opt/dist/GPTQModel/{args.run_id}/logs/{args.test_script}.log")
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return 1

    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    matches = [
        f"{index}:{line}"
        for index, line in enumerate(lines, start=1)
        if LOG_ERROR_PATTERN.search(line)
    ]
    for line in matches[:50]:
        print(line)

    tail_lines = lines[-200:]
    for line in tail_lines:
        print(line)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--base-url", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--test-script", required=True)
    run.add_argument("--runner", required=True)
    run.add_argument("--gpu-id", default="")
    run.add_argument("--model-test-mode")
    run.add_argument("--clear-cuda", action="store_true")
    run.add_argument("--xpu-mode", action="store_true")
    run.add_argument("--monitor-interval-sec", type=int, default=60)
    run.set_defaults(handler=command_run)

    check_log = subparsers.add_parser("check-log")
    check_log.add_argument("--run-id", required=True)
    check_log.add_argument("--test-script", required=True)
    check_log.set_defaults(handler=command_check_log)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
