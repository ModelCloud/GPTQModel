import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ci_common import append_github_env
from ci_gpu import (
    build_job_request,
    extract_gpu_ids,
    normalize_base_url,
    request_json,
)

ERROR_PATTERN = re.compile(
    r"nvcc fatal|error:|fatal error|ModuleNotFoundError|ImportError|AssertionError|Exception|is the correct path|No such file or directory|Repo id must be in"
)


def kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        # Process group may already have exited; nothing to kill.
        return


def start_keepalive_monitor(
        *,
        proc: subprocess.Popen[str],
        keepalive_endpoint: str,
        keepalive_payload: dict[str, object],
        expected_gpu_ids: str,
        interval_sec: int,
) -> tuple[threading.Thread, threading.Event, dict[str, int]]:
    stop_event = threading.Event()
    state = {"forced_exit_code": 0}

    def worker() -> None:
        print(f"start to keep alive... {keepalive_endpoint}")
        while not stop_event.wait(interval_sec):
            response = None
            retry_started_at = time.monotonic()
            while not stop_event.is_set():
                try:
                    response = request_json(
                        keepalive_endpoint,
                        method="POST",
                        body=keepalive_payload,
                        timeout=10,
                    )
                    break
                except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                    if time.monotonic() - retry_started_at >= 130:  # max 2 minutes
                        break
                    print(f"Keepalive request failed: {exc}")
                    if stop_event.wait(3):
                        return
                    continue

            if response is None:
                resp = None
            else:
                resp = extract_gpu_ids(response)
            if response is None or resp == "-1":
                print(f"Server returned {resp}, terminating job...")
                state["forced_exit_code"] = 3
                kill_process_group(proc)
                stop_event.set()
                return
            if expected_gpu_ids and resp != expected_gpu_ids:
                print(f"Keepalive returned mismatched GPUs {resp}, expected {expected_gpu_ids}.")
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
    for cmd in (["uv", "pip", "uninstall", "vllm", "-y"], ["uv", "pip", "list"]):
        print(f"+ {' '.join(cmd)}")
        subprocess.run(cmd, check=False)


def run_tests(args: argparse.Namespace) -> int:
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

    keepalive_endpoint = f"{normalize_base_url(args.base_url)}/keepalive"
    keepalive_payload = build_job_request(
        runner_name=args.runner,
        run_id=args.run_id,
        test_name=args.test_script,
    )

    monitor_thread = None
    monitor_stop = None
    monitor_state = {"forced_exit_code": 0}
    if env.get("CUDA_VISIBLE_DEVICES", ""):
        monitor_thread, monitor_stop, monitor_state = start_keepalive_monitor(
            proc=proc,
            keepalive_endpoint=keepalive_endpoint,
            keepalive_payload=keepalive_payload,
            expected_gpu_ids=env.get("CUDA_VISIBLE_DEVICES", ""),
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
            print(f"{stat.st_size:>10} {entry.name}")
    except OSError as exc:
        print(f"Failed to list log dir: {exc}")

    return 0


def check_log(args: argparse.Namespace) -> int:
    log_file = Path(args.log_root) / args.run_id / "logs" / f"{args.test_script}.log"
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return 1

    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"Failed to read log file {log_file}: {exc}")
        return 1

    matched = 0
    for lineno, line in enumerate(lines, start=1):
        if ERROR_PATTERN.search(line):
            print(f"{lineno}:{line}")
            matched += 1
            if matched >= args.max_matches:
                break

    tail = lines[-args.tail_lines:]
    if tail:
        print(f"--- tail -n {args.tail_lines} {log_file}")
        for line in tail:
            print(line)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--test-script", required=True)
    run_parser.add_argument("--runner", required=True)
    run_parser.add_argument("--gpu-id", default="")
    run_parser.add_argument("--model-test-mode")
    run_parser.add_argument("--clear-cuda", action="store_true")
    run_parser.add_argument("--xpu-mode", action="store_true")
    run_parser.add_argument("--monitor-interval-sec", type=int, default=60)

    check_parser = subparsers.add_parser("check-log")
    check_parser.add_argument("--run-id", required=True)
    check_parser.add_argument("--test-script", required=True)
    check_parser.add_argument("--log-root", default="/opt/dist/GPTQModel")
    check_parser.add_argument("--tail-lines", type=int, default=200)
    check_parser.add_argument("--max-matches", type=int, default=50)

    args = parser.parse_args()
    if args.command == "run":
        return run_tests(args)
    if args.command == "check-log":
        return check_log(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
