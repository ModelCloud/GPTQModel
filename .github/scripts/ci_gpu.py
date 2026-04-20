import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ci_common import append_github_env


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def build_server_info() -> dict[str, str]:
    from device_smi import Device
    os_info = Device("os")
    cpu_model = Device("cpu").model
    platform_name = (
            os.environ.get("GPU_PLATFORM")
            or cpu_model
    )
    return {
        "platform": platform_name,
        "arch": os_info.arch,
        "system": os_info.name,
    }


def request_json(
        url: str,
        *,
        method: str = "GET",
        body: dict[str, object] | None = None,
        timeout: float,
) -> object | None:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        if not raw.strip():
            return None
        return json.loads(raw)


def request_json_with_retry(
        url: str,
        *,
        method: str,
        body: dict[str, object] | None,
        timeout: float,
        retries: int,
        retry_delay: float,
) -> object | None:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return request_json(url, method=method, body=body, timeout=timeout)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)
    if last_error is not None:
        print(f"Request failed after retries: {last_error}")
    return None


def extract_gpu_ids(response: object | None) -> str:
    if not isinstance(response, dict):
        return ""
    gpu_ids = response.get("gpuIds")
    return gpu_ids.strip() if isinstance(gpu_ids, str) else ""


def is_valid_gpu_response(value: str) -> bool:
    if not value:
        return False
    for part in value.split(","):
        if not part:
            return False
        if part.startswith("-"):
            if not part[1:].isdigit():
                return False
        elif not part.isdigit():
            return False
    return True


def query_gpu_inventory() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,memory.total,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    gpus: list[dict[str, object]] = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 13:
            raise ValueError(f"Unexpected nvidia-smi output line: {line}")
        gpus.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "util": int(fields[2]),
                "memUsed": int(fields[3]),
                "memFree": int(fields[4]),
                "memTotal": int(fields[5]),
                "driver": fields[6],
                "name": fields[7],
                "serial": fields[8],
                "displayActive": fields[9].lower() == "enabled",
                "displayMode": fields[10].lower() == "enabled",
                "temperature": int(fields[11]),
                "sm": fields[12],
            }
        )
    return gpus


def build_get_request(
        *,
        runner_name: str,
        run_id: str,
        test_name: str,
        count: str,
        sm: str = "0",
) -> dict[str, object]:
    job: dict[str, object] = {
        "jobId": int(run_id),
        "count": int(count),
        "test": test_name,
        "exclusive": True,
        "timestamp": now_ms(),
    }
    sm_value = sm.strip()
    if sm_value and sm_value != "0":
        job["sm"] = sm_value

    return {
        "server": build_server_info(),
        "job": job,
        "gpu": query_gpu_inventory(),
    }


def build_job_request(*, runner_name: str, run_id: str, test_name: str) -> dict[str, object]:
    return {
        "server": build_server_info(),
        "job": {
            "jobId": int(run_id),
            "test": test_name,
        },
    }


def format_info_url(base_url: str, platform_name: str) -> str:
    query = urllib.parse.urlencode({"platform": platform_name, "plain": "true"})
    return f"{normalize_base_url(base_url)}/info?{query}"


def print_status(base_url: str, runner_name: str) -> None:
    server = build_server_info()
    try:
        status = request_json(
            format_info_url(base_url, server["platform"]),
            timeout=10,
        )
    except Exception as exc:
        print(f"Request failed for allocator info: {exc}")
        return
    if status is not None:
        print(status)


def allocate_gpu(args: argparse.Namespace) -> int:
    start_s = time.time()
    endpoint = f"{normalize_base_url(args.base_url)}/get"
    sm_value = args.sm.strip()

    print("Requesting GPU from allocator")
    print(
        f"run_id={args.run_id} test={args.test} runner={args.runner} count={args.count} "
        f"sm={(sm_value if sm_value and sm_value != '0' else '<not-sent>')}"
    )

    while True:
        request_body = build_get_request(
            runner_name=args.runner,
            run_id=args.run_id,
            test_name=args.test,
            count=args.count,
            sm=args.sm,
        )
        print(f"requesting GPU with: {endpoint}")
        print(f"request job payload: {request_body['job']}")
        print(
            "request gpu payload: "
            f"{[{key: gpu.get(key) for key in ('index', 'name', 'sm')} for gpu in request_body['gpu']]}"
        )

        response = request_json_with_retry(
            endpoint,
            method="POST",
            body=request_body,
            timeout=args.request_timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        resp = extract_gpu_ids(response)

        print(f"resp={{{resp}}}")

        if not is_valid_gpu_response(resp):
            print(f"Allocator returned invalid response: {resp!r} (temporary error)")
            print_status(args.base_url, args.runner)
            time.sleep(args.sleep_sec)
            continue

        if resp.startswith("-") and "," not in resp:
            elapsed = int(time.time() - start_s)
            if elapsed >= args.timeout_sec:
                print(
                    f"Timed out after {args.timeout_sec}s waiting for GPU "
                    f"(last response={resp})"
                )
                print_status(args.base_url, args.runner)
                return 1

            print(
                f"No GPU available (response={resp}). Waiting {args.sleep_sec}s..."
                f" elapsed={elapsed}s"
            )
            print_status(args.base_url, args.runner)
            time.sleep(args.sleep_sec)
            continue

        if args.require_single and "," in resp:
            print(f"Allocator returned multiple GPUs for job requiring one GPU: {resp}")
            return 1

        print(f"Allocated GPU ID: {resp}")
        append_github_env("CUDA_VISIBLE_DEVICES", resp)
        print(f"CUDA_VISIBLE_DEVICES set to {resp}")
        print(subprocess.getoutput(f"nvidia-smi -i {resp} --query-gpu=name --format=csv"))
        print_status(args.base_url, args.runner)
        return 0


def release_gpu(args: argparse.Namespace) -> int:
    request_body = build_job_request(
        runner_name=args.runner,
        run_id=args.run_id,
        test_name=args.test,
    )
    url = f"{normalize_base_url(args.base_url)}/release"
    print(url)

    try:
        response = request_json(url, method="POST", body=request_body, timeout=args.timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"Failed to release GPU: {exc}")
        return 0

    resp = extract_gpu_ids(response)
    print(f"response: {resp}")
    if args.gpu_id and resp not in {args.gpu_id, "-1"}:
        print(f"Error: response ({resp}) != expected ({args.gpu_id})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    allocate_parser = subparsers.add_parser("allocate")
    allocate_parser.add_argument("--base-url", required=True)
    allocate_parser.add_argument("--run-id", required=True)
    allocate_parser.add_argument("--test", required=True)
    allocate_parser.add_argument("--runner", required=True)
    allocate_parser.add_argument("--count", required=True)
    allocate_parser.add_argument("--sm", default="0")
    allocate_parser.add_argument("--sleep-sec", type=float, default=5)
    allocate_parser.add_argument("--timeout-sec", type=int, default=18000)
    allocate_parser.add_argument("--request-timeout", type=float, default=10)
    allocate_parser.add_argument("--retries", type=int, default=3)
    allocate_parser.add_argument("--retry-delay", type=float, default=1)
    allocate_parser.add_argument("--require-single", action="store_true")

    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--base-url", required=True)
    release_parser.add_argument("--run-id", required=True)
    release_parser.add_argument("--gpu-id", default="")
    release_parser.add_argument("--test", required=True)
    release_parser.add_argument("--runner", required=True)
    release_parser.add_argument("--timeout", type=float, default=10)

    args = parser.parse_args()
    if args.command == "allocate":
        return allocate_gpu(args)
    if args.command == "release":
        return release_gpu(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
