import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


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


def fetch_with_retry(url: str, *, timeout: float, retries: int, retry_delay: float) -> str:
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


def print_status(base_url: str) -> None:
    status = fetch_text(f"{base_url}/gpu/status", timeout=10, suppress_error=True).strip()
    if status:
        print(status)


def append_github_env(name: str, value: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--count", required=True)
    parser.add_argument("--sleep-sec", type=float, default=5)
    parser.add_argument("--timeout-sec", type=int, default=18000)
    parser.add_argument("--request-timeout", type=float, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--require-single", action="store_true")
    args = parser.parse_args()

    encoded_test = urllib.parse.quote(args.test, safe="")
    encoded_runner = urllib.parse.quote(args.runner, safe="")
    start_s = time.time()

    print("Requesting GPU from allocator")
    print(
        f"run_id={args.run_id} test={args.test} runner={args.runner} count={args.count}"
    )

    while True:
        ts_ms = now_ms()
        url = (
            f"{args.base_url}/gpu/get?runid={args.run_id}&timestamp={ts_ms}"
            f"&test={encoded_test}&runner={encoded_runner}&count={args.count}"
        )
        print(f"requesting GPU with: {url}")

        resp = fetch_with_retry(
            url,
            timeout=args.request_timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        ).replace("\r", "").replace("\n", "").strip()

        print(f"resp={{{resp}}}")

        if not is_valid_gpu_response(resp):
            print(f"Allocator returned invalid response: {resp!r} (temporary error)")
            print_status(args.base_url)
            time.sleep(args.sleep_sec)
            continue

        if resp.startswith("-") and "," not in resp:
            elapsed = int(time.time() - start_s)
            if elapsed >= args.timeout_sec:
                print(
                    f"Timed out after {args.timeout_sec}s waiting for GPU "
                    f"(last response={resp})"
                )
                print_status(args.base_url)
                return 1

            print(
                f"No GPU available (response={resp}). Waiting {args.sleep_sec}s..."
                f" elapsed={elapsed}s"
            )
            print_status(args.base_url)
            time.sleep(args.sleep_sec)
            continue

        if args.require_single and "," in resp:
            print(f"Allocator returned multiple GPUs for job requiring one GPU: {resp}")
            return 1

        print(f"Allocated GPU ID: {resp}")
        append_github_env("CUDA_VISIBLE_DEVICES", resp)
        append_github_env("STEP_TIMESTAMP", str(now_ms()))
        print(f"CUDA_VISIBLE_DEVICES set to {resp}")
        print_status(args.base_url)
        return 0


if __name__ == "__main__":
    sys.exit(main())
