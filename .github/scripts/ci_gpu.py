import argparse
import sys
import time
import urllib.parse

from ci_common import append_github_env, fetch_text, fetch_with_retry, now_ms


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


def print_status(base_url: str) -> None:
    status = fetch_text(f"{base_url}/gpu/status", timeout=10, suppress_error=True).strip()
    if status:
        print(status)


def command_allocate(args: argparse.Namespace) -> int:
    encoded_test = urllib.parse.quote(args.test, safe="")
    encoded_runner = urllib.parse.quote(args.runner, safe="")
    start_seconds = time.time()

    print("Requesting GPU from allocator")
    print(
        f"run_id={args.run_id} test={args.test} runner={args.runner} count={args.count}"
    )

    while True:
        timestamp_ms = now_ms()
        url = (
            f"{args.base_url}/gpu/get?runid={args.run_id}&timestamp={timestamp_ms}"
            f"&test={encoded_test}&runner={encoded_runner}&count={args.count}"
        )
        print(f"requesting GPU with: {url}")

        response = fetch_with_retry(
            url,
            timeout=args.request_timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        ).replace("\r", "").replace("\n", "").strip()

        print(f"resp={{{response}}}")

        if not is_valid_gpu_response(response):
            print(f"Allocator returned invalid response: {response!r} (temporary error)")
            print_status(args.base_url)
            time.sleep(args.sleep_sec)
            continue

        if response.startswith("-") and "," not in response:
            elapsed = int(time.time() - start_seconds)
            if elapsed >= args.timeout_sec:
                print(
                    f"Timed out after {args.timeout_sec}s waiting for GPU "
                    f"(last response={response})"
                )
                print_status(args.base_url)
                return 1

            print(
                f"No GPU available (response={response}). Waiting {args.sleep_sec}s..."
                f" elapsed={elapsed}s"
            )
            print_status(args.base_url)
            time.sleep(args.sleep_sec)
            continue

        if args.require_single and "," in response:
            print(f"Allocator returned multiple GPUs for job requiring one GPU: {response}")
            return 1

        print(f"Allocated GPU ID: {response}")
        append_github_env("CUDA_VISIBLE_DEVICES", response)
        append_github_env("STEP_TIMESTAMP", str(now_ms()))
        print(f"CUDA_VISIBLE_DEVICES set to {response}")
        print_status(args.base_url)
        return 0


def command_release(args: argparse.Namespace) -> int:
    encoded_test = urllib.parse.quote(args.test, safe="")
    encoded_runner = urllib.parse.quote(args.runner, safe="")
    url = (
        f"{args.base_url}/gpu/release?runid={args.run_id}&gpu={args.gpu_id}"
        f"&timestamp={args.timestamp}&test={encoded_test}&runner={encoded_runner}"
    )
    print(url)

    try:
        response = fetch_text(url, timeout=args.timeout).strip()
    except Exception as exc:
        print(f"Failed to release GPU: {exc}")
        return 0

    print(f"response: {response}")
    if response != args.gpu_id:
        print(f"Error: response ({response}) != expected ({args.gpu_id})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    allocate = subparsers.add_parser("allocate")
    allocate.add_argument("--base-url", required=True)
    allocate.add_argument("--run-id", required=True)
    allocate.add_argument("--test", required=True)
    allocate.add_argument("--runner", required=True)
    allocate.add_argument("--count", required=True)
    allocate.add_argument("--sleep-sec", type=float, default=5)
    allocate.add_argument("--timeout-sec", type=int, default=18000)
    allocate.add_argument("--request-timeout", type=float, default=10)
    allocate.add_argument("--retries", type=int, default=3)
    allocate.add_argument("--retry-delay", type=float, default=1)
    allocate.add_argument("--require-single", action="store_true")
    allocate.set_defaults(handler=command_allocate)

    release = subparsers.add_parser("release")
    release.add_argument("--base-url", required=True)
    release.add_argument("--run-id", required=True)
    release.add_argument("--gpu-id", required=True)
    release.add_argument("--timestamp", required=True)
    release.add_argument("--test", required=True)
    release.add_argument("--runner", required=True)
    release.add_argument("--timeout", type=float, default=10)
    release.set_defaults(handler=command_release)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
