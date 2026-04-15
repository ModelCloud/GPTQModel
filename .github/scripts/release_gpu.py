import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request


def fetch_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    encoded_test = urllib.parse.quote(args.test, safe="")
    encoded_runner = urllib.parse.quote(args.runner, safe="")
    url = (
        f"{args.base_url}/gpu/release?runid={args.run_id}&gpu={args.gpu_id}"
        f"&timestamp={args.timestamp}&test={encoded_test}&runner={encoded_runner}"
    )
    print(url)

    try:
        resp = fetch_text(url, timeout=args.timeout).strip()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Failed to release GPU: {exc}")
        return 0

    print(f"response: {resp}")
    if resp != args.gpu_id:
        print(f"Error: response ({resp}) != expected ({args.gpu_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
