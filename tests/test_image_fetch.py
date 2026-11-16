# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bounds and diagnostics for remote image fetching in calibration data.

Regression coverage for the unguarded ``requests.get(url, stream=True)`` calls
that used to make a bad image URL hang a quantization run forever.
"""

import ast
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from gptqmodel.models.definitions.interns1 import InternS1QModel
from gptqmodel.models.definitions.ovis2_5 import Ovis2_5QModel
from gptqmodel.utils import image as image_utils
from gptqmodel.utils.image import ImageFetchError, fetch_image, load_image_from_url


def _png_bytes(size=(8, 8)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (17, 34, 51)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png_bytes()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, status, body=b"", headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        path = self.path
        if path == "/endless-headers":
            self.server.header_started.set()
            try:
                self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                self.wfile.flush()
                while not self.server.release_hang.wait(0.02):
                    self.wfile.write(b"x")
                    self.wfile.flush()
            except OSError:
                # Only the client can end this stream before fixture teardown.
                self.server.header_disconnected.set()
        elif path == "/ok":
            self._send(200, PNG, {"Content-Type": "image/png", "Content-Length": str(len(PNG))})
        elif path == "/http-error":
            self._send(404, b"missing")
        elif path == "/flaky-then-ok":
            server = self.server
            server.hits += 1
            if server.hits == 1:
                self._send(503, b"busy")
            else:
                self._send(200, PNG, {"Content-Length": str(len(PNG))})
        elif path == "/oversize-declared":
            self._send(200, PNG, {"Content-Length": str(len(PNG) + 1024)})
        elif path == "/oversize-streamed":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PNG) * 4))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for _ in range(4):
                    self.wfile.write(PNG)
            except OSError:
                # The client aborts once the size cap is hit; expected here.
                pass
        elif path == "/slow-drip":
            # Trickle bytes below the per-read timeout so only a wall-clock
            # budget can stop the download.
            drip = 400
            self.send_response(200)
            self.send_header("Content-Length", str(drip))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for _ in range(drip):
                    if self.server.release_hang.wait(0.05):
                        break
                    self.wfile.write(b"\x00")
                    self.wfile.flush()
            except OSError:
                # The client aborts once its deadline expires; expected here.
                pass
        elif path == "/slow-headers":
            # Delay the response headers themselves; the per-read timeout only
            # starts once a read begins, so only a total budget can stop this.
            self.server.release_hang.wait(0.4)
            self._send(200, PNG, {"Content-Length": str(len(PNG))})
        elif path.startswith("/redirect/"):
            # A chain of individually responsive redirects that together
            # outlive the wall-clock budget.
            remaining = int(path.rsplit("/", 1)[1])
            self.server.release_hang.wait(0.1)
            target = f"/redirect/{remaining - 1}" if remaining else "/ok"
            self._send(302, headers={"Location": target, "Content-Length": "0"})
        elif path == "/redirect-to-metadata":
            self._send(
                302,
                headers={"Location": "http://169.254.169.254/latest/meta-data", "Content-Length": "0"},
            )
        elif path == "/header-drip":
            # Trickle header bytes below the per-read timeout so the read
            # timeout keeps resetting and only the budget can stop the request.
            self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Drip: ")
            self.wfile.flush()
            try:
                for _ in range(12):
                    if self.server.release_hang.wait(0.03):
                        break
                    self.wfile.write(b"x")
                    self.wfile.flush()
                self.wfile.write(
                    b"\r\nContent-Length: "
                    + str(len(PNG)).encode()
                    + b"\r\nConnection: close\r\n\r\n"
                    + PNG
                )
                self.wfile.flush()
            except OSError:
                # The client gave up at its deadline; expected here.
                pass
        elif path == "/huge-dimensions":
            # Tiny compressed payload that expands to a huge bitmap.
            buffer = BytesIO()
            Image.new("L", (5000, 5000), 0).save(buffer, format="PNG")
            data = buffer.getvalue()
            self._send(200, data, {"Content-Length": str(len(data))})
        elif path == "/not-an-image":
            self._send(200, b"definitely not a png", {"Content-Length": "19"})
        elif path == "/empty":
            self._send(200, b"", {"Content-Length": "0"})
        elif path == "/hang":
            self.server.release_hang.wait(30)
        else:  # pragma: no cover - defensive
            self._send(500, b"unknown")


@pytest.fixture()
def image_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.hits = 0
    server.release_hang = threading.Event()
    server.header_started = threading.Event()
    server.header_disconnected = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.release_hang.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fetch_image_http_url_returns_decoded_image(image_server):
    base, _ = image_server
    image = fetch_image({"image": f"{base}/ok"}, allow_private_addresses=True)
    assert isinstance(image, Image.Image)
    assert image.size == (8, 8)


def test_load_image_from_url_uses_bounded_timeout(image_server, monkeypatch):
    base, _ = image_server
    observed = {}
    original_download = image_utils._download_image_bytes

    def spy_download(url, **kwargs):
        observed["kwargs"] = kwargs
        return original_download(url, **kwargs)

    monkeypatch.setattr(image_utils, "_download_image_bytes", spy_download)
    load_image_from_url(f"{base}/ok", timeout=(1, 2), allow_private_addresses=True)
    assert observed["kwargs"]["timeout"] == (1.0, 2.0)
    observed.clear()
    fetch_image({"image_url": f"{base}/ok"}, allow_private_addresses=True)
    assert observed["kwargs"]["timeout"] is not None

    # Check the network side of the subprocess boundary independently too.
    original_get = image_utils.requests.get

    def spy_get(url, **kwargs):
        observed["request"] = kwargs
        return original_get(url, **kwargs)

    monkeypatch.setattr(image_utils.requests, "get", spy_get)
    image_utils._download_image_bytes_in_worker(
        f"{base}/ok", timeout=(1.0, 2.0), max_bytes=1024,
        deadline=time.monotonic() + 10, total_timeout=10,
        allow_private_addresses=True,
    )
    assert observed["request"] == {
        "stream": True, "timeout": (1.0, 2.0), "allow_redirects": False,
    }


def test_hanging_server_is_bounded_by_read_timeout(image_server):
    base, server = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(
            f"{base}/hang",
            timeout=(0.2, 0.2),
            attempts=2,
            backoff_seconds=0, allow_private_addresses=True,
        )
    assert "timed out" in str(excinfo.value)
    assert excinfo.value.url.endswith("/hang")


def test_timeout_budget_is_not_multiplied_by_retries(image_server):
    base, _ = image_server
    started = time.monotonic()
    with pytest.raises(ImageFetchError):
        load_image_from_url(f"{base}/hang", timeout=(0.2, 0.2), total_timeout=0.6, attempts=3, allow_private_addresses=True)
    # Timed-out attempts are not retried, so this must finish well under the
    # 3 * (connect + read) budget the naive implementation would need.
    assert time.monotonic() - started < 3.0


def test_retries_transient_http_failures(image_server):
    base, server = image_server
    image = load_image_from_url(f"{base}/flaky-then-ok", attempts=2, backoff_seconds=0, allow_private_addresses=True)
    assert image.size == (8, 8)
    assert server.hits == 2


def test_permanent_request_errors_are_not_retried(monkeypatch):
    calls = []
    original_download = image_utils._download_image_bytes

    def spy(url, **kwargs):
        calls.append(url)
        return original_download(url, **kwargs)

    monkeypatch.setattr(image_utils, "_download_image_bytes", spy)
    with pytest.raises(ImageFetchError, match="unsupported URL scheme|request failed"):
        load_image_from_url("not-a-url", attempts=3, backoff_seconds=0, allow_private_addresses=True)
    assert len(calls) == 1


def test_invalid_env_limit_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_IMAGE_FETCH_MAX_BYTES", "abc")
    with pytest.warns(RuntimeWarning):
        assert image_utils._resolve_max_bytes(None) == image_utils.DEFAULT_IMAGE_FETCH_MAX_BYTES


def test_client_error_is_not_retried(image_server):
    base, server = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/http-error", attempts=3, backoff_seconds=0, allow_private_addresses=True)
    assert "HTTP 404" in str(excinfo.value)


def test_declared_oversize_body_is_rejected(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/oversize-declared", max_bytes=len(PNG), allow_private_addresses=True)
    assert "exceeding" in str(excinfo.value)


def test_streamed_oversize_body_is_rejected(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/oversize-streamed", max_bytes=len(PNG), allow_private_addresses=True)
    assert "byte limit" in str(excinfo.value)


def test_undecodable_body_raises_diagnostic_error(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/not-an-image", allow_private_addresses=True)
    assert "decodable image" in str(excinfo.value)


def test_empty_body_is_a_failure(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/empty", attempts=2, backoff_seconds=0, allow_private_addresses=True)
    assert "empty response" in str(excinfo.value)


def test_slow_drip_body_is_bounded_by_total_timeout(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(
            f"{base}/slow-drip",
            timeout=(1.0, 1.0),
            total_timeout=0.3,
            attempts=1, allow_private_addresses=True,
        )
    assert "total time limit" in str(excinfo.value)


def test_slow_headers_are_bounded_by_total_timeout(image_server):
    base, _ = image_server
    started = time.monotonic()
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(
            f"{base}/slow-headers",
            timeout=(1.0, 1.0),
            total_timeout=0.2,
            attempts=1, allow_private_addresses=True,
        )
    assert "total time limit" in str(excinfo.value)
    assert time.monotonic() - started < 0.8


def test_redirect_chain_is_bounded_by_total_timeout(image_server):
    base, _ = image_server
    started = time.monotonic()
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(
            f"{base}/redirect/3",
            timeout=(1.0, 1.0),
            total_timeout=0.2,
            attempts=1, allow_private_addresses=True,
        )
    assert "total time limit" in str(excinfo.value)
    assert time.monotonic() - started < 0.8


def test_header_drip_is_bounded_by_total_timeout(image_server):
    base, _ = image_server
    started = time.monotonic()
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(
            f"{base}/header-drip",
            timeout=(0.1, 0.1),
            total_timeout=0.2,
            attempts=1, allow_private_addresses=True,
        )
    assert "total time limit" in str(excinfo.value)
    assert time.monotonic() - started < 0.8


def test_header_deadlines_release_workers_and_sockets(image_server, monkeypatch):
    base, server = image_server
    children = []
    original_popen = subprocess.Popen

    def track_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", track_child)
    prior_threads = {t.ident for t in threading.enumerate() if t.name == "gptqmodel-image-fetch"}
    for _ in range(3):
        server.header_started.clear()
        server.header_disconnected.clear()
        with pytest.raises(ImageFetchError, match="total time limit"):
            load_image_from_url(
                f"{base}/endless-headers", timeout=(0.3, 0.3),
                total_timeout=0.6, attempts=1, allow_private_addresses=True,
            )
        # Ensure this exercised a live header reader, not just process startup.
        assert server.header_started.is_set()
        assert server.header_disconnected.wait(0.5), "timed-out request still owns its socket"
        assert {t.ident for t in threading.enumerate() if t.name == "gptqmodel-image-fetch"} == prior_threads
        assert children and all(child.returncode is not None for child in children)
        assert all(child.stdin.closed and child.stdout.closed and child.stderr.closed for child in children)
    assert len(children) == 3
    # Cancellation must not poison subsequent requests.
    assert load_image_from_url(f"{base}/ok", allow_private_addresses=True).size == (8, 8)


def test_fetch_worker_is_reaped_on_caller_interrupt(image_server, monkeypatch):
    base, server = image_server
    children = []
    original_popen = subprocess.Popen

    def interrupt_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        communicate = child.communicate

        def interrupted(*args, **kwargs):
            child.communicate = communicate
            # Submit the request, then interrupt while the peer is still open.
            child.stdin.write(args[0])
            child.stdin.close()
            child.stdin = None
            assert server.header_started.wait(3)
            raise KeyboardInterrupt

        child.communicate = interrupted
        return child

    monkeypatch.setattr(subprocess, "Popen", interrupt_child)
    with pytest.raises(KeyboardInterrupt):
        load_image_from_url(f"{base}/endless-headers", total_timeout=10, allow_private_addresses=True)
    assert server.header_disconnected.wait(0.5)
    assert children[0].returncode is not None
    assert children[0].stdout.closed and children[0].stderr.closed


def test_decode_time_is_bounded_by_total_timeout(image_server, monkeypatch):
    base, _ = image_server
    original_open = image_utils.Image.open

    def slow_open(*args, **kwargs):
        time.sleep(0.4)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(image_utils.Image, "open", slow_open)
    started = time.monotonic()
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/ok", total_timeout=0.2, attempts=1, allow_private_addresses=True)
    assert "total time limit" in str(excinfo.value)
    assert time.monotonic() - started < 0.8


def test_pixel_bomb_is_rejected_before_decode(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/huge-dimensions", max_pixels=10_000, allow_private_addresses=True)
    assert "pixel limit" in str(excinfo.value)


def test_max_pixels_env_override(image_server, monkeypatch):
    base, _ = image_server
    monkeypatch.setenv("GPTQMODEL_IMAGE_FETCH_MAX_PIXELS", "10")
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/ok", allow_private_addresses=True)
    assert "pixel limit" in str(excinfo.value)


def test_max_bytes_env_override(image_server, monkeypatch):
    base, _ = image_server
    monkeypatch.setenv("GPTQMODEL_IMAGE_FETCH_MAX_BYTES", "16")
    with pytest.raises(ImageFetchError):
        load_image_from_url(f"{base}/ok", allow_private_addresses=True)


def test_invalid_max_bytes_raises():
    with pytest.raises(ValueError):
        load_image_from_url("http://127.0.0.1:1/x", max_bytes=0, allow_private_addresses=True)


@pytest.mark.parametrize("model_cls", [InternS1QModel, Ovis2_5QModel])
def test_replace_image_with_pil_goes_through_bounded_fetch(image_server, monkeypatch, model_cls):
    base, _ = image_server
    calls = []
    original = image_utils.load_image_from_url

    def spy(url, **kwargs):
        calls.append(url)
        kwargs.setdefault("allow_private_addresses", True)
        return original(url, **kwargs)

    monkeypatch.setattr(image_utils, "load_image_from_url", spy)
    monkeypatch.setattr(
        "gptqmodel.models.definitions.interns1.load_image_from_url", spy, raising=False
    )
    monkeypatch.setattr(
        "gptqmodel.models.definitions.ovis2_5.load_image_from_url", spy, raising=False
    )

    sample = [
        {"role": "user", "content": [{"type": "image", "image": f"{base}/ok"}]},
    ]
    out = model_cls.replace_image_with_pil(sample)
    assert calls == [f"{base}/ok"]
    assert isinstance(out[0]["content"][0]["image"], Image.Image)


@pytest.mark.parametrize("model_cls", [InternS1QModel, Ovis2_5QModel])
def test_replace_image_with_pil_surfaces_fetch_failure(monkeypatch, model_cls):
    def fail(url, **kwargs):
        raise ImageFetchError(url, "simulated failure")

    monkeypatch.setattr(
        "gptqmodel.models.definitions.interns1.load_image_from_url", fail, raising=False
    )
    monkeypatch.setattr(
        "gptqmodel.models.definitions.ovis2_5.load_image_from_url", fail, raising=False
    )

    sample = [{"role": "user", "content": [{"type": "image", "image": "https://example.invalid/x.png"}]}]
    with pytest.raises(ImageFetchError):
        model_cls.replace_image_with_pil(sample)


def _empty_except_without_comment(source_path):
    """Return ``except ...: pass`` blocks that carry no explanatory comment.

    Mirrors CodeQL's ``py/empty-except`` rule: a comment anywhere between the
    ``except`` line and the last statement of its body satisfies it.
    """

    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not all(isinstance(stmt, ast.Pass) for stmt in node.body):
            continue
        last_line = max(stmt.end_lineno or stmt.lineno for stmt in node.body)
        commented = any(
            "#" in lines[number - 1]
            for number in range(node.lineno, last_line + 1)
        )
        if not commented:
            offenders.append(node.lineno)
    return offenders


def test_best_effort_except_blocks_are_documented():
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    for source_path in (repo_root / "gptqmodel" / "utils" / "image.py", here):
        assert _empty_except_without_comment(source_path) == [], (
            f"{source_path.name} has an empty `except` with no explanatory comment"
        )


def test_private_and_link_local_destinations_are_blocked_by_default():
    for url in (
        "http://127.0.0.1/x",
        "http://10.0.0.1/x",
        "http://169.254.169.254/latest/meta-data",
    ):
        with pytest.raises(ImageFetchError, match="private|link-local"):
            image_utils._validate_remote_url(url, allow_private_addresses=False)

    # An explicit opt-in permits loopback/private hosts...
    image_utils._validate_remote_url("http://127.0.0.1/x", allow_private_addresses=True)
    # ...but never the link-local metadata range.
    with pytest.raises(ImageFetchError, match="link-local"):
        image_utils._validate_remote_url(
            "http://169.254.169.254/latest/meta-data", allow_private_addresses=True
        )


def test_redirect_to_link_local_is_blocked(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError, match="link-local"):
        load_image_from_url(
            f"{base}/redirect-to-metadata",
            attempts=1,
            allow_private_addresses=True,
        )


def test_redirect_chain_respects_the_redirect_limit(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError, match="redirect limit"):
        load_image_from_url(
            f"{base}/redirect/3",
            max_redirects=1,
            attempts=1,
            allow_private_addresses=True,
        )


def test_local_image_paths_apply_the_pixel_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_IMAGE_FETCH_MAX_PIXELS", "100")
    path = tmp_path / "large.png"
    Image.new("L", (100, 100), 0).save(path)

    with pytest.raises(ImageFetchError, match="pixel limit"):
        fetch_image({"image": str(path)})
    with pytest.raises(ImageFetchError, match="pixel limit"):
        fetch_image({"image": f"file://{path}"})
