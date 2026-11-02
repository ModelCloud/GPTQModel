# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bounds and diagnostics for remote image fetching in calibration data.

Regression coverage for the unguarded ``requests.get(url, stream=True)`` calls
that used to make a bad image URL hang a quantization run forever.
"""

import ast
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
        if path == "/ok":
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
    image = fetch_image({"image": f"{base}/ok"})
    assert isinstance(image, Image.Image)
    assert image.size == (8, 8)


def test_load_image_from_url_uses_bounded_timeout(image_server, monkeypatch):
    base, server = image_server
    observed = {}
    original_get = image_utils.requests.get

    def spy_get(url, **kwargs):
        observed["kwargs"] = kwargs
        return original_get(url, **kwargs)

    monkeypatch.setattr(image_utils.requests, "get", spy_get)
    load_image_from_url(f"{base}/ok", timeout=(1, 2))
    assert observed["kwargs"]["stream"] is True
    assert observed["kwargs"]["timeout"] == (1.0, 2.0)

    # The public wrapper must also pass a timeout instead of relying on the
    # requests default (which is "wait forever").
    observed.clear()
    fetch_image({"image_url": f"{base}/ok"})
    assert observed["kwargs"]["timeout"] is not None


def test_hanging_server_is_bounded_by_read_timeout(image_server):
    base, server = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(
            f"{base}/hang",
            timeout=(0.2, 0.2),
            attempts=2,
            backoff_seconds=0,
        )
    assert "timed out" in str(excinfo.value)
    assert excinfo.value.url.endswith("/hang")


def test_timeout_budget_is_not_multiplied_by_retries(image_server):
    base, _ = image_server
    started = time.monotonic()
    with pytest.raises(ImageFetchError):
        load_image_from_url(f"{base}/hang", timeout=(0.2, 0.2), total_timeout=0.6, attempts=3)
    # Timed-out attempts are not retried, so this must finish well under the
    # 3 * (connect + read) budget the naive implementation would need.
    assert time.monotonic() - started < 3.0


def test_retries_transient_http_failures(image_server):
    base, server = image_server
    image = load_image_from_url(f"{base}/flaky-then-ok", attempts=2, backoff_seconds=0)
    assert image.size == (8, 8)
    assert server.hits == 2


def test_permanent_request_errors_are_not_retried(monkeypatch):
    calls = []

    def fail(url, **kwargs):
        calls.append(url)
        raise image_utils.requests.exceptions.MissingSchema("boom")

    monkeypatch.setattr(image_utils.requests, "get", fail)
    with pytest.raises(ImageFetchError):
        load_image_from_url("not-a-url", attempts=3, backoff_seconds=0)
    assert len(calls) == 1


def test_invalid_env_limit_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_IMAGE_FETCH_MAX_BYTES", "abc")
    with pytest.warns(RuntimeWarning):
        assert image_utils._resolve_max_bytes(None) == image_utils.DEFAULT_IMAGE_FETCH_MAX_BYTES


def test_client_error_is_not_retried(image_server):
    base, server = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/http-error", attempts=3, backoff_seconds=0)
    assert "HTTP 404" in str(excinfo.value)


def test_declared_oversize_body_is_rejected(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/oversize-declared", max_bytes=len(PNG))
    assert "exceeding" in str(excinfo.value)


def test_streamed_oversize_body_is_rejected(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/oversize-streamed", max_bytes=len(PNG))
    assert "byte limit" in str(excinfo.value)


def test_undecodable_body_raises_diagnostic_error(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/not-an-image")
    assert "decodable image" in str(excinfo.value)


def test_empty_body_is_a_failure(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/empty", attempts=2, backoff_seconds=0)
    assert "empty response" in str(excinfo.value)


def test_slow_drip_body_is_bounded_by_total_timeout(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(
            f"{base}/slow-drip",
            timeout=(1.0, 1.0),
            total_timeout=0.3,
            attempts=1,
        )
    assert "total time limit" in str(excinfo.value)


def test_pixel_bomb_is_rejected_before_decode(image_server):
    base, _ = image_server
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/huge-dimensions", max_pixels=10_000)
    assert "pixel limit" in str(excinfo.value)


def test_max_pixels_env_override(image_server, monkeypatch):
    base, _ = image_server
    monkeypatch.setenv("GPTQMODEL_IMAGE_FETCH_MAX_PIXELS", "10")
    with pytest.raises(ImageFetchError) as excinfo:
        load_image_from_url(f"{base}/ok")
    assert "pixel limit" in str(excinfo.value)


def test_max_bytes_env_override(image_server, monkeypatch):
    base, _ = image_server
    monkeypatch.setenv("GPTQMODEL_IMAGE_FETCH_MAX_BYTES", "16")
    with pytest.raises(ImageFetchError):
        load_image_from_url(f"{base}/ok")


def test_invalid_max_bytes_raises():
    with pytest.raises(ValueError):
        load_image_from_url("http://127.0.0.1:1/x", max_bytes=0)


@pytest.mark.parametrize("model_cls", [InternS1QModel, Ovis2_5QModel])
def test_replace_image_with_pil_goes_through_bounded_fetch(image_server, monkeypatch, model_cls):
    base, _ = image_server
    calls = []
    original = image_utils.load_image_from_url

    def spy(url, **kwargs):
        calls.append(url)
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
