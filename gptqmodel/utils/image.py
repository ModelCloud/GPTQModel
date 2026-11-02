# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import base64
import os
import socket
import threading
import time
import warnings
from io import BytesIO
from typing import Optional, Tuple, Union

import requests
from PIL import Image


# Bounds for fetching remote images (multimodal calibration data).  Without
# them a broken/hostile URL can hang a quantization run forever or exhaust
# memory.  The byte and pixel limits can also be tuned from the environment.
DEFAULT_IMAGE_FETCH_CONNECT_TIMEOUT = 5.0
DEFAULT_IMAGE_FETCH_READ_TIMEOUT = 30.0
DEFAULT_IMAGE_FETCH_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_IMAGE_FETCH_MAX_PIXELS = 64 * 1024 * 1024
DEFAULT_IMAGE_FETCH_TOTAL_TIMEOUT = 120.0
DEFAULT_IMAGE_FETCH_ATTEMPTS = 3
DEFAULT_IMAGE_FETCH_BACKOFF_SECONDS = 0.5

_IMAGE_FETCH_MAX_BYTES_ENV = "GPTQMODEL_IMAGE_FETCH_MAX_BYTES"
_IMAGE_FETCH_MAX_PIXELS_ENV = "GPTQMODEL_IMAGE_FETCH_MAX_PIXELS"

# HTTP statuses that are worth retrying: rate limiting and server-side errors.
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

# requests exceptions caused by the URL itself; retrying cannot help.
_PERMANENT_REQUEST_ERRORS = tuple(
    exc
    for exc in (
        getattr(requests.exceptions, name, None)
        for name in (
            "MissingSchema",
            "InvalidSchema",
            "InvalidURL",
            "URLRequired",
            "InvalidHeader",
            "TooManyRedirects",
        )
    )
    if exc is not None
)


class ImageFetchError(RuntimeError):
    """A remote image could not be fetched within the configured bounds."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to fetch image URL {_shorten_url(url)!r}: {reason}")


def _shorten_url(url: str, limit: int = 200) -> str:
    if len(url) <= limit:
        return url
    return f"{url[:limit]}..."


def _abort_response(response: requests.Response) -> None:
    """Best-effort shutdown of the response socket from the watchdog thread.

    Calling ``response.close()`` while another thread reads is not safe (it
    mutates ``http.client`` state mid-read), so only shut the socket down: that
    makes the pending read fail immediately without corrupting the reader.
    """

    sockets = []
    raw = getattr(response, "raw", None)
    connection = getattr(raw, "_connection", None)
    sockets.append(getattr(connection, "sock", None))
    try:
        sockets.append(response.raw._fp.fp.raw._sock)
    except Exception:
        # Response backends need not expose this private path; the socket from
        # ``raw._connection`` above is still tried.
        pass
    for sock in sockets:
        if sock is None:
            continue
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            # Best effort: the socket may already be closed or racing another
            # thread's teardown; aborting is only an optimization.
            pass


class _FetchFailure(Exception):
    def __init__(self, reason: str, *, retryable: bool):
        self.reason = reason
        self.retryable = retryable
        super().__init__(reason)


def _resolve_limit(value: Optional[int], env_name: str, default: int) -> int:
    if value is None:
        raw = os.getenv(env_name)
        if raw is not None:
            try:
                value = int(raw)
            except ValueError:
                warnings.warn(
                    f"{env_name}={raw!r} is not an integer; using the default of {default}.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                value = None
    if value is None:
        return default
    value = int(value)
    if value <= 0:
        raise ValueError(f"{env_name} must be positive, got {value}")
    return value


def _resolve_max_bytes(max_bytes: Optional[int]) -> int:
    return _resolve_limit(max_bytes, _IMAGE_FETCH_MAX_BYTES_ENV, DEFAULT_IMAGE_FETCH_MAX_BYTES)


def _resolve_max_pixels(max_pixels: Optional[int]) -> int:
    return _resolve_limit(max_pixels, _IMAGE_FETCH_MAX_PIXELS_ENV, DEFAULT_IMAGE_FETCH_MAX_PIXELS)


def _resolve_timeout(
    timeout: Optional[Union[float, Tuple[float, float]]],
) -> Union[float, Tuple[float, float]]:
    if timeout is None:
        return (DEFAULT_IMAGE_FETCH_CONNECT_TIMEOUT, DEFAULT_IMAGE_FETCH_READ_TIMEOUT)
    if isinstance(timeout, (tuple, list)):
        if len(timeout) != 2:
            raise ValueError(f"timeout must be a number or (connect, read) pair, got {timeout!r}")
        return float(timeout[0]), float(timeout[1])
    timeout = float(timeout)
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    return timeout


def _download_image_bytes(
    url: str,
    *,
    timeout: Union[float, Tuple[float, float]],
    max_bytes: int,
    total_timeout: float,
) -> bytes:
    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except requests.Timeout as exc:
        # A stall on one attempt usually means the origin is unreachable; do
        # not multiply the wall-clock cost by retrying it.
        raise _FetchFailure(f"timed out after {timeout} seconds ({exc})", retryable=False) from exc
    except _PERMANENT_REQUEST_ERRORS as exc:
        raise _FetchFailure(f"request failed ({exc})", retryable=False) from exc
    except requests.RequestException as exc:
        raise _FetchFailure(f"request failed ({exc})", retryable=True) from exc

    with response:
        status = response.status_code
        if status >= 400:
            raise _FetchFailure(
                f"HTTP {status} {response.reason or ''}".strip(),
                retryable=status in _RETRYABLE_STATUS_CODES,
            )

        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError:
                declared_bytes = None
            if declared_bytes is not None and declared_bytes > max_bytes:
                raise _FetchFailure(
                    f"response is {declared_bytes} bytes, exceeding the {max_bytes} byte limit",
                    retryable=False,
                )

        buffer = BytesIO()
        received = 0
        deadline = time.monotonic() + total_timeout
        # A per-read timeout alone lets a server keep a download alive forever
        # by trickling a byte at a time.  A watchdog closes the socket when the
        # whole download exceeds the wall-clock budget.
        watchdog = threading.Timer(total_timeout, _abort_response, args=(response,))
        watchdog.daemon = True
        watchdog.start()
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                received += len(chunk)
                if time.monotonic() > deadline:
                    raise _FetchFailure(
                        f"download exceeded the {total_timeout} second total time limit",
                        retryable=False,
                    )
                if received > max_bytes:
                    raise _FetchFailure(
                        f"response exceeded the {max_bytes} byte limit",
                        retryable=False,
                    )
                buffer.write(chunk)
        except requests.Timeout as exc:
            raise _FetchFailure(f"timed out while reading the response ({exc})", retryable=False) from exc
        except requests.RequestException as exc:
            if time.monotonic() >= deadline:
                raise _FetchFailure(
                    f"download exceeded the {total_timeout} second total time limit",
                    retryable=False,
                ) from exc
            raise _FetchFailure(f"failed while reading the response ({exc})", retryable=True) from exc
        except (OSError, ValueError, AttributeError) as exc:
            # Aborting the socket from the watchdog surfaces as a socket or
            # file-object error, not a requests exception.
            if time.monotonic() >= deadline:
                raise _FetchFailure(
                    f"download exceeded the {total_timeout} second total time limit",
                    retryable=False,
                ) from exc
            raise _FetchFailure(f"failed while reading the response ({exc})", retryable=True) from exc
        finally:
            watchdog.cancel()

    data = buffer.getvalue()
    if not data:
        # An empty body is a property of the resource, not a transient fault
        # (legitimate 204/304 responses land here too).
        raise _FetchFailure("empty response body", retryable=False)
    return data


def load_image_from_url(
    url: str,
    *,
    timeout: Optional[Union[float, Tuple[float, float]]] = None,
    max_bytes: Optional[int] = None,
    max_pixels: Optional[int] = None,
    total_timeout: Optional[float] = None,
    attempts: Optional[int] = None,
    backoff_seconds: Optional[float] = None,
) -> Image.Image:
    """Download ``url`` and return a decoded :class:`PIL.Image.Image`.

    The download is bounded by connect/read timeouts, a maximum byte count and
    a small number of retries for transient failures.  Terminal failures (bad
    status, oversized body, undecodable bytes) raise :class:`ImageFetchError`
    with the URL and reason so calibration failures are diagnosable instead of
    hanging or producing an opaque PIL error.
    """

    if not isinstance(url, str) or not url:
        raise ImageFetchError(str(url), "image URL must be a non-empty string")

    timeout = _resolve_timeout(timeout)
    max_bytes = _resolve_max_bytes(max_bytes)
    max_pixels = _resolve_max_pixels(max_pixels)
    if total_timeout is None:
        total_timeout = DEFAULT_IMAGE_FETCH_TOTAL_TIMEOUT
    total_timeout = float(total_timeout)
    if total_timeout <= 0:
        raise ValueError(f"total_timeout must be positive, got {total_timeout}")
    attempts = DEFAULT_IMAGE_FETCH_ATTEMPTS if attempts is None else int(attempts)
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    if backoff_seconds is None:
        backoff_seconds = DEFAULT_IMAGE_FETCH_BACKOFF_SECONDS
    backoff_seconds = max(0.0, float(backoff_seconds))

    last_failure: Optional[_FetchFailure] = None
    # The wall-clock budget covers every attempt, not just one download.
    deadline = time.monotonic() + total_timeout
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            last_failure = _FetchFailure(
                f"download exceeded the {total_timeout} second total time limit",
                retryable=False,
            )
            break
        try:
            data = _download_image_bytes(
                url, timeout=timeout, max_bytes=max_bytes, total_timeout=remaining
            )
        except _FetchFailure as failure:
            last_failure = failure
            if not failure.retryable or attempt == attempts:
                break
            delay = backoff_seconds * (2 ** (attempt - 1))
            if time.monotonic() + delay >= deadline:
                break
            time.sleep(delay)
            continue

        try:
            image = Image.open(BytesIO(data))
        except Image.DecompressionBombError as exc:
            raise ImageFetchError(url, f"image is too large to decode safely ({exc})") from exc
        except Exception as exc:  # PIL raises several unrelated exception types
            raise ImageFetchError(url, f"response is not a decodable image ({exc})") from exc

        # Check the declared dimensions before ``load()`` allocates the pixel
        # buffer: a small compressed file can still expand to a huge bitmap.
        width, height = image.size
        if width * height > max_pixels:
            image.close()
            raise ImageFetchError(
                url,
                f"image is {width}x{height} ({width * height} pixels), exceeding the "
                f"{max_pixels} pixel limit",
            )

        try:
            image.load()
        except Exception as exc:
            image.close()
            raise ImageFetchError(url, f"response is not a decodable image ({exc})") from exc
        return image

    assert last_failure is not None
    attempted = f" after {attempt} attempt(s)" if attempt > 1 else ""
    raise ImageFetchError(url, f"{last_failure.reason}{attempted}")


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def fetch_image(
    ele: dict[str, str | Image.Image],
    image_patch_size: int | None = None,
) -> Image.Image:
    # Some multimodal model adapters forward image-related kwargs here even
    # though local image loading does not need them.
    del image_patch_size
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = load_image_from_url(image)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    return image_obj
