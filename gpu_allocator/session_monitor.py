# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Optional session-liveness monitors for the GPU allocator."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class SessionMonitor:
    """Abstract base for checking whether an external session is still alive."""

    def is_alive(self, session_id: str) -> bool:
        """Return ``True`` if ``session_id`` is still active.

        This method must be thread-safe and should not raise exceptions;
        failures must be reported via logs and the caller should treat the
        session as alive to avoid prematurely killing active work.
        """
        raise NotImplementedError  # pragma: no cover


class NullSessionMonitor(SessionMonitor):
    """Always reports a session as alive (no external monitoring)."""

    def is_alive(self, session_id: str) -> bool:
        return True


class DevinApiSessionMonitor(SessionMonitor):
    """Check Devin session liveness via the Devin REST API.

    The monitor requires a Devin API token and organization id. It caches
    results per session id for ``cache_ttl_seconds`` to avoid hammering the
    API during the TTL janitor's periodic sweeps.
    """

    API_BASE = "https://api.devin.ai"

    def __init__(
        self,
        api_token: str,
        org_id: str,
        *,
        api_base: Optional[str] = None,
        cache_ttl_seconds: float = 30.0,
        timeout_seconds: float = 10.0,
    ):
        self._api_token = api_token
        self._org_id = org_id
        self._api_base = api_base or self.API_BASE
        self._cache_ttl_seconds = float(cache_ttl_seconds)
        self._timeout_seconds = float(timeout_seconds)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> Optional["DevinApiSessionMonitor"]:
        """Create a monitor from ``DEVIN_API_TOKEN`` and ``DEVIN_ORG_ID`` if set."""
        token = os.environ.get("DEVIN_API_TOKEN") or os.environ.get("DEVIN_API_KEY")
        org_id = os.environ.get("DEVIN_ORG_ID")
        if token and org_id:
            return cls(token, org_id)
        return None

    def _looks_like_devin_id(self, session_id: str) -> bool:
        return isinstance(session_id, str) and session_id.startswith("devin-")

    def _fetch(self, session_id: str) -> bool:
        if not self._looks_like_devin_id(session_id):
            return True

        url = f"{self._api_base}/v3/organizations/{self._org_id}/sessions/{session_id}"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            # 404 means the session does not exist / has been deleted.
            if exc.code == 404:
                return False
            # 403/401 means the token cannot access this endpoint; do not kill leases.
            return True
        except Exception:
            # Network or parsing failure: fail safe and assume alive.
            return True

        status = data.get("status")
        # "exit", "error", and "suspended" are terminal or effectively dead.
        return status not in {"exit", "error", "suspended"}

    def is_alive(self, session_id: str) -> bool:
        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(session_id)
            if entry is not None and now - entry["ts"] < self._cache_ttl_seconds:
                return entry["alive"]

        alive = self._fetch(session_id)

        with self._cache_lock:
            self._cache[session_id] = {"ts": now, "alive": alive}
        return alive
