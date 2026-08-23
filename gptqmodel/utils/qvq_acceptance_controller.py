# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent process controller for Qwen3 acceptance measurements."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

CONTROLLER_SCHEMA = "qvq-acceptance-controller-v1"
CONTROLLER_STAGES = ("quantization_producer", "fresh_process_reload", "acceptance_evaluation")
_ENV_PREFIX = "GPTQMODEL_QVQ_CONTROLLER_"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _new_identity() -> str:
    return secrets.token_hex(32)


def _openssl(args: Sequence[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["openssl", *args],
            input=input_bytes,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("OpenSSL Ed25519 controller signing is unavailable or failed") from error


def _new_signing_key() -> tuple[bytes, bytes]:
    with TemporaryDirectory(prefix="qvq-controller-key-") as temporary:
        private_path = Path(temporary) / "private.pem"
        public_path = Path(temporary) / "public.pem"
        _openssl(["genpkey", "-algorithm", "ED25519", "-out", str(private_path)])
        _openssl(["pkey", "-in", str(private_path), "-pubout", "-out", str(public_path)])
        return private_path.read_bytes(), public_path.read_bytes()


def _sign(private_key: bytes, payload: bytes) -> str:
    with TemporaryDirectory(prefix="qvq-controller-sign-") as temporary:
        key_path = Path(temporary) / "private.pem"
        message_path = Path(temporary) / "message.bin"
        signature_path = Path(temporary) / "signature.bin"
        key_path.write_bytes(private_key)
        message_path.write_bytes(payload)
        _openssl(
            [
                "pkeyutl",
                "-sign",
                "-inkey",
                str(key_path),
                "-rawin",
                "-in",
                str(message_path),
                "-out",
                str(signature_path),
            ]
        )
        return base64.b64encode(signature_path.read_bytes()).decode("ascii")


def verify_controller_signature(transcript: Mapping[str, Any]) -> bool:
    unsigned = dict(transcript)
    signature = unsigned.pop("controller_signature_ed25519", None)
    public_key = unsigned.get("controller_public_key_pem")
    if not isinstance(signature, str) or not isinstance(public_key, str):
        return False
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError):
        return False
    with TemporaryDirectory(prefix="qvq-controller-verify-") as temporary:
        public_path = Path(temporary) / "public.pem"
        message_path = Path(temporary) / "message.bin"
        signature_path = Path(temporary) / "signature.bin"
        public_path.write_text(public_key, encoding="ascii")
        message_path.write_bytes(_canonical(unsigned))
        signature_path.write_bytes(signature_bytes)
        try:
            _openssl(
                [
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(public_path),
                    "-rawin",
                    "-in",
                    str(message_path),
                    "-sigfile",
                    str(signature_path),
                ]
            )
        except RuntimeError:
            return False
    return True


def controller_authority_receipt(transcript: Mapping[str, Any]) -> dict[str, Any]:
    public_key = transcript.get("controller_public_key_pem")
    if not isinstance(public_key, str):
        raise TypeError("controller transcript lacks a public key")
    return {
        "schema": "qvq-acceptance-controller-authority-v1",
        "controller_instance_id": transcript.get("controller_instance_id"),
        "run_nonce": transcript.get("run_nonce"),
        "controller_public_key_sha256": hashlib.sha256(public_key.encode("ascii")).hexdigest(),
        "transcript_sha256": hashlib.sha256(_canonical(transcript)).hexdigest(),
    }


def controller_environment() -> dict[str, str]:
    keys = ("RUN_NONCE", "STAGE", "STAGE_NONCE", "PROCESS_INSTANCE_ID", "EVENT_FD")
    values = {key.lower(): os.environ.get(_ENV_PREFIX + key) for key in keys}
    if any(value is None for value in values.values()):
        raise RuntimeError("acceptance stage was not spawned by the independent controller")
    return {key: str(value) for key, value in values.items()}


def emit_controller_measurement(stage: str, measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Send one nonce-bound measurement to the parent and wait for its pre-return acknowledgement."""

    authority = controller_environment()
    if authority["stage"] != stage:
        raise RuntimeError(f"controller authorized {authority['stage']!r}, not {stage!r}")
    event = {
        "run_nonce": authority["run_nonce"],
        "stage": stage,
        "stage_nonce": authority["stage_nonce"],
        "process_instance_id": authority["process_instance_id"],
        "measurement": dict(measurement),
    }
    descriptor = int(authority["event_fd"])
    with socket.socket(fileno=descriptor) as channel:
        channel.sendall(_canonical(event) + b"\n")
        acknowledgement = channel.makefile("rb").readline()
    expected = _canonical({"acknowledged_event_sha256": _sha256(event)}) + b"\n"
    if acknowledgement != expected:
        raise RuntimeError("controller did not acknowledge the live stage measurement")
    return event


class AcceptanceController:
    """Own unpredictable identities and observe child measurements/spawn/exit facts."""

    def __init__(self) -> None:
        self.run_nonce = _new_identity()
        self.controller_instance_id = _new_identity()
        self._private_key, public_key = _new_signing_key()
        self._public_key = public_key.decode("ascii")
        self._records: list[dict[str, Any]] = []

    def spawn_stage(
        self,
        stage: str,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if stage not in CONTROLLER_STAGES or any(record["stage"] == stage for record in self._records):
            raise ValueError(f"controller stage is invalid or replayed: {stage!r}")
        stage_nonce = _new_identity()
        process_instance_id = _new_identity()
        parent_channel, child_channel = socket.socketpair()
        environment = os.environ.copy()
        environment.update(
            {
                _ENV_PREFIX + "RUN_NONCE": self.run_nonce,
                _ENV_PREFIX + "STAGE": stage,
                _ENV_PREFIX + "STAGE_NONCE": stage_nonce,
                _ENV_PREFIX + "PROCESS_INSTANCE_ID": process_instance_id,
                _ENV_PREFIX + "EVENT_FD": str(child_channel.fileno()),
            }
        )
        spawned_ns = time.monotonic_ns()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            pass_fds=(child_channel.fileno(),),
        )
        child_channel.close()
        parent_channel.settimeout(timeout)
        try:
            with parent_channel.makefile("rb") as reader:
                raw_event = reader.readline()
            if not raw_event:
                raise RuntimeError(f"controller stage {stage!r} exited without a measurement")
            event = json.loads(raw_event)
            expected = {
                "run_nonce": self.run_nonce,
                "stage": stage,
                "stage_nonce": stage_nonce,
                "process_instance_id": process_instance_id,
            }
            if not isinstance(event, dict) or any(event.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"controller stage {stage!r} returned unauthorized identity data")
            event_sha256 = _sha256(event)
            received_ns = time.monotonic_ns()
            parent_channel.sendall(_canonical({"acknowledged_event_sha256": event_sha256}) + b"\n")
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            parent_channel.close()
        exit_code = process.wait(timeout=timeout)
        exited_ns = time.monotonic_ns()
        record = {
            "stage": stage,
            "stage_nonce": stage_nonce,
            "process_instance_id": process_instance_id,
            "pid": process.pid,
            "parent_pid": os.getpid(),
            "argv_sha256": hashlib.sha256(b"\0".join(os.fsencode(item) for item in command)).hexdigest(),
            "spawned_monotonic_ns": spawned_ns,
            "event_received_monotonic_ns": received_ns,
            "exited_monotonic_ns": exited_ns,
            "exit_code": exit_code,
            "event": event,
            "event_sha256": event_sha256,
            "previous_record_sha256": self._records[-1]["record_sha256"] if self._records else None,
        }
        record["record_sha256"] = _sha256(record)
        self._records.append(record)
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)
        return record

    def signed_transcript(self, *, require_complete: bool = True) -> dict[str, Any]:
        actual_stages = tuple(record["stage"] for record in self._records)
        if (require_complete and actual_stages != CONTROLLER_STAGES) or actual_stages != CONTROLLER_STAGES[: len(actual_stages)]:
            raise RuntimeError("controller transcript stages are incomplete or out of order")
        transcript = {
            "schema": CONTROLLER_SCHEMA,
            "controller_instance_id": self.controller_instance_id,
            "run_nonce": self.run_nonce,
            "controller_public_key_pem": self._public_key,
            "processes": list(self._records),
        }
        transcript["controller_signature_ed25519"] = _sign(self._private_key, _canonical(transcript))
        return transcript


__all__ = [
    "CONTROLLER_SCHEMA",
    "CONTROLLER_STAGES",
    "AcceptanceController",
    "controller_authority_receipt",
    "controller_environment",
    "emit_controller_measurement",
    "verify_controller_signature",
]
