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

CONTROLLER_SCHEMA = "qvq-acceptance-controller-v2"
CONTROLLER_STAGES = ("quantization_producer", "fresh_process_reload", "acceptance_evaluation")
_ENV_PREFIX = "GPTQMODEL_QVQ_CONTROLLER_"
VERIFIER_PRIVATE_KEY_ENV = "GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY"
PINNED_VERIFIER_PUBLIC_KEY = (
    Path(__file__).resolve().parents[2] / "configs" / "qwen3_8b_acceptance_verifier_public.pem"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _validate_required_command(stage: str, command: Sequence[str]) -> None:
    command = list(command)
    expected_script = _REPO_ROOT / "scripts" / (
        "qvq_quantize.py" if stage == CONTROLLER_STAGES[0] else "accept_qwen3_8b_qvq.py"
    )
    if len(command) < 3 or Path(command[1]).resolve() != expected_script:
        raise ValueError(f"controller {stage!r} command substituted the required executable")
    tail = command[2:]
    if stage == CONTROLLER_STAGES[0]:
        required = {
            "--model": "/monster/data/model/Qwen3-8B", "--calibration-rows": "512",
            "--yaqa-rows": "512", "--validation-rows": "512",
            "--quant-config": str(_REPO_ROOT / "configs" / "qwen3_8b_qvq_w2_acceptance.json"),
        }
        if "--verify-qwen3-acceptance-payload-parity" not in tail:
            raise ValueError("producer command lacks mandatory live parity mode")
    else:
        subcommand = "payload-hashes" if stage == CONTROLLER_STAGES[1] else "evaluate"
        if not tail or tail[0] != subcommand:
            raise ValueError(f"controller {stage!r} command substituted the required subcommand")
        required = {}
    for flag, expected in required.items():
        if tail.count(flag) != 1 or tail[tail.index(flag) + 1] != expected:
            raise ValueError(f"controller {stage!r} command has noncanonical {flag}")


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


def _pinned_public_key() -> bytes:
    try:
        value = PINNED_VERIFIER_PUBLIC_KEY.read_bytes()
    except OSError as error:
        raise RuntimeError("pinned Qwen3 acceptance verifier public key is unavailable") from error
    if not value.startswith(b"-----BEGIN PUBLIC KEY-----"):
        raise RuntimeError("pinned Qwen3 acceptance verifier public key is malformed")
    return value


def _load_trusted_signing_key() -> tuple[bytes, bytes]:
    configured = os.environ.get(VERIFIER_PRIVATE_KEY_ENV)
    if not configured:
        raise RuntimeError(f"trusted verifier signing authority is absent: set {VERIFIER_PRIVATE_KEY_ENV}")
    private_path = Path(configured)
    if not private_path.is_absolute() or not private_path.is_file():
        raise RuntimeError("trusted verifier private key must be an existing absolute external path")
    if private_path.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise RuntimeError("trusted verifier private key must not reside in the candidate repository")
    if private_path.stat().st_mode & 0o077:
        raise RuntimeError("trusted verifier private key permissions must be 0600 or stricter")
    private_key = private_path.read_bytes()
    with TemporaryDirectory(prefix="qvq-controller-key-check-") as temporary:
        derived_path = Path(temporary) / "derived-public.pem"
        _openssl(["pkey", "-in", str(private_path), "-pubout", "-out", str(derived_path)])
        derived_public = derived_path.read_bytes()
    pinned_public = _pinned_public_key()
    if derived_public != pinned_public:
        raise RuntimeError("trusted verifier private key does not match the pinned public trust root")
    return private_key, pinned_public


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
    if not isinstance(signature, str):
        return False
    try:
        public_key = _pinned_public_key()
    except RuntimeError:
        return False
    if unsigned.get("verifier_public_key_sha256") != hashlib.sha256(public_key).hexdigest():
        return False
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError):
        return False
    with TemporaryDirectory(prefix="qvq-controller-verify-") as temporary:
        public_path = Path(temporary) / "public.pem"
        message_path = Path(temporary) / "message.bin"
        signature_path = Path(temporary) / "signature.bin"
        public_path.write_bytes(public_key)
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
    public_key = _pinned_public_key()
    fingerprint = hashlib.sha256(public_key).hexdigest()
    if transcript.get("verifier_public_key_sha256") != fingerprint:
        raise ValueError("controller transcript does not identify the pinned verifier trust root")
    return {
        "schema": "qvq-acceptance-controller-trust-root-v2",
        "controller_instance_id": transcript.get("controller_instance_id"),
        "run_nonce": transcript.get("run_nonce"),
        "verifier_public_key_sha256": fingerprint,
        "transcript_sha256": hashlib.sha256(_canonical(transcript)).hexdigest(),
    }


def controller_environment() -> dict[str, str]:
    keys = (
        "RUN_NONCE", "CONTROLLER_INSTANCE_ID", "CONTROLLER_PID", "STAGE", "STAGE_NONCE",
        "PROCESS_INSTANCE_ID", "EVENT_FD",
    )
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
        "controller_instance_id": authority["controller_instance_id"],
        "controller_pid": int(authority["controller_pid"]),
        "stage": stage,
        "stage_nonce": authority["stage_nonce"],
        "process_instance_id": authority["process_instance_id"],
        "child_pid": os.getpid(),
        "child_parent_pid": os.getppid(),
        "measurement": dict(measurement),
    }
    descriptor = int(authority["event_fd"])
    with socket.socket(fileno=descriptor) as channel:
        channel.sendall(_canonical(event) + b"\n")
        acknowledgement = channel.makefile("rb").readline()
    try:
        acknowledged = json.loads(acknowledgement)
    except json.JSONDecodeError as error:
        raise RuntimeError("controller returned a malformed acknowledgement") from error
    if acknowledged.get("acknowledged_event_sha256") != _sha256(event) or (
        stage == CONTROLLER_STAGES[0]
        and acknowledged.get("validation_policy") != "qwen3-producer-pre-save-v1"
    ):
        raise RuntimeError("controller did not acknowledge the live stage measurement")
    return event


class AcceptanceController:
    """Own unpredictable identities and observe child measurements/spawn/exit facts."""

    def __init__(self) -> None:
        self.run_nonce = _new_identity()
        self.controller_instance_id = _new_identity()
        self._private_key, public_key = _load_trusted_signing_key()
        self._public_key_sha256 = hashlib.sha256(public_key).hexdigest()
        self.controller_pid = os.getpid()
        self.controller_parent_pid = os.getppid()
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
        _validate_required_command(stage, command)
        stage_nonce = _new_identity()
        process_instance_id = _new_identity()
        parent_channel, child_channel = socket.socketpair()
        environment = os.environ.copy()
        environment.update(
            {
                _ENV_PREFIX + "RUN_NONCE": self.run_nonce,
                _ENV_PREFIX + "CONTROLLER_INSTANCE_ID": self.controller_instance_id,
                _ENV_PREFIX + "CONTROLLER_PID": str(self.controller_pid),
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
                "controller_instance_id": self.controller_instance_id,
                "controller_pid": self.controller_pid,
                "stage": stage,
                "stage_nonce": stage_nonce,
                "process_instance_id": process_instance_id,
            }
            if not isinstance(event, dict) or any(event.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"controller stage {stage!r} returned unauthorized identity data")
            if event.get("child_pid") != process.pid or event.get("child_parent_pid") != self.controller_pid:
                raise RuntimeError(f"controller stage {stage!r} returned false parent/child process facts")
            event_sha256 = _sha256(event)
            received_ns = time.monotonic_ns()
            acknowledgement = {"acknowledged_event_sha256": event_sha256}
            if stage == CONTROLLER_STAGES[0]:
                from gptqmodel.utils.qvq_acceptance import (
                    validate_producer_pre_save_measurement,
                )

                validate_producer_pre_save_measurement(event["measurement"], event=event)
                acknowledgement["validation_policy"] = "qwen3-producer-pre-save-v1"
                acknowledgement["validated_before_save_monotonic_ns"] = time.monotonic_ns()
            parent_channel.sendall(_canonical(acknowledgement) + b"\n")
            acknowledged_ns = time.monotonic_ns()
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
            "argv": list(command),
            "argv_sha256": hashlib.sha256(b"\0".join(os.fsencode(item) for item in command)).hexdigest(),
            "spawned_monotonic_ns": spawned_ns,
            "event_received_monotonic_ns": received_ns,
            "acknowledgement_sent_monotonic_ns": acknowledged_ns,
            "exited_monotonic_ns": exited_ns,
            "exit_code": exit_code,
            "event": event,
            "event_sha256": event_sha256,
            "acknowledgement": acknowledgement,
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
            "controller_pid": self.controller_pid,
            "controller_parent_pid": self.controller_parent_pid,
            "verifier_public_key_sha256": self._public_key_sha256,
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
