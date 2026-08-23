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
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

CONTROLLER_SCHEMA = "qvq-acceptance-controller-v3"
CONTROLLER_STAGES = ("quantization_producer", "fresh_process_reload", "acceptance_evaluation")
_ENV_PREFIX = "GPTQMODEL_QVQ_CONTROLLER_"
VERIFIER_PRIVATE_KEY_ENV = "GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY"
TRUST_CONFIG_ENV = "GPTQMODEL_QVQ_TRUST_CONFIG"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_nofollow(path: Path, *, exact_mode: int | None = None) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"trusted file cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"trusted file is not regular: {path}")
        if before.st_uid != os.geteuid():
            raise RuntimeError(f"trusted file has unsafe ownership: {path}")
        if exact_mode is not None and stat.S_IMODE(before.st_mode) != exact_mode:
            raise RuntimeError(f"trusted file mode must be exactly {exact_mode:04o}: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"trusted file changed while being read: {path}")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def load_trust_config() -> dict[str, Any]:
    configured = os.environ.get(TRUST_CONFIG_ENV)
    if not configured or not Path(configured).is_absolute():
        raise RuntimeError(f"operator trust configuration is absent: set {TRUST_CONFIG_ENV} to an absolute path")
    if Path(configured).is_relative_to(_REPO_ROOT):
        raise RuntimeError("operator trust configuration must reside outside the candidate repository")
    raw, _identity = _read_nofollow(Path(configured), exact_mode=0o600)
    try:
        trust = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("operator trust configuration is malformed") from error
    required = {"schema", "verifier_public_key", "verifier_public_key_sha256", "python_executable",
                "python_executable_sha256", "openssl_executable", "openssl_executable_sha256", "stage_scripts"}
    if not isinstance(trust, dict) or set(trust) != required or trust.get("schema") != "qvq-acceptance-trust-v1":
        raise RuntimeError("operator trust configuration has an invalid closed schema")
    for path_key, digest_key in (("verifier_public_key", "verifier_public_key_sha256"),
                                 ("python_executable", "python_executable_sha256"),
                                 ("openssl_executable", "openssl_executable_sha256")):
        path = Path(trust[path_key])
        if path_key == "verifier_public_key" and path.is_relative_to(_REPO_ROOT):
            raise RuntimeError("operator verifier public key must reside outside the candidate repository")
        if not path.is_absolute() or not path.is_file() or _sha256_file(path) != trust[digest_key]:
            raise RuntimeError(f"operator-trusted {path_key} identity is absent or mismatched")
    scripts = trust["stage_scripts"]
    if not isinstance(scripts, dict) or set(scripts) != set(CONTROLLER_STAGES):
        raise RuntimeError("operator trust configuration lacks exact stage script identities")
    for stage, identity in scripts.items():
        path = Path(identity.get("path", "")) if isinstance(identity, dict) else Path("")
        if not path.is_absolute() or not path.is_file() or set(identity) != {"path", "sha256"} or _sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"operator-trusted {stage} script identity is absent or mismatched")
    return trust


def trusted_python_executable() -> str:
    return str(Path(load_trust_config()["python_executable"]).resolve())


def controller_dataset_evidence(values: Mapping[str, str]) -> dict[str, Any]:
    """Read frozen producer inputs and prove JSONL/manifest identity and disjointness independently."""

    evidence: dict[str, Any] = {}
    identity_sets: dict[str, set[str]] = {}
    content_sets: dict[str, set[str]] = {}
    for name in ("calibration", "yaqa", "validation"):
        source = Path(values[f"--{name}-dataset"])
        manifest_path = source.with_suffix(".manifest.json")
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"controller cannot read trusted {name} JSONL/manifest inputs") from error
        expected_split = "yaqa_tuning" if name == "yaqa" else name
        if (
            len(lines) != 512
            or manifest.get("schema_version") != 1
            or manifest.get("count") != 512
            or manifest.get("split") != expected_split
        ):
            raise RuntimeError(f"controller {name} input does not contain exactly 512 canonical rows")
        identities: set[str] = set()
        content_hashes: set[str] = set()
        samples = manifest.get("samples")
        if not isinstance(samples, list) or len(samples) != 512:
            raise RuntimeError(f"controller {name} manifest census is invalid")
        for ordinal, (line, sample) in enumerate(zip(lines, samples)):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"controller {name} JSONL row {ordinal} is invalid") from error
            identity = row.get("identity") if isinstance(row, dict) else None
            if not isinstance(identity, str) or not identity:
                raise RuntimeError(f"controller {name} JSONL row {ordinal} lacks identity")
            content_hash = hashlib.sha256(
                json.dumps(row.get("content"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if sample != {"ordinal": ordinal, "identity": identity, "content_sha256": content_hash}:
                raise RuntimeError(f"controller {name} manifest row {ordinal} does not match JSONL content")
            identities.add(identity)
            content_hashes.add(content_hash)
        if len(identities) != 512 or len(content_hashes) != 512:
            raise RuntimeError(f"controller {name} inputs contain duplicate identity or content")
        identity_sets[name] = identities
        content_sets[name] = content_hashes
        evidence[name] = {
            "source": str(source), "config": None, "split": "train", "row_start": 0, "rows": 512,
            "content_sha256": _sha256_file(source), "identity_manifest": str(manifest_path),
            "identity_manifest_sha256": _sha256_file(manifest_path), "manifest_verified": True,
        }
    for left_index, left in enumerate(identity_sets):
        for right in tuple(identity_sets)[left_index + 1:]:
            if identity_sets[left] & identity_sets[right] or content_sets[left] & content_sets[right]:
                raise RuntimeError(f"controller producer inputs are not disjoint: {left}/{right}")
    return evidence


def _observe_linux_process(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    try:
        stat_fields = (proc / "stat").read_text(encoding="ascii").split()
        executable = (proc / "exe").resolve(strict=True)
        cmdline = (proc / "cmdline").read_bytes()
    except OSError as error:
        raise RuntimeError(f"controller cannot independently inspect child PID {pid}") from error
    if not cmdline.endswith(b"\0"):
        raise RuntimeError(f"controller observed malformed cmdline for child PID {pid}")
    return {
        "pid": pid, "ppid": int(stat_fields[3]), "start_time_ticks": int(stat_fields[21]),
        "executable": str(executable), "executable_sha256": _sha256_file(executable),
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def validate_required_command(stage: str, command: Sequence[str]) -> dict[str, str]:
    """Require the exact trusted interpreter/script and a closed, ordered stage argv."""

    command = list(command)
    trust = load_trust_config()
    script_identity = trust["stage_scripts"].get(stage)
    if not script_identity or len(command) < 3:
        raise ValueError(f"controller stage is invalid: {stage!r}")
    if command[0] != str(Path(trust["python_executable"]).resolve()):
        raise ValueError(f"controller {stage!r} argv[0] is not the operator-trusted interpreter")
    if Path(command[1]).resolve() != Path(script_identity["path"]):
        raise ValueError(f"controller {stage!r} command substituted the required executable")
    tail = command[2:]
    if stage == CONTROLLER_STAGES[0]:
        option_names = (
            "--model", "--output", "--quant-config", "--calibration-dataset", "--calibration-rows",
            "--yaqa-dataset", "--yaqa-rows", "--validation-dataset", "--validation-rows", "--device",
        )
        expected_tail_length = len(option_names) * 2 + 1
        if len(tail) != expected_tail_length or tail[-1] != "--verify-qwen3-acceptance-payload-parity":
            raise ValueError("producer command has unknown, duplicate, missing, or extra flags")
        values = {name: tail[index * 2 + 1] for index, name in enumerate(option_names)}
        if any(tail[index * 2] != name for index, name in enumerate(option_names)):
            raise ValueError("producer command options are not in exact canonical order")
        fixed = {
            "--model": "/monster/data/model/Qwen3-8B",
            "--quant-config": str(_REPO_ROOT / "configs" / "qwen3_8b_qvq_w2_acceptance.json"),
            "--calibration-rows": "512", "--yaqa-rows": "512", "--validation-rows": "512",
        }
    else:
        subcommand = "payload-hashes" if stage == CONTROLLER_STAGES[1] else "evaluate"
        option_names = (
            ("--checkpoint", "--device", "--output")
            if stage == CONTROLLER_STAGES[1]
            else ("--dense-model", "--revision", "--checkpoint", "--manifest-dir", "--validation-jsonl",
                  "--held-out-diagnostics-jsonl", "--diverse-jsonl", "--device", "--maximum-bpw", "--score-min",
                  "--final-kl-max-nats", "--output")
        )
        if len(tail) != 1 + len(option_names) * 2 or tail[0] != subcommand:
            raise ValueError(f"controller {stage!r} command substituted the required subcommand")
        if any(tail[1 + index * 2] != name for index, name in enumerate(option_names)):
            raise ValueError(f"controller {stage!r} options are not exact and canonical")
        values = {name: tail[1 + index * 2 + 1] for index, name in enumerate(option_names)}
        fixed = {"--dense-model": "/monster/data/model/Qwen3-8B"} if stage == CONTROLLER_STAGES[2] else {}
    for flag, expected in fixed.items():
        if values.get(flag) != expected:
            raise ValueError(f"controller {stage!r} command has noncanonical {flag}")
    path_options = {
        "--output", "--quant-config", "--calibration-dataset", "--yaqa-dataset", "--validation-dataset",
        "--checkpoint", "--manifest-dir", "--validation-jsonl", "--held-out-diagnostics-jsonl", "--diverse-jsonl",
    }
    for flag in path_options & values.keys():
        if values[flag] != str(Path(values[flag]).expanduser().resolve()):
            raise ValueError(f"controller {stage!r} {flag} is not an absolute normalized path")
    return values


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _new_identity() -> str:
    return secrets.token_hex(32)


def _openssl(args: Sequence[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    executable = str(Path(load_trust_config()["openssl_executable"]).resolve())
    try:
        return subprocess.run(
            [executable, *args],
            input=input_bytes,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("OpenSSL Ed25519 controller signing is unavailable or failed") from error


def _pinned_public_key() -> bytes:
    trust = load_trust_config()
    value, _identity = _read_nofollow(Path(trust["verifier_public_key"]))
    if hashlib.sha256(value).hexdigest() != trust["verifier_public_key_sha256"]:
        raise RuntimeError("operator verifier public key fingerprint mismatches trusted deployment input")
    if not value.startswith(b"-----BEGIN PUBLIC KEY-----"):
        raise RuntimeError("pinned Qwen3 acceptance verifier public key is malformed")
    return value


def _load_trusted_signing_key() -> tuple[bytes, bytes, Path, os.stat_result]:
    configured = os.environ.get(VERIFIER_PRIVATE_KEY_ENV)
    if not configured:
        raise RuntimeError(f"trusted verifier signing authority is absent: set {VERIFIER_PRIVATE_KEY_ENV}")
    private_path = Path(configured)
    if not private_path.is_absolute():
        raise RuntimeError("trusted verifier private key must be an existing absolute external path")
    if private_path.is_relative_to(Path(__file__).resolve().parents[2]):
        raise RuntimeError("trusted verifier private key must not reside in the candidate repository")
    private_key, identity = _read_nofollow(private_path, exact_mode=0o600)
    with TemporaryDirectory(prefix="qvq-controller-key-check-") as temporary:
        private_copy = Path(temporary) / "stable-private.pem"
        derived_path = Path(temporary) / "derived-public.pem"
        private_copy.write_bytes(private_key)
        private_copy.chmod(0o600)
        _openssl(["pkey", "-in", str(private_copy), "-pubout", "-out", str(derived_path)])
        derived_public = derived_path.read_bytes()
    pinned_public = _pinned_public_key()
    if derived_public != pinned_public:
        raise RuntimeError("trusted verifier private key does not match the pinned public trust root")
    return private_key, pinned_public, private_path, identity


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
        "schema": "qvq-acceptance-controller-trust-root-v3",
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
        self._private_key, public_key, self._private_key_path, self._private_key_identity = _load_trusted_signing_key()
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
        command_values = validate_required_command(stage, command)
        controller_datasets = controller_dataset_evidence(command_values) if stage == CONTROLLER_STAGES[0] else None
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
            os_process = _observe_linux_process(process.pid)
            trust = load_trust_config()
            expected_cmdline = b"\0".join(os.fsencode(item) for item in command) + b"\0"
            if (
                os_process["ppid"] != self.controller_pid
                or os_process["executable"] != str(Path(trust["python_executable"]).resolve())
                or os_process["executable_sha256"] != trust["python_executable_sha256"]
                or os_process["cmdline_sha256"] != hashlib.sha256(expected_cmdline).hexdigest()
            ):
                raise RuntimeError(f"controller stage {stage!r} OS process facts do not match authority")
            event_sha256 = _sha256(event)
            received_ns = time.monotonic_ns()
            acknowledgement = {"acknowledged_event_sha256": event_sha256}
            if stage == CONTROLLER_STAGES[0]:
                from gptqmodel.utils.qvq_acceptance import (
                    validate_producer_pre_save_measurement,
                )

                validate_producer_pre_save_measurement(
                    event["measurement"], event=event, controller_datasets=controller_datasets
                )
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
            "os_process": os_process,
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
            "controller_datasets": controller_datasets,
            "previous_record_sha256": self._records[-1]["record_sha256"] if self._records else None,
        }
        record["record_sha256"] = _sha256(record)
        self._records.append(record)
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)
        return record

    def signed_transcript(self, *, require_complete: bool = True) -> dict[str, Any]:
        try:
            current = os.lstat(self._private_key_path)
        except OSError as error:
            raise RuntimeError("trusted verifier private-key path changed after controller initialization") from error
        expected = self._private_key_identity
        if (
            stat.S_IMODE(current.st_mode) != 0o600
            or not stat.S_ISREG(current.st_mode)
            or current.st_uid != os.geteuid()
            or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise RuntimeError("trusted verifier private-key path changed after controller initialization")
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
            "trust_config_sha256": hashlib.sha256(_canonical(load_trust_config())).hexdigest(),
            "controller_datasets": (
                self._records[0]["controller_datasets"] if self._records else None
            ),
            "processes": list(self._records),
        }
        transcript["controller_signature_ed25519"] = _sign(self._private_key, _canonical(transcript))
        return transcript


__all__ = [
    "CONTROLLER_SCHEMA",
    "CONTROLLER_STAGES",
    "AcceptanceController",
    "controller_authority_receipt",
    "controller_dataset_evidence",
    "controller_environment",
    "emit_controller_measurement",
    "load_trust_config",
    "trusted_python_executable",
    "validate_required_command",
    "verify_controller_signature",
]
