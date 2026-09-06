# SPDX-License-Identifier: Apache-2.0
"""Quantization continuation composed from processor and packed-module state."""

import hashlib
import json
import random
import signal
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save

from ..adapter.adapter import Lora
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.paroquant import ParoLinear
from ..quantization.config import METHOD
from ..utils.device_telemetry import emit_device_telemetry
from ..utils.offload import set_submodule
from .awq_processor import AWQProcessor
from .checkpoint_devices import checkpoint_device_topology
from .checkpoint_modules import (
    is_packed_module,
    packed_module_spec,
    restore_packed_module,
)
from .checkpoint_store import CheckpointError
from .continuation import ContinuationCodec
from .eora_processor import EoraProcessor
from .gptq_processor import GPTQProcessor
from .paroquant_processor import ParoQuantProcessor
from .qqq_processor import QQQProcessor
from .weight_only_processor import WeightOnlyProcessor


@contextmanager
def checkpoint_session(config, model):
    from .checkpoint import CheckpointExtension

    if threading.current_thread() is not threading.main_thread():
        raise ValueError(
            "checkpoint quantization must run on the main thread for signal handling"
        )
    with CheckpointExtension(config, QuantizationCheckpointAdapter()) as extension:
        attempt = tempfile.mkdtemp(prefix="attempt-", dir=extension.store.root)
        model.quantize_config.offload_to_disk_path = attempt
        handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }

        def request_stop(signum, frame):
            extension.request_stop()

        try:
            for sig in handlers:
                signal.signal(sig, request_stop)
            yield extension
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def validate_checkpoint_support(model, *, embed_quant_config=None, adapter=None):
    config = model.quantize_config
    if (
        config.method not in set(METHOD)
        or not config.true_sequential
        or getattr(config, "gptaq", None) is not None
        or getattr(config, "foem", None) is not None
        or (adapter is not None and type(adapter) is not Lora)
        or config.lm_head
        or embed_quant_config is not None
        or config.dynamic
        or getattr(config, "rotation", None)
        or (config.adapter is not None and type(config.adapter) is not Lora)
    ):
        raise NotImplementedError(
            "checkpoint requires sequential quantization without dynamic exclusions, "
            "GPTAQ/FOEM, or embedding/lm_head quantization"
        )
    if model.model.config.model_type not in {"llama", "qwen3_moe"}:
        raise NotImplementedError(
            "checkpoint state adapters currently support llama and qwen3_moe"
        )
    if not config.offload_to_disk:
        raise ValueError("checkpoint currently requires offload_to_disk=True")


def _source_identity(model):
    root = Path(model.model_local_path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (root / "model.safetensors").exists():
        shards = ["model.safetensors"]
    else:
        raise NotImplementedError(
            "checkpoint requires a local safetensors source model"
        )
    result = {}
    for name in ["config.json", *shards]:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise CheckpointError("source shard escapes the model directory")
        with path.open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


class QuantizationCheckpointAdapter:
    VERSION = 3

    def bind(self, context):
        self.execution = context.execution
        if self.execution is None:
            raise CheckpointError("checkpoint requires execution placement state")
        self.model = context.model
        processor_types = tuple(type(processor) for processor in context.processors)
        supported_processors = {
            (GPTQProcessor,),
            (GPTQProcessor, EoraProcessor),
            (AWQProcessor,),
            (AWQProcessor, EoraProcessor),
            (WeightOnlyProcessor,),
            (QQQProcessor,),
            (QQQProcessor, EoraProcessor),
            (ParoQuantProcessor,),
            (ParoQuantProcessor, EoraProcessor),
        }
        if self.model.quantize_config.method == METHOD.EXL3:
            from .exllamav3_processor import EXL3Processor

            supported_processors.add((EXL3Processor,))
        if processor_types not in supported_processors:
            raise NotImplementedError(
                "checkpoint has no state adapter for this processor chain"
            )
        self.processors = context.processors
        self.processor = context.processors[0]
        self.shared_state = context.shared_state
        self.offload = Path(self.model.quantize_config.offload_to_disk_path)
        self._artifacts = {}
        config = self.model.quantize_config.to_dict()
        # Location is not an algorithm setting. Keep every other serialized
        # setting conservatively, including device/packing execution policy.
        config.get("meta", {}).pop("offload_to_disk_path", None)
        config.get("meta", {}).pop("telemetry", None)
        return {
            "adapter_version": self.VERSION,
            "device_topology": checkpoint_device_topology(
                self.execution.execution_device_pools()
            ),
            "source": _source_identity(self.model),
            "quantization": config,
            "calibration": hashlib.sha256(
                ContinuationCodec.dumps(
                    [processor.inputs_cache.unwrap() for processor in self.processors]
                )
            ).hexdigest(),
            "processors": [type(processor).__name__ for processor in self.processors],
            "packed_kernel": str(self.model.qlinear_kernel),
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }

    def capture(self):
        artifacts = {}
        specs = {}
        for name, module in self.model.model.named_modules():
            if not is_packed_module(module):
                continue
            expected_kernel = (
                ParoLinear
                if self.model.quantize_config.method == METHOD.PARO
                else self.model.qlinear_kernel
            )
            if (
                isinstance(module, BaseQuantLinear)
                and type(module) is not expected_kernel
            ):
                raise NotImplementedError(
                    f"unsupported checkpoint quantized module: {type(module).__name__}"
                )
            bundle = self._artifacts.get(
                name, self.offload / name / "module.safetensors"
            )
            if bundle.exists():
                artifacts[name] = bundle
            else:
                artifacts[name] = save(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in module.state_dict().items()
                    }
                )
            specs[name] = packed_module_spec(module, self.model.quantize_config)
        np_state = np.random.get_state()
        processor_states = [
            processor.continuation_state_dict() for processor in self.processors
        ]
        # Scaling methods also mutate non-quantized weights (e.g. layer norms).
        # Preserve every materialized direct parameter/buffer, including device
        # tags; untouched lazy meta tensors remain backed by the source model.
        dense = {}
        dense_aliases = {}
        tensor_owners = {}
        for name, module in self.model.model.named_modules():
            if any(name == packed or name.startswith(packed + ".") for packed in specs):
                continue
            values = {
                key: value
                for key, value in (
                    *module.named_parameters(recurse=False),
                    *module.named_buffers(recurse=False),
                )
                if value.device.type != "meta"
            }
            if values:
                dense[name] = values
                for key, value in values.items():
                    location = (name, key)
                    owner = tensor_owners.setdefault(id(value), location)
                    if owner != location:
                        dense_aliases[f"{name}.{key}"] = owner
        with self.processor.lock:
            state = {
                "version": self.VERSION,
                "shared_state": self.shared_state,
                "processors": processor_states,
                "dense": dense,
                "dense_aliases": dense_aliases,
                "specs": specs,
                "execution": self.execution.execution_state_dict(),
                "rng": torch.random.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_initialized()
                else [],
                "python_rng": random.getstate(),
                "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:]),
            }
            # The caller serializes immediately while the boundary is quiescent;
            # preserve source device tags until the codec makes CPU-owned copies.
        emit_device_telemetry(
            "checkpoint_execution_captured",
            execution=state["execution"],
            cuda_rng_indices=list(range(len(state["cuda_rng"]))),
        )
        return state, artifacts

    def committed(self, artifacts):
        # Completed modules are immutable for the remainder of quantization.
        # Reuse store objects, not copies of earlier offload bundles.
        self._artifacts = artifacts

    def restore(self, state, artifacts):
        if state["version"] != self.VERSION or set(state["specs"]) != set(artifacts):
            raise CheckpointError("incompatible quantization continuation schema")
        if len(state["processors"]) != len(self.processors):
            raise CheckpointError("checkpoint processor count differs")
        config = self.model.quantize_config
        self.execution.load_execution_state_dict(state["execution"])
        actual_execution = self.execution.execution_state_dict()
        matched = actual_execution == state["execution"]
        emit_device_telemetry(
            "checkpoint_execution_restored",
            expected=state["execution"],
            actual=actual_execution,
            matched=matched,
        )
        if not matched:
            raise CheckpointError("checkpoint execution placement restore mismatch")
        restored = {}
        dense_targets = []
        for location, owner in state.get("dense_aliases", {}).items():
            if (
                not isinstance(location, str)
                or "." not in location
                or not isinstance(owner, (list, tuple))
                or len(owner) != 2
                or not all(isinstance(part, str) for part in owner)
            ):
                raise CheckpointError("invalid dense tensor alias")
            name, key = location.rsplit(".", 1)
            owner_name, owner_key = owner
            value = state["dense"].get(name, {}).get(key)
            owner_value = state["dense"].get(owner_name, {}).get(owner_key)
            if (
                not isinstance(value, torch.Tensor)
                or not isinstance(owner_value, torch.Tensor)
                or value.device != owner_value.device
                or value.dtype != owner_value.dtype
                or value.shape != owner_value.shape
                or not torch.equal(
                    value.contiguous().reshape(-1).view(torch.uint8),
                    owner_value.contiguous().reshape(-1).view(torch.uint8),
                )
            ):
                raise CheckpointError("checkpoint dense tensor alias differs")
        for name, values in state["dense"].items():
            original = self.model.model.get_submodule(name)
            for key, value in values.items():
                target = getattr(original, key, None)
                if (
                    not isinstance(target, torch.Tensor)
                    or target.shape != value.shape
                    or target.dtype != value.dtype
                ):
                    raise CheckpointError(
                        f"checkpoint dense tensor schema differs: {name}.{key}"
                    )
                dense_targets.append(
                    (
                        original,
                        key,
                        value,
                        isinstance(target, torch.nn.Parameter),
                        target.requires_grad,
                    )
                )
        # Validate every module and bundle before replacing anything in the model.
        for name, spec in state["specs"].items():
            if any(
                not part or not part.replace("_", "").isalnum()
                for part in name.split(".")
            ):
                raise CheckpointError("invalid checkpoint module name")
            original = self.model.model.get_submodule(name)
            if (
                getattr(original, "in_features", None) != spec["in_features"]
                or getattr(original, "out_features", None) != spec["out_features"]
            ):
                raise CheckpointError(f"checkpoint module dimensions differ: {name}")
            module = restore_packed_module(
                spec,
                name=name,
                config=config,
                kernel=self.model.qlinear_kernel,
                lm_head_name=self.model.lm_head,
            )
            expected = module.state_dict()
            with safe_open(artifacts[name], framework="pt", device="cpu") as tensors:
                if set(expected) != set(tensors.keys()) or any(
                    value.shape != tensors.get_tensor(key).shape
                    or value.dtype != tensors.get_tensor(key).dtype
                    for key, value in expected.items()
                ):
                    raise CheckpointError(f"checkpoint tensor schema differs: {name}")
                if spec.get("kind") == "exl3":
                    # EXL3's writer reads scalar codebook metadata from its
                    # buffers. Its normal boundary state is materialized CPU,
                    # unlike the disk-offloaded BaseQuantLinear lifecycle.
                    module.load_state_dict(
                        {key: tensors.get_tensor(key) for key in tensors.keys()},  # noqa: SIM118 -- safe_open is not iterable
                        assign=True,
                    )
            restored[name] = module
        # These files are disposable per-attempt indexes, never checkpoint
        # generations. Packed modules remain meta until the normal save path
        # resolves their offload references; no original-weight replay occurs.
        for name, module in restored.items():
            directory = self.offload / name
            directory.mkdir(parents=True, exist_ok=True)
            bundle = artifacts[name]
            with bundle.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                header = json.loads(stream.read(header_size))
            index = {}
            for key, tensor in module.state_dict().items():
                start, end = header[key]["data_offsets"]
                index[key] = {
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                    "safetensors_file": str(bundle.absolute()),
                    "weight_name": key,
                    "data_offsets": [8 + header_size + start, 8 + header_size + end],
                }
            (directory / "index.json").write_text(json.dumps(index))
            set_submodule(self.model.model, name, module)
        for module, key, value, parameter, requires_grad in dense_targets:
            setattr(
                module,
                key,
                torch.nn.Parameter(value, requires_grad=requires_grad)
                if parameter
                else value,
            )
        for location, (owner_name, owner_key) in state.get("dense_aliases", {}).items():
            name, key = location.rsplit(".", 1)
            owner = self.model.model.get_submodule(owner_name)
            setattr(
                self.model.model.get_submodule(name), key, getattr(owner, owner_key)
            )
        for processor, processor_state in zip(self.processors, state["processors"]):
            processor.load_continuation_state_dict(processor_state)
        self.shared_state.clear()
        self.shared_state.update(state["shared_state"])
        torch.random.set_rng_state(state["rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        algorithm, keys, *rest = state["numpy_rng"]
        np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), *rest))
        actual_numpy = np.random.get_state()
        actual_cuda = torch.cuda.get_rng_state_all() if state["cuda_rng"] else []
        rng_matched = (
            torch.equal(torch.random.get_rng_state(), state["rng"])
            and len(actual_cuda) == len(state["cuda_rng"])
            and all(
                torch.equal(actual, expected)
                for actual, expected in zip(actual_cuda, state["cuda_rng"])
            )
            and random.getstate() == state["python_rng"]
            and (actual_numpy[0], actual_numpy[1].tolist(), *actual_numpy[2:])
            == state["numpy_rng"]
        )
        emit_device_telemetry(
            "checkpoint_rng_restored",
            matched=rng_matched,
            cuda_rng_indices=list(range(len(actual_cuda))),
        )
        if not rng_matched:
            raise CheckpointError("checkpoint RNG state restore mismatch")
        self._artifacts = artifacts
        emit_device_telemetry(
            "checkpoint_adapter_restored",
            packed_module_count=len(restored),
            packed_tensor_devices=sorted(
                {
                    str(tensor.device)
                    for module in restored.values()
                    for tensor in module.state_dict().values()
                }
            ),
            packed_storage="disk"
            if all(spec.get("kind") != "exl3" for spec in state["specs"].values())
            else "mixed",
            cuda_rng_indices=list(range(len(state["cuda_rng"]))),
        )


# Compatibility for callers of the original GPTQ-only adapter.
GPTQCheckpointAdapter = QuantizationCheckpointAdapter
