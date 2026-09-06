# SPDX-License-Identifier: Apache-2.0
"""Initial, deliberately bounded adapter for sequential GPTQ continuation."""

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

from ..nn_modules.qlinear import BaseQuantLinear
from ..quantization.config import METHOD, resolve_quant_format
from ..utils.offload import set_submodule
from .checkpoint_devices import checkpoint_device_topology
from .checkpoint_store import CheckpointError
from .continuation import ContinuationCodec
from .gptq_processor import GPTQProcessor


@contextmanager
def checkpoint_session(config, model):
    from .checkpoint import CheckpointExtension

    if threading.current_thread() is not threading.main_thread():
        raise ValueError(
            "checkpoint quantization must run on the main thread for signal handling"
        )
    with CheckpointExtension(config, GPTQCheckpointAdapter()) as extension:
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
        config.method != METHOD.GPTQ
        or not config.true_sequential
        or config.gptaq is not None
        or config.foem is not None
        or adapter is not None
        or config.lm_head
        or embed_quant_config is not None
        or config.dynamic
        or config.rotation
        or config.adapter is not None
    ):
        raise NotImplementedError(
            "checkpoint currently supports sequential GPTQ without adapters, dynamic exclusions, "
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


class GPTQCheckpointAdapter:
    VERSION = 2

    def bind(self, context):
        self.execution = context.execution
        if self.execution is None:
            raise CheckpointError("checkpoint requires execution placement state")
        self.model = context.model
        if (
            len(context.processors) != 1
            or type(context.processors[0]) is not GPTQProcessor
        ):
            raise NotImplementedError("checkpoint requires exactly one GPTQ processor")
        self.processor = context.processors[0]
        self.shared_state = context.shared_state
        self.offload = Path(self.model.quantize_config.offload_to_disk_path)
        self._artifacts = {}
        config = self.model.quantize_config.to_dict()
        # Location is not an algorithm setting. Keep every other serialized
        # setting conservatively, including device/packing execution policy.
        config.get("meta", {}).pop("offload_to_disk_path", None)
        return {
            "adapter_version": self.VERSION,
            "device_topology": checkpoint_device_topology(
                self.execution.execution_device_pools()
            ),
            "source": _source_identity(self.model),
            "quantization": config,
            "calibration": hashlib.sha256(
                ContinuationCodec.dumps(self.processor.inputs_cache.unwrap())
            ).hexdigest(),
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }

    def capture(self):
        artifacts = {}
        specs = {}
        for name, module in self.model.model.named_modules():
            if not isinstance(module, BaseQuantLinear):
                continue
            if type(module) is not self.model.qlinear_kernel:
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
            specs[name] = {
                "bits": module.bits,
                "group_size": module.group_size,
                "desc_act": module.desc_act,
                "sym": module.sym,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
            }
        np_state = np.random.get_state()
        with self.processor.lock:
            state = {
                "version": self.VERSION,
                "cache": self.processor.inputs_cache.unwrap(),
                "shared_state": self.shared_state,
                "log": self.processor.log,
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
        return state, artifacts

    def committed(self, artifacts):
        # Completed modules are immutable for the remainder of quantization.
        # Reuse store objects, not copies of earlier offload bundles.
        self._artifacts = artifacts

    def restore(self, state, artifacts):
        if state["version"] != self.VERSION or set(state["specs"]) != set(artifacts):
            raise CheckpointError("incompatible GPTQ continuation schema")
        config = self.model.quantize_config
        self.execution.load_execution_state_dict(state["execution"])
        restored = {}
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
            with torch.device("meta"):
                module = self.model.qlinear_kernel(
                    **spec,
                    pack_dtype=config.pack_dtype,
                    name=name,
                    lm_head_name=self.model.lm_head,
                    format=resolve_quant_format(config.format, config.method),
                    register_buffers=True,
                )
            expected = module.state_dict()
            with safe_open(artifacts[name], framework="pt", device="cpu") as tensors:
                if set(expected) != set(tensors.keys()) or any(
                    value.shape != tensors.get_tensor(key).shape
                    or value.dtype != tensors.get_tensor(key).dtype
                    for key, value in expected.items()
                ):
                    raise CheckpointError(f"checkpoint tensor schema differs: {name}")
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
        with self.processor.lock:
            self.processor.receive_input_cache(state["cache"])
            self.processor.log = state["log"]
        self.shared_state.clear()
        self.shared_state.update(state["shared_state"])
        torch.random.set_rng_state(state["rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        algorithm, keys, *rest = state["numpy_rng"]
        np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), *rest))
        self._artifacts = artifacts
