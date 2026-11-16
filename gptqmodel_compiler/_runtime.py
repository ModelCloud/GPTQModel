# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pure-Torch prelude copied into each standalone generated module.

No architecture class or compiler interpreter executes model operations here.
The generated _execute method is straight-line Torch code. Request publication,
validation and checkpoint IO are deliberately outside its fullgraph region.
"""

import math
import re
import threading

import torch
from torch import nn
from torch.nn import functional as F


MANIFEST: dict = {}  # Replaced by the verified literal in each generated module.


def _check_torch():
    if not re.fullmatch(r"2\.15\.\d+(?:\+[\w.]+)?", torch.__version__):
        raise RuntimeError(f"generated code requires stable PyTorch 2.15.x; found {torch.__version__}")


def _frequencies(d, a, device):
    f = 1.0 / (a["theta"] ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    wavelength = 2 * math.pi / f
    low, high = a["original_context"] / a["low_frequency"], a["original_context"] / a["high_frequency"]
    smooth = (a["original_context"] / wavelength - a["low_frequency"]) / (a["high_frequency"] - a["low_frequency"])
    middle = (1 - smooth) * f / a["factor"] + smooth * f
    return torch.where(wavelength < high, f, torch.where(wavelength > low, f / a["factor"], middle))


def _rope(q, k, positions, frequency, nq, nkv, d):
    phase = positions.float()[:, None] * frequency[None, :]
    phase = torch.cat((phase, phase), dim=-1)
    cos, sin = phase.cos().to(q.dtype)[None, None], phase.sin().to(q.dtype)[None, None]
    q = q.reshape(q.shape[0], q.shape[1], nq, d).transpose(1, 2)
    k = k.reshape(k.shape[0], k.shape[1], nkv, d).transpose(1, 2)
    qrot = torch.cat((-q[..., d // 2 :], q[..., : d // 2]), dim=-1)
    krot = torch.cat((-k[..., d // 2 :], k[..., : d // 2]), dim=-1)
    return q * cos + qrot * sin, k * cos + krot * sin


def _rms(x, weight, eps):
    z = x.float()
    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    return weight * z.to(x.dtype)


def _attention(q, k, v, positions, optimized):
    output_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    mask = torch.arange(k.shape[-2], device=q.device)[None, :] <= positions[:, None]
    mask = mask[None, None]
    if optimized:
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, enable_gqa=True)
    else:
        groups = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        score = score.masked_fill(~mask, float("-inf"))
        p = torch.softmax(score, dim=-1)
        a = p @ v
    return a.transpose(1, 2).reshape(q.shape[0], q.shape[2], q.shape[1] * q.shape[-1]).to(output_dtype)


class _Linear(nn.Linear):
    def reset_parameters(self):
        # Checkpoint loading is explicit; do not initialize a billion unused values.
        pass


class RequestState:
    """All cache resources are request-owned; model instances contain no KV state."""

    def __init__(self, program_id, batch, cache, signature, length=0, phase="prefill"):
        self.program_id, self.batch, self.cache = program_id, batch, tuple(cache)
        self.signature, self.length, self.phase = signature, length, phase
        self._lock = threading.Lock()

    def fork(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot snapshot an executing request")
        try:
            return RequestState(
                self.program_id,
                self.batch,
                tuple(x.clone() for x in self.cache),
                self.signature,
                self.length,
                self.phase,
            )
        finally:
            self._lock.release()


class _ModelBase(nn.Module):
    def __init__(self, device="cpu", compile_graph=None):
        super().__init__()
        _check_torch()
        self.projections, self.weights = nn.ModuleDict(), nn.ParameterDict()
        self._loaded, self._compiled = False, None
        self._execution_lock = threading.Lock()
        dtype = getattr(torch, MANIFEST["dtype"])
        for item in MANIFEST["storage"]:
            key, shape = item["key"], item["shape"]
            if item["kind"] == "linear":
                self.projections[key] = _Linear(shape[1], shape[0], bias=False, device=device, dtype=dtype)
                self.projections[key].requires_grad_(False)
            else:
                self.weights[key] = nn.Parameter(torch.empty(shape, device=device, dtype=dtype), requires_grad=False)
        for alias in MANIFEST["linear_aliases"]:
            weight = self.parameter_view(alias["parameter"])
            module = _Linear(weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=dtype)
            module.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight, requires_grad=False)
            self.projections[alias["key"]] = module
        for key, a in MANIFEST["rope"].items():
            self.register_buffer(key, _frequencies(a["head_dim"], a, device), persistent=False)
        self.eval()
        if compile_graph is None:
            compile_graph = MANIFEST["target"] == "optimized"
        if compile_graph:
            self.compile_graph()

    def parameter_view(self, name):
        b = MANIFEST["bindings"][name]
        if b["kind"] == "parameter":
            return self.weights[b["key"]]
        weight = self.projections[b["key"]].weight
        return weight[b["start"] : b["stop"]]

    def _signature(self):
        # A stale prefix must not survive weight mutation, replacement or a move.
        return tuple(
            (id(p), p._version, tuple(p.shape), tuple(p.stride()), str(p.dtype), str(p.device), p.data_ptr())
            for p in self.parameters()
        )

    @torch.no_grad()
    def load_weights(self, weights, strict=True):
        expected = MANIFEST["parameters"]
        missing = [name for name, p in expected.items() if not p["alias_of"] and name not in weights]
        extra = set(weights) - set(expected)
        if missing or (strict and extra):
            raise ValueError(f"checkpoint keys mismatch: missing={missing}, extra={sorted(extra)}")
        for name in weights:
            if name not in expected:
                continue
            tensor, p = weights[name], expected[name]
            if tuple(tensor.shape) != tuple(p["shape"]) or tensor.dtype not in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ):
                raise ValueError(f"checkpoint tensor mismatch: {name}")
            if p["alias_of"] and not torch.equal(tensor, weights[p["alias_of"]]):
                raise ValueError(f"tied checkpoint tensors differ: {name}")
        self._loaded = False
        for name, p in expected.items():
            if not p["alias_of"]:
                self.parameter_view(name).copy_(weights[name])
        self._loaded = True
        return self

    @torch.no_grad()
    def initialize_for_test(self, seed=0):
        generator = torch.Generator(device=next(self.parameters()).device).manual_seed(seed)
        for name, p in MANIFEST["parameters"].items():
            if not p["alias_of"]:
                value = self.parameter_view(name)
                if value.ndim == 1:
                    value.fill_(1)
                else:
                    value.normal_(mean=0, std=0.02, generator=generator)
        self._loaded = True
        return self

    def compile_graph(self):
        if MANIFEST["target"] != "optimized":
            raise ValueError("eager target must remain defused and hookable for quantization")
        self._compiled = torch.compile(self._execute, backend="inductor", fullgraph=True, dynamic=True)
        return self

    def new_request(self, batch_size=1, phase="prefill"):
        if not self._loaded or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("load weights and provide a positive batch size before creating a request")
        if phase not in ("prefill", "decode", "verify", "calibrate"):
            raise ValueError("unsupported entry phase")
        weight = next(self.parameters())
        cache = []
        for r in MANIFEST["resources"]:
            cache.extend(
                torch.empty(batch_size, r["heads"], 0, r["head_dim"], device=weight.device, dtype=weight.dtype)
                for _ in range(2)
            )
        return RequestState(MANIFEST["program_id"], batch_size, cache, self._signature(), phase=phase)

    def _validate(self, tokens, state):
        if not self._loaded:
            raise ValueError("weights have not been loaded")
        weight = next(self.parameters())
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.ndim != 2:
            raise ValueError("tokens must be int64[B,T]")
        if tokens.device != weight.device or tokens.shape[0] != state.batch or tokens.shape[1] < 1:
            raise ValueError("token device/batch/chunk mismatch")
        if tokens.min().item() < 0 or tokens.max().item() >= MANIFEST["vocabulary"]:
            raise ValueError("token ID outside vocabulary")
        if state.program_id != MANIFEST["program_id"] or state.signature != self._signature():
            raise ValueError("request schema or weight version is stale")
        if state.phase not in ("prefill", "decode", "verify", "calibrate"):
            raise ValueError("illegal request phase")
        if type(state.length) is not int or state.length < 0 or len(state.cache) != 2 * len(MANIFEST["resources"]):
            raise ValueError("invalid cache extent/binding count")
        if state.phase == "decode" and tokens.shape[1] != 1:
            raise ValueError("decode entry requires one token; use prefill/verify for chunks")
        if weight.dtype != getattr(torch, MANIFEST["dtype"]):
            raise ValueError("parameter dtype changed: recompile the typed program")
        for item in MANIFEST["storage"]:
            value = self.projections[item["key"]].weight if item["kind"] == "linear" else self.weights[item["key"]]
            if (
                tuple(value.shape) != tuple(item["shape"])
                or value.dtype != weight.dtype
                or value.device != weight.device
            ):
                raise ValueError("parameter storage changed: recompile the typed program")
        for alias in MANIFEST["linear_aliases"]:
            value, source = self.projections[alias["key"]].weight, self.parameter_view(alias["parameter"])
            if value.shape != source.shape or value.data_ptr() != source.data_ptr():
                raise ValueError("tied projection alias was broken")
        for key in MANIFEST["rope"]:
            if getattr(self, key).dtype != torch.float32 or getattr(self, key).device != weight.device:
                raise ValueError("RoPE precision/placement changed: recompile the typed program")
        occupied = {p.untyped_storage().data_ptr() for p in self.parameters()}
        for index, r in enumerate(MANIFEST["resources"]):
            if state.length + tokens.shape[1] > r["capacity"]:
                raise ValueError(f"resource capacity exceeded: {r['name']}")
            for cache in state.cache[2 * index : 2 * index + 2]:
                if (
                    not isinstance(cache, torch.Tensor)
                    or cache.dtype != weight.dtype
                    or cache.device != weight.device
                    or tuple(cache.shape) != (state.batch, r["heads"], state.length, r["head_dim"])
                    or not cache.is_contiguous()
                ):
                    raise ValueError(f"resource dtype/shape/layout mismatch: {r['name']}")
                if cache.numel():
                    address = cache.untyped_storage().data_ptr()
                    if address in occupied:
                        raise ValueError("resource storage aliases another resource or parameter")
                    occupied.add(address)

    @torch.no_grad()
    def forward(self, tokens, state=None):
        if state is None:
            state = self.new_request(tokens.shape[0])
        if not isinstance(state, RequestState):
            raise ValueError("expected an explicit RequestState from this generated module")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("concurrent model execution is unqualified; use serialized calls")
        locked = state._lock.acquire(blocking=False)
        try:
            if not locked:
                raise RuntimeError("request is already executing")
            self._validate(tokens, state)
            positions = torch.arange(
                state.length, state.length + tokens.shape[1], device=tokens.device, dtype=torch.int64
            )
            logits, next_cache = (self._compiled or self._execute)(tokens, positions, state.cache)
            # Publish together, only after all operations succeed. Fork for verify/calibration.
            state.cache = tuple(x.contiguous() for x in next_cache)
            state.length += tokens.shape[1]
            return logits, state
        finally:
            if locked:
                state._lock.release()
            self._execution_lock.release()

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        result, state, current = [tokens], self.new_request(tokens.shape[0]), tokens
        for _ in range(max_new_tokens):
            logits, state = self(current, state)
            current = logits[:, -1, :].argmax(-1, keepdim=True)
            result.append(current)
            state.phase = "decode"
        return torch.cat(result, dim=1)
