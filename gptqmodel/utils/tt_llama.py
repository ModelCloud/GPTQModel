# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Device-resident Llama decode for Tenstorrent block-float checkpoints.

Everything after the token id stays on the chip: embedding lookup, RMSNorm,
fused QKV projection, HF-style RoPE, KV-cache update, flash-decode SDPA,
gate/up with SiLU fused into the multiply, down projection, residual adds, the
final norm, the tied bf8 head and (optionally) argmax.  One step is a fixed op
graph, so it can be captured once as a trace and replayed.

Weights must come from a checkpoint written with ``block_axis="out"``
(:func:`~gptqmodel.utils.tt_blockfloat_quant.gptq_solve_blockfloat`): the
operand of ``ttnn.linear(x, W^T)`` is ``[in, out]`` and a bf4_b/bf8_b tile shares
one exponent per 16 values of the last dim, so only an out-axis checkpoint
uploads without re-quantization.  Upload drift is measured per tensor and must
be zero.

Measured on a Blackhole p300c, Llama-3.2-1B-Instruct (bf8 attention/down,
bf4 gate/up, bf8 tied head), 32 users per chip, traced:

* the matmul core grid is the largest lever: ``ttnn``'s default grid made
  ``down_proj`` 0.147 ms and the tuned 4x8 grid 0.048 ms;
* HiFi4 costs ~2 % of the step against HiFi2 and lifts teacher-forced top-1
  agreement with an fp32 reference from 96.8 % to 98.1 % (CPU bf16 itself:
  98.4 %);
* gate|up fused into one matmul measured no faster than two tuned matmuls.

Multi-card serving is data parallel with **one process per card**: a 1xN
``MeshDevice`` running the same trace measured ~2x slower per step than a single
chip on this software stack, while four independent processes scaled linearly.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

from .ttnn_runtime import import_ttnn


TILE = 32
#: Users per chip.  The decode head-shuffle ops require exactly one tile row.
DECODE_BATCH = 32

#: Matmul core grids (``y``, ``x``) measured fastest for M=32 decode on a
#: p300c (110 Tensix cores).  Keyed by projection.
P300C_DECODE_GRIDS = {
    "qkv": (2, 8),
    "o": (2, 8),
    "gate": (4, 11),
    "up": (4, 11),
    "down": (4, 8),
    "head": (8, 11),
}


@dataclass(frozen=True)
class LlamaDims:
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int
    eps: float


def llama_dims(config) -> LlamaDims:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return LlamaDims(
        hidden=int(config.hidden_size),
        intermediate=int(config.intermediate_size),
        heads=int(config.num_attention_heads),
        kv_heads=int(config.num_key_value_heads),
        head_dim=int(head_dim),
        layers=int(config.num_hidden_layers),
        vocab=int(config.vocab_size),
        eps=float(config.rms_norm_eps),
    )


def validate_llama_config(config) -> LlamaDims:
    """Reject geometries the decode op chain cannot run, before touching a chip."""

    if getattr(config, "model_type", None) != "llama":
        raise NotImplementedError(f"TT Llama decode supports model_type 'llama', got {config.model_type!r}")
    if getattr(config, "attention_bias", False) or getattr(config, "mlp_bias", False):
        raise NotImplementedError("TT Llama decode does not implement projection biases")
    if not getattr(config, "tie_word_embeddings", False):
        raise NotImplementedError("TT Llama decode expects a tied embedding/head")
    d = llama_dims(config)
    if d.head_dim * d.heads != d.hidden:
        raise NotImplementedError("head_dim * num_attention_heads must equal hidden_size")
    if d.head_dim != TILE and d.head_dim % (2 * TILE):
        raise NotImplementedError(f"head_dim must be 32 or a multiple of 64, got {d.head_dim}")
    if d.heads > TILE or d.kv_heads > TILE:
        raise NotImplementedError("decode head ops support at most 32 query and 32 KV heads")
    for name, value in (("hidden_size", d.hidden), ("intermediate_size", d.intermediate)):
        if value % TILE:
            raise NotImplementedError(f"{name}={value} must be a multiple of {TILE}")
    return d


def hf_rope_tables(config, max_seq: int):
    """``cos``/``sin`` ``[max_seq, head_dim]`` exactly as transformers computes them.

    Built by the model's own rotary module so Llama-3 frequency scaling is
    reproduced rather than re-derived.
    """

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary = LlamaRotaryEmbedding(config)
    probe = torch.zeros(1, 1, dtype=torch.float32)
    cos, sin = rotary(probe, torch.arange(max_seq).unsqueeze(0))
    return cos[0].float(), sin[0].float()


# --------------------------------------------------------------------------- weights


def load_tt_llama_weights(checkpoint_dir: str):
    """Read an out-axis block-float Llama checkpoint as device operands.

    Returns ``(config, weights)`` where ``weights`` maps

    * ``model.layers.N.<proj>`` -> ``(W^T [in, out] float32 grid values, bits)``,
    * ``"head"`` -> ``(E^T [hidden, vocab], bits)`` and ``"embed"`` -> ``(E, bits)``,
    * norm names -> ``(weight, 16)``.

    Nothing is re-quantized: codes are expanded exactly, and the shapes are the
    stored ``[in, out]`` matrices.
    """

    from transformers import AutoConfig

    from .tt_blockfloat_quant import _unpack_codes, load_hf_state_dict, recompose_blockfloat

    config = AutoConfig.from_pretrained(checkpoint_dir)
    qcfg = getattr(config, "quantization_config", None) or {}
    if not isinstance(qcfg, dict):
        qcfg = dict(qcfg)
    if qcfg.get("quant_method") != "tt-blockfloat":
        raise ValueError(f"{checkpoint_dir} is not a tt-blockfloat checkpoint")
    axis = qcfg.get("block_axis", "in")
    if axis != "out":
        raise ValueError(
            f"{checkpoint_dir} stores exponent blocks along {axis!r}. ttnn.linear "
            "consumes W^T [in, out] blocked along out_features, so this checkpoint "
            "would be re-quantized on upload (3-5 % bf4 weight error measured). "
            "Re-quantize it with block_axis='out' (scripts/tt_quantize_blockfloat.py "
            "--block-axis out --act-order)."
        )
    validate_llama_config(config)
    widths = dict(qcfg.get("bits_per_layer") or {})
    default_bits = int(qcfg.get("bits", 4))
    state = load_hf_state_dict(checkpoint_dir)

    weights = {}
    for name in state:
        if not name.endswith(".codes"):
            continue
        prefix = name[: -len(".codes")]
        bits = int(widths.get(prefix, default_bits))
        shape = tuple(int(v) for v in state[f"{prefix}.code_shape"])
        codes = _unpack_codes(state[name], bits, shape)
        stored = recompose_blockfloat(codes, state[f"{prefix}.exponents"], bits=bits)
        if prefix.endswith("embed_tokens"):
            weights["head"] = (stored, bits)               # E^T [hidden, vocab]
            weights["embed"] = (stored.T.contiguous(), bits)  # lookup table E
        else:
            weights[prefix] = (stored, bits)
    for name, tensor in state.items():
        if name.endswith("norm.weight"):
            weights[name] = (tensor.to(torch.float32), 16)
    if "head" not in weights:
        raise ValueError(f"{checkpoint_dir} has no block-float embedding to tie the head to")
    return config, weights


# --------------------------------------------------------------------------- decoder


class TTLlamaDecoder:
    """Batched greedy decode for one chip (or a ``1xN`` mesh, data parallel).

    ``mesh`` is a ``ttnn`` device or mesh device.  Each chip serves
    :data:`DECODE_BATCH` users; on a mesh the weights are replicated and the
    users, KV cache and positions are sharded across chips.
    """

    def __init__(
        self,
        mesh,
        config,
        weights: dict,
        *,
        max_seq: int = 2048,
        fidelity: str = "HiFi4",
        kv_dtype: str = "bfloat16",
        k_chunk: Optional[int] = None,
        grids: Optional[dict] = None,
        check_drift: bool = True,
    ):
        ttnn = import_ttnn()
        self.ttnn = ttnn
        self.mesh = mesh
        self.config = config
        self.d = validate_llama_config(config)
        self.chips = int(mesh.get_num_devices()) if hasattr(mesh, "get_num_devices") else 1
        self.users = DECODE_BATCH * self.chips
        # SDPA-decode reads K/V in chunks; 256 measured fastest from ~pos 300 up,
        # 128 below.  The cache length must be a multiple of the chunk.
        self.k_chunk = int(k_chunk or (256 if max_seq >= 512 else 128))
        self.max_seq = int(math.ceil(max_seq / self.k_chunk) * self.k_chunk)
        self.kv_dtype = getattr(ttnn, kv_dtype)
        self.check_drift = check_drift
        self.drift: dict = {}
        grid_shapes = dict(P300C_DECODE_GRIDS, **(grids or {}))
        self.grids = {k: ttnn.CoreGrid(y=v[0], x=v[1]) for k, v in grid_shapes.items()}
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity),
            fp32_dest_acc_en=True, packer_l1_acc=True, math_approx_mode=False,
        )
        self.compute_sdpa = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False,
        )
        self.sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
            q_chunk_size=TILE, k_chunk_size=self.k_chunk, exp_approx_mode=False,
        )
        self._replicate = ttnn.ReplicateTensorToMesh(mesh) if self.chips > 1 else None
        self._shard_users = ttnn.ShardTensorToMesh(mesh, dim=1) if self.chips > 1 else None
        self._shard_rows = ttnn.ShardTensorToMesh(mesh, dim=0) if self.chips > 1 else None
        self._gather_users = ttnn.ConcatMeshToTensor(mesh, dim=2) if self.chips > 1 else None
        self._upload(weights)
        self._allocate_cache()
        self._upload_rope()
        self._trace = None

    # -- upload -----------------------------------------------------------

    def _put(self, tensor, dtype, name=None, layout=None):
        ttnn = self.ttnn
        layout = ttnn.TILE_LAYOUT if layout is None else layout
        device_tensor = ttnn.from_torch(
            tensor, dtype=dtype, layout=layout, device=self.mesh,
            mesh_mapper=self._replicate, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.check_drift and name is not None and dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b):
            composer = ttnn.ConcatMeshToTensor(self.mesh, dim=0) if self.chips > 1 else None
            back = ttnn.to_torch(device_tensor, mesh_composer=composer).float()
            back = back[: tensor.shape[0]].reshape(tensor.shape)
            self.drift[name] = float((back - tensor.float()).abs().max())
        return device_tensor

    def _blockfloat(self, bits: int):
        return self.ttnn.bfloat4_b if bits == 4 else self.ttnn.bfloat8_b

    def _norm(self, weight):
        return self._put(weight.reshape(1, 1, -1, TILE).float(), self.ttnn.bfloat16,
                         layout=self.ttnn.ROW_MAJOR_LAYOUT)

    def _upload(self, weights):
        ttnn = self.ttnn
        self.layers = []
        for index in range(self.d.layers):
            prefix = f"model.layers.{index}."
            q, q_bits = weights[prefix + "self_attn.q_proj"]
            k, k_bits = weights[prefix + "self_attn.k_proj"]
            v, v_bits = weights[prefix + "self_attn.v_proj"]
            if not q_bits == k_bits == v_bits:
                raise ValueError(f"layer {index}: q/k/v widths differ ({q_bits}, {k_bits}, {v_bits}); cannot fuse")
            o, o_bits = weights[prefix + "self_attn.o_proj"]
            g, g_bits = weights[prefix + "mlp.gate_proj"]
            u, u_bits = weights[prefix + "mlp.up_proj"]
            down, d_bits = weights[prefix + "mlp.down_proj"]
            self.layers.append({
                # Concatenating along out_features at tile multiples keeps every
                # 16-value exponent block whole, so the fused QKV is still exact.
                "qkv": self._put(torch.cat([q, k, v], dim=1), self._blockfloat(q_bits), prefix + "qkv"),
                "o": self._put(o, self._blockfloat(o_bits), prefix + "o_proj"),
                "gate": self._put(g, self._blockfloat(g_bits), prefix + "gate_proj"),
                "up": self._put(u, self._blockfloat(u_bits), prefix + "up_proj"),
                "down": self._put(down, self._blockfloat(d_bits), prefix + "down_proj"),
                "attn_norm": self._norm(weights[prefix + "input_layernorm.weight"][0]),
                "mlp_norm": self._norm(weights[prefix + "post_attention_layernorm.weight"][0]),
            })
        self.final_norm = self._norm(weights["model.norm.weight"][0])
        head, head_bits = weights["head"]
        self.head = self._put(head, self._blockfloat(head_bits), "lm_head")
        # The lookup table holds the same bf8 grid values in a row-major bf16
        # table: ``ttnn.embedding`` over a bf8 tile table is exact but measured
        # 2.05 ms per step against 18 us, and a tied table can only be on the
        # bf8 grid in one orientation -- the head's.
        embed, _ = weights["embed"]
        self.embed = self._put(embed.to(torch.bfloat16).reshape(1, 1, *embed.shape), ttnn.bfloat16,
                               layout=ttnn.ROW_MAJOR_LAYOUT)

    def _allocate_cache(self):
        ttnn = self.ttnn
        zeros = torch.zeros(self.users, self.d.kv_heads, self.max_seq, self.d.head_dim, dtype=torch.bfloat16)
        self.k_cache, self.v_cache = [], []
        for _ in range(self.d.layers):
            for cache in (self.k_cache, self.v_cache):
                cache.append(ttnn.from_torch(zeros, dtype=self.kv_dtype, layout=ttnn.TILE_LAYOUT,
                                             device=self.mesh, mesh_mapper=self._shard_rows))

    def _upload_rope(self):
        cos, sin = hf_rope_tables(self.config, self.max_seq)
        rm = self.ttnn.ROW_MAJOR_LAYOUT
        self.cos_table = self._put(cos.reshape(1, 1, *cos.shape), self.ttnn.bfloat16, layout=rm)
        self.sin_table = self._put(sin.reshape(1, 1, *sin.shape), self.ttnn.bfloat16, layout=rm)

    # -- step ---------------------------------------------------------------

    def _user_height_sharded(self, cols: int):
        """One user per core on an 8x4 rectangle (a split grid breaks concat_heads)."""

        ttnn = self.ttnn
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
        return ttnn.create_sharded_memory_config(
            shape=(TILE, cols), core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )

    def _linear(self, x, weight, grid, memory_config=None):
        ttnn = self.ttnn
        return ttnn.linear(x, weight, compute_kernel_config=self.compute, core_grid=self.grids[grid],
                           memory_config=memory_config or ttnn.L1_MEMORY_CONFIG)

    def step(self, tokens, positions_rm, positions):
        """One decode step on device tensors; returns ``[1, 1, users, vocab]`` logits.

        ``tokens`` and ``positions_rm`` are uint32 row-major ``[1, users]``;
        ``positions`` is int32 ``[users]``.  Use :meth:`inputs` to build them.
        """

        ttnn = self.ttnn
        d = self.d
        batch = DECODE_BATCH
        x = ttnn.reshape(
            ttnn.embedding(tokens, self.embed, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            (1, 1, batch, d.hidden),
        )
        cos_rows = ttnn.reshape(ttnn.embedding(positions_rm, self.cos_table, layout=ttnn.TILE_LAYOUT),
                                (1, batch, 1, d.head_dim))
        sin_rows = ttnn.reshape(ttnn.embedding(positions_rm, self.sin_table, layout=ttnn.TILE_LAYOUT),
                                (1, batch, 1, d.head_dim))
        cos = sin = None
        scale = 1.0 / math.sqrt(d.head_dim)
        for index, layer in enumerate(self.layers):
            h = ttnn.rms_norm(x, epsilon=d.eps, weight=layer["attn_norm"], compute_kernel_config=self.compute)
            qkv = self._linear(h, layer["qkv"], "qkv")
            ttnn.deallocate(h)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=d.heads, num_kv_heads=d.kv_heads,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)
            if cos is None:
                # RoPE tables must be sharded on the same cores as the heads; an
                # interleaved table silently applies user 0's angle to everyone.
                cos = ttnn.to_memory_config(cos_rows, q.memory_config())
                sin = ttnn.to_memory_config(sin_rows, q.memory_config())
                ttnn.deallocate(cos_rows)
                ttnn.deallocate(sin_rows)
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
            ttnn.experimental.paged_update_cache(self.k_cache[index], k, update_idxs_tensor=positions)
            ttnn.experimental.paged_update_cache(self.v_cache[index], v, update_idxs_tensor=positions)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q, self.k_cache[index], self.v_cache[index], cur_pos_tensor=positions, scale=scale,
                program_config=self.sdpa_program, compute_kernel_config=self.compute_sdpa,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            sharded = ttnn.to_memory_config(attn, self._user_height_sharded(d.head_dim))
            ttnn.deallocate(attn)
            attn = ttnn.experimental.nlp_concat_heads_decode(sharded, num_heads=d.heads)
            ttnn.deallocate(sharded)
            attn_l1 = ttnn.to_memory_config(attn, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn)
            o = self._linear(attn_l1, layer["o"], "o")
            ttnn.deallocate(attn_l1)
            residual = ttnn.add(x, o)
            ttnn.deallocate(o)
            ttnn.deallocate(x)
            x = residual
            h = ttnn.rms_norm(x, epsilon=d.eps, weight=layer["mlp_norm"], compute_kernel_config=self.compute)
            gate = self._linear(h, layer["gate"], "gate")
            up = self._linear(h, layer["up"], "up")
            ttnn.deallocate(h)
            act = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                           memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            mlp = self._linear(act, layer["down"], "down")
            ttnn.deallocate(act)
            residual = ttnn.add(x, mlp)
            ttnn.deallocate(mlp)
            ttnn.deallocate(x)
            x = residual
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        h = ttnn.rms_norm(x, epsilon=d.eps, weight=self.final_norm, compute_kernel_config=self.compute)
        ttnn.deallocate(x)
        logits = self._linear(h, self.head, "head", memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        return logits

    def step_argmax(self, tokens, positions_rm, positions):
        """One step returning uint32 next-token ids ``[.., users]`` on device."""

        ttnn = self.ttnn
        logits = self.step(tokens, positions_rm, positions)
        rows = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        ids = ttnn.argmax(rows, dim=-1)
        ttnn.deallocate(rows)
        return ids

    # -- host helpers -----------------------------------------------------------

    def host_inputs(self, tokens, positions):
        """Host tensors for :meth:`step` from ``[users]`` token ids and positions."""

        ttnn = self.ttnn
        tokens = torch.as_tensor(tokens, dtype=torch.int32).reshape(1, self.users)
        positions = torch.as_tensor(positions, dtype=torch.int32).reshape(1, self.users)
        if int(positions.max()) >= self.max_seq:
            raise ValueError(f"position {int(positions.max())} exceeds the KV cache ({self.max_seq})")
        return (
            ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._shard_users),
            ttnn.from_torch(positions, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._shard_users),
            ttnn.from_torch(positions.reshape(self.users), dtype=ttnn.int32, mesh_mapper=self._shard_rows),
        )

    def inputs(self, tokens, positions):
        return [self.ttnn.to_device(t, self.mesh) for t in self.host_inputs(tokens, positions)]

    def read_ids(self, ids) -> torch.Tensor:
        return self.ttnn.to_torch(ids, mesh_composer=self._gather_users).reshape(-1)[: self.users].to(torch.int64)

    def read_logits(self, logits) -> torch.Tensor:
        out = self.ttnn.to_torch(logits, mesh_composer=self._gather_users).float()
        return out.reshape(self.users, -1)[:, : self.d.vocab]

    # -- trace ------------------------------------------------------------------

    def capture(self, device_inputs):
        """Capture :meth:`step_argmax` over ``device_inputs`` as a replayable trace.

        Requires the mesh to have been opened with a trace region.  Returns the
        output tensor the trace writes into.
        """

        ttnn = self.ttnn
        self.release()
        warm = self.step_argmax(*device_inputs)  # compile outside the trace
        ttnn.deallocate(warm)
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = self.step_argmax(*device_inputs)
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self._trace = (trace, device_inputs, out)
        return out

    def replay(self, host_inputs=None, *, blocking: bool = False):
        """Copy new inputs into the captured tensors (optional) and replay."""

        ttnn = self.ttnn
        if self._trace is None:
            raise RuntimeError("capture() a trace before replay()")
        trace, device_inputs, out = self._trace
        if host_inputs is not None:
            for host, device in zip(host_inputs, device_inputs):
                ttnn.copy_host_to_device_tensor(host, device)
        ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=blocking)
        return out

    def release(self):
        if self._trace is not None:
            self.ttnn.release_trace(self.mesh, self._trace[0])
            self._trace = None

    # -- generation -------------------------------------------------------------

    def generate(self, prompts, max_new_tokens: int, *, eos_token_ids=(), use_trace: bool = True):
        """Greedy-generate for up to ``users`` prompts (lists of token ids).

        Prompts are fed one position per step (decode-path prefill), so every
        user advances in lock-step from position 0.  Returns the generated ids
        per prompt, stopping at an EOS token.
        """

        if not prompts or len(prompts) > self.users:
            raise ValueError(f"need 1..{self.users} prompts, got {len(prompts)}")
        rows = list(prompts) + [prompts[0]] * (self.users - len(prompts))
        lengths = [len(p) for p in rows]
        total = max(lengths) + max_new_tokens - 1
        if total > self.max_seq:
            raise ValueError(f"prompt + generation ({total}) exceeds max_seq {self.max_seq}")
        eos = {int(t) for t in eos_token_ids}
        current = [int(p[0]) for p in rows]
        generated = [[] for _ in rows]
        done = [False] * len(rows)
        device_inputs = self.inputs(current, [0] * self.users)
        if use_trace:
            self.capture(device_inputs)
        for step in range(total):
            host = self.host_inputs(current, [step] * self.users)
            if use_trace:
                ids = self.read_ids(self.replay(host))
            else:
                live = [self.ttnn.to_device(t, self.mesh) for t in host]
                ids = self.read_ids(self.step_argmax(*live))
            for user in range(len(rows)):
                if step + 1 < lengths[user]:
                    current[user] = int(rows[user][step + 1])
                    continue
                token = int(ids[user])
                if not done[user]:
                    generated[user].append(token)
                    done[user] = token in eos or len(generated[user]) >= max_new_tokens
                current[user] = token
            if all(done[: len(prompts)]):
                break
        if use_trace:
            self.release()
        return generated[: len(prompts)]


def checkpoint_block_axis(checkpoint_dir: str) -> str:
    """``quantization_config.block_axis`` of a checkpoint directory (default ``"in"``)."""

    with open(os.path.join(checkpoint_dir, "config.json")) as handle:
        qcfg = json.load(handle).get("quantization_config") or {}
    return str(qcfg.get("block_axis", "in"))


__all__ = [
    "DECODE_BATCH",
    "LlamaDims",
    "P300C_DECODE_GRIDS",
    "TTLlamaDecoder",
    "checkpoint_block_axis",
    "hf_rope_tables",
    "llama_dims",
    "load_tt_llama_weights",
    "validate_llama_config",
]
