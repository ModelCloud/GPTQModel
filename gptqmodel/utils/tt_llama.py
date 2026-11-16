# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Device-resident Llama prefill and decode for Tenstorrent block-float checkpoints.

Weights must come from a checkpoint written with ``block_axis="out"``
(:func:`~gptqmodel.utils.tt_blockfloat_quant.gptq_solve_blockfloat`): the
operand of ``ttnn.linear(x, W^T)`` is ``[in, out]`` and a bf4_b/bf8_b tile shares
one exponent per 16 values of the last dim, so only an out-axis checkpoint
uploads without re-quantization.  Upload drift is measured per tensor and must
be zero.

Storage follows the repository's quantized-inference rule: every projection
and the tied head live on the chip as bf4_b/bf8_b only; nothing dense is kept on
device or host.  The embedding is held as its packed bf8 codes plus exponents on
the host, and each step dequantizes only the rows of the tokens it feeds (one
row per user per decode step, one row per prompt token in prefill).  Each
projection is expanded to float on the host only for the duration of its own
upload.

KV state is a paged block pool (``[blocks, kv_heads, block_size, head_dim]`` per
layer) with a per-user page table, so the pool is sized in tokens rather than
``users x max_context`` and long and short requests share it.

* **Prefill** runs one prompt at a time in fixed chunks: projections over the
  whole chunk, chunked causal flash attention over the paged prefix, and the
  chunk's K/V written straight into that user's blocks.  The head is evaluated
  only on the tile holding the last real token.
* **Decode** runs every user slot in one step (inactive slots point at a
  scratch block), fully on device through argmax, and is captured once as a
  trace and replayed.

Multi-card serving is data parallel with **one process per card**; a ``1xN``
``MeshDevice`` running the same trace measured ~2x slower per step than a single
chip on this software stack, so this class drives exactly one chip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .ttnn_runtime import import_ttnn


TILE = 32
#: User slots per decode step.  The decode head-shuffle ops work on one tile row.
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
    """Reject geometries the op chain cannot run, before touching a chip."""

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


def hf_rope_tables(config, positions: int):
    """``cos``/``sin`` ``[positions, head_dim]`` exactly as transformers computes them.

    Built by the model's own rotary module so Llama-3 frequency scaling is
    reproduced rather than re-derived.
    """

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary = LlamaRotaryEmbedding(config)
    probe = torch.zeros(1, 1, dtype=torch.float32)
    cos, sin = rotary(probe, torch.arange(positions).unsqueeze(0))
    return cos[0].float(), sin[0].float()


# --------------------------------------------------------------------------- checkpoint


@dataclass
class PackedBlockfloat:
    """One stored matrix as checkpoint codes: ``rows x cols`` blocked along cols."""

    codes: torch.Tensor
    exponents: torch.Tensor
    shape: tuple
    bits: int

    def expand(self) -> torch.Tensor:
        """Exact float32 values of the stored matrix (caller owns and frees it)."""

        from .tt_blockfloat_quant import _unpack_codes, recompose_blockfloat

        codes = _unpack_codes(self.codes, self.bits, self.shape)
        return recompose_blockfloat(codes, self.exponents, bits=self.bits)

    def column_major(self) -> "ColumnGather":
        """A packed copy laid out column-major, for fast per-column gathers."""

        return ColumnGather(self)

    def expand_columns(self, columns: torch.Tensor) -> torch.Tensor:
        """Exact values of selected columns only, as ``[len(columns), rows]``.

        For the embedding (stored as ``E^T [hidden, vocab]``) a column is one
        token's row of ``E``; only those codes are decoded.
        """

        from .tt_blockfloat_quant import stored_mantissa_bits

        if self.bits != 8:
            raise NotImplementedError("column gather is implemented for 8-bit storage")
        rows, cols = self.shape
        columns = columns.to(torch.int64)
        codes = self.codes.reshape(rows, cols)[:, columns].to(torch.float32)
        exps = self.exponents[:, columns // 16].to(torch.float32)
        levels = float(1 << stored_mantissa_bits(self.bits))
        offset = 2 * levels - 1
        return ((codes - offset) / levels * torch.pow(2.0, exps)).T.contiguous()


class ColumnGather:
    """bf8 codes of a ``rows x cols`` matrix stored column-major (still packed).

    The tied embedding is stored as ``E^T [hidden, vocab]``, so one token is a
    column: gathering it from the checkpoint layout strides across the whole
    263 MB code array.  This keeps the same uint8 codes and int8 exponents
    transposed (a packed alternate layout, not an expansion) so a token's codes
    are contiguous.
    """

    def __init__(self, packed: PackedBlockfloat):
        from .tt_blockfloat_quant import stored_mantissa_bits

        if packed.bits != 8:
            raise NotImplementedError("column gather is implemented for 8-bit storage")
        rows, cols = packed.shape
        self.rows = rows
        self.codes = packed.codes.reshape(rows, cols).T.contiguous()      # [cols, rows] uint8
        self.exponents = packed.exponents.T.contiguous()                  # [cols / 16, rows] int8
        levels = float(1 << stored_mantissa_bits(8))
        self._offset = 2 * levels - 1
        self._levels = levels

    def __call__(self, columns: torch.Tensor) -> torch.Tensor:
        columns = torch.as_tensor(columns, dtype=torch.int64)
        codes = self.codes[columns].to(torch.float32)
        scale = torch.pow(2.0, self.exponents[columns // 16].to(torch.float32))
        return (codes - self._offset) / self._levels * scale


def load_tt_llama_weights(checkpoint_dir: str):
    """Read an out-axis block-float Llama checkpoint without expanding it.

    Returns ``(config, weights)`` where ``weights`` maps each projection prefix
    and ``"embed"`` (``E^T [hidden, vocab]``) to a :class:`PackedBlockfloat`, and
    each norm name to its (dense, unquantized) weight.
    """

    from transformers import AutoConfig

    from .tt_blockfloat_quant import load_hf_state_dict

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
        packed = PackedBlockfloat(
            codes=state[name].contiguous(),
            exponents=state[f"{prefix}.exponents"].contiguous(),
            shape=tuple(int(v) for v in state[f"{prefix}.code_shape"]),
            bits=int(widths.get(prefix, default_bits)),
        )
        weights["embed" if prefix.endswith("embed_tokens") else prefix] = packed
    for name, tensor in state.items():
        if name.endswith("norm.weight"):
            weights[name] = tensor.to(torch.float32)
    if "embed" not in weights:
        raise ValueError(f"{checkpoint_dir} has no block-float embedding to tie the head to")
    if weights["embed"].bits != 8:
        raise NotImplementedError("the tied embedding/head must be stored at 8 bits")
    return config, weights


# --------------------------------------------------------------------------- KV pool


def kv_pool_geometry(max_context: int, kv_tokens: int, block_size: int):
    """``(blocks per user, pool blocks incl. scratch)`` satisfying both device ops.

    * Chunked SDPA needs page-table rows of a multiple of 32 bytes, so the
      per-user block count is rounded up to 8 int32 entries.
    * ``paged_update_cache`` requires the page-table width to be no larger than
      the pool, so a small pool (a short ``generate``) is grown to at least one
      full row plus the scratch block.  Without this a one-prompt, 100-token
      run sized a 7-block pool against an 8-wide table and failed in the op.
    """

    per_user = math.ceil(math.ceil(max_context / block_size) / 8) * 8
    blocks = max(math.ceil(kv_tokens / block_size), per_user) + 1  # + scratch block 0
    return per_user, blocks


class KVBlockPool:
    """Host-side allocator for the paged KV cache.

    Block 0 is reserved as a scratch target for idle user slots, so a decode
    step over all slots never writes into a live sequence.
    """

    def __init__(self, num_blocks: int, block_size: int, users: int, max_blocks_per_user: int):
        if num_blocks < 2:
            raise ValueError("need at least one usable block besides the scratch block")
        self.block_size = block_size
        self.users = users
        self.max_blocks_per_user = max_blocks_per_user
        self.free = list(range(num_blocks - 1, 0, -1))
        self.table = torch.zeros(users, max_blocks_per_user, dtype=torch.int32)
        self.owned = [[] for _ in range(users)]

    def reserve(self, user: int, tokens: int) -> None:
        need = math.ceil(tokens / self.block_size)
        if need > self.max_blocks_per_user:
            raise ValueError(f"{tokens} tokens exceed the per-user context "
                             f"({self.max_blocks_per_user * self.block_size})")
        while len(self.owned[user]) < need:
            if not self.free:
                raise MemoryError("KV block pool exhausted")
            block = self.free.pop()
            self.table[user, len(self.owned[user])] = block
            self.owned[user].append(block)

    def release(self, user: int) -> None:
        self.free.extend(reversed(self.owned[user]))
        self.owned[user] = []
        self.table[user].zero_()


# --------------------------------------------------------------------------- model


class TTLlama:
    """Paged prefill + traced batched greedy decode on one Tenstorrent chip.

    ``kv_tokens`` sizes the shared KV pool (all users together);
    ``max_context`` bounds one user's sequence.  Both are rounded up to whole
    ``block_size`` blocks.
    """

    def __init__(
        self,
        device,
        config,
        weights: dict,
        *,
        max_context: int = 8192,
        kv_tokens: Optional[int] = None,
        block_size: int = 64,
        prefill_chunk: int = 2048,
        fidelity: str = "HiFi4",
        kv_dtype: str = "bfloat16",
        decode_k_chunk: Optional[int] = None,
        grids: Optional[dict] = None,
        check_drift: bool = True,
    ):
        ttnn = import_ttnn()
        self.ttnn = ttnn
        self.device = device
        if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
            raise NotImplementedError("TTLlama drives one chip; run one process per card for data parallelism")
        self.config = config
        self.d = validate_llama_config(config)
        self.users = DECODE_BATCH
        if block_size % TILE or prefill_chunk % max(block_size, 128):
            raise ValueError("block_size must be a tile multiple and prefill_chunk a multiple of max(block_size, 128)")
        self.block_size = block_size
        self.prefill_chunk = prefill_chunk
        self.max_blocks_per_user, self.num_blocks = kv_pool_geometry(
            max_context, kv_tokens or self.users * min(max_context, 4096), block_size)
        self.max_context = self.max_blocks_per_user * block_size
        self.kv_dtype = getattr(ttnn, kv_dtype)
        self.check_drift = check_drift
        self.drift: dict = {}
        grid_shapes = dict(P300C_DECODE_GRIDS, **(grids or {}))
        self.grids = {k: ttnn.CoreGrid(y=v[0], x=v[1]) for k, v in grid_shapes.items()}
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity),
            fp32_dest_acc_en=True, packer_l1_acc=True, math_approx_mode=False,
        )
        self.compute_sdpa = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False,
        )
        grid = device.compute_with_storage_grid_size()
        # SDPA-decode reads K/V in chunks: 256 measured fastest from ~pos 300 up.
        self.decode_sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=TILE,
            k_chunk_size=int(decode_k_chunk or 256), exp_approx_mode=False,
        )
        self.prefill_sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False,
        )
        self._upload(weights)
        # Only the packed codes survive: a column-major copy for row gathers.
        self.embed = weights["embed"].column_major()
        self.pool = KVBlockPool(self.num_blocks, block_size, self.users, self.max_blocks_per_user)
        self._allocate_cache()
        self.cos, self.sin = hf_rope_tables(config, self.max_context)
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.cos_table = self._put(self.cos.reshape(1, 1, *self.cos.shape), ttnn.bfloat16, layout=rm)
        self.sin_table = self._put(self.sin.reshape(1, 1, *self.sin.shape), ttnn.bfloat16, layout=rm)
        self.page_table = ttnn.from_torch(self.pool.table, dtype=ttnn.int32, device=device)
        self._trace = None

    # -- upload -----------------------------------------------------------

    def _put(self, tensor, dtype, name=None, layout=None, memory_config=None):
        ttnn = self.ttnn
        device_tensor = ttnn.from_torch(
            tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT if layout is None else layout, device=self.device,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.check_drift and name is not None and dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b):
            back = ttnn.to_torch(device_tensor).float().reshape(tensor.shape)
            self.drift[name] = float((back - tensor.float()).abs().max())
        return device_tensor

    def _put_blockfloat(self, packed: PackedBlockfloat, name: str, extra: Sequence[PackedBlockfloat] = ()):
        """Expand (and concatenate along out) for one upload, then drop the floats."""

        parts = [packed, *extra]
        if len({p.bits for p in parts}) != 1:
            raise ValueError(f"{name}: fused parts have different widths {[p.bits for p in parts]}")
        dense = torch.cat([p.expand() for p in parts], dim=1) if extra else packed.expand()
        try:
            return self._put(dense, self.ttnn.bfloat4_b if packed.bits == 4 else self.ttnn.bfloat8_b, name)
        finally:
            del dense

    def _norm(self, weight):
        return self._put(weight.reshape(1, 1, -1, TILE).float(), self.ttnn.bfloat16, layout=self.ttnn.ROW_MAJOR_LAYOUT)

    def _upload(self, weights):
        self.layers = []
        for index in range(self.d.layers):
            p = f"model.layers.{index}."
            # Concatenating along out_features at tile multiples keeps every
            # 16-value exponent block whole, so the fused QKV is still exact.
            self.layers.append({
                "qkv": self._put_blockfloat(weights[p + "self_attn.q_proj"], p + "qkv",
                                            (weights[p + "self_attn.k_proj"], weights[p + "self_attn.v_proj"])),
                "o": self._put_blockfloat(weights[p + "self_attn.o_proj"], p + "o_proj"),
                "gate": self._put_blockfloat(weights[p + "mlp.gate_proj"], p + "gate_proj"),
                "up": self._put_blockfloat(weights[p + "mlp.up_proj"], p + "up_proj"),
                "down": self._put_blockfloat(weights[p + "mlp.down_proj"], p + "down_proj"),
                "attn_norm": self._norm(weights[p + "input_layernorm.weight"]),
                "mlp_norm": self._norm(weights[p + "post_attention_layernorm.weight"]),
            })
        self.final_norm = self._norm(weights["model.norm.weight"])
        self.head = self._put_blockfloat(weights["embed"], "lm_head")

    def _allocate_cache(self):
        ttnn = self.ttnn
        shape = [self.num_blocks, self.d.kv_heads, self.block_size, self.d.head_dim]
        self.k_cache, self.v_cache = [], []
        for _ in range(self.d.layers):
            for cache in (self.k_cache, self.v_cache):
                # Zero-filled on device, never left uninitialized: flash-decode reads
                # whole K/V chunks, including masked positions past a sequence's end
                # (scratch block 0 and unwritten tails).  Masking multiplies their
                # contribution by zero, which still yields NaN when uninitialized
                # memory decodes as Inf/NaN -- measured as a 6e36 attention output
                # and all-zero logits on a fresh allocation.
                cache.append(ttnn.zeros(shape, dtype=self.kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG))

    def kv_bytes(self) -> int:
        per = 1.0625 if self.kv_dtype == self.ttnn.bfloat8_b else 2.0
        return int(2 * self.d.layers * self.num_blocks * self.d.kv_heads * self.block_size * self.d.head_dim * per)

    def _sync_page_table(self):
        self.ttnn.copy_host_to_device_tensor(
            self.ttnn.from_torch(self.pool.table, dtype=self.ttnn.int32), self.page_table)

    # -- shared pieces ------------------------------------------------------

    def _embed_rows(self, token_ids) -> torch.Tensor:
        """bf16 ``[n, hidden]`` rows of the bf8 embedding, decoded for these ids only."""

        return self.embed(torch.as_tensor(token_ids)).to(torch.bfloat16)

    def _linear(self, x, weight, grid=None, memory_config=None):
        ttnn = self.ttnn
        kwargs = {"core_grid": self.grids[grid]} if grid else {}
        return ttnn.linear(x, weight, compute_kernel_config=self.compute,
                           memory_config=memory_config or ttnn.L1_MEMORY_CONFIG, **kwargs)

    def _mlp(self, x, layer, grid: bool, memory_config):
        ttnn = self.ttnn
        h = ttnn.rms_norm(x, epsilon=self.d.eps, weight=layer["mlp_norm"], compute_kernel_config=self.compute)
        gate = self._linear(h, layer["gate"], "gate" if grid else None, memory_config)
        up = self._linear(h, layer["up"], "up" if grid else None, memory_config)
        ttnn.deallocate(h)
        act = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=memory_config)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self._linear(act, layer["down"], "down" if grid else None, memory_config)
        ttnn.deallocate(act)
        residual = ttnn.add(x, out, memory_config=memory_config)
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        return residual

    # -- prefill ------------------------------------------------------------

    def prefill(self, user: int, token_ids: Sequence[int]) -> torch.Tensor:
        """Run a prompt for ``user`` into its KV blocks; return last-token logits ``[vocab]``.

        The prompt is processed in ``prefill_chunk`` pieces; the final piece is
        padded to a whole ``max(block_size, 128)`` span (padding sits after the
        real tokens, so causal attention never reads it, and decode overwrites
        it).
        """

        ttnn = self.ttnn
        d = self.d
        n = len(token_ids)
        if n == 0:
            raise ValueError("empty prompt")
        align = max(self.block_size, 128)
        padded = math.ceil(n / align) * align
        if padded > self.max_context:
            raise ValueError(f"prompt of {n} tokens exceeds max_context {self.max_context}")
        self.pool.release(user)
        self.pool.reserve(user, padded)
        self._sync_page_table()
        ids = torch.as_tensor(list(token_ids) + [0] * (padded - n), dtype=torch.int64)
        user_table = ttnn.from_torch(self.pool.table[user:user + 1], dtype=ttnn.int32, device=self.device)
        # The chunk offset is read from device memory, so one compiled program
        # serves every chunk; a scalar offset keys the program cache per chunk
        # and measured 5.0 s cold against 0.42 s warm for an 8k prompt.
        chunk_start = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=self.device)
        dram = ttnn.DRAM_MEMORY_CONFIG
        scale = 1.0 / math.sqrt(d.head_dim)
        logits = None
        for start in range(0, padded, self.prefill_chunk):
            stop = min(start + self.prefill_chunk, padded)
            length = stop - start
            x = ttnn.from_torch(self._embed_rows(ids[start:stop]).reshape(1, 1, length, d.hidden),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=dram)
            cos = ttnn.from_torch(self.cos[start:stop].reshape(1, 1, length, d.head_dim), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=self.device)
            sin = ttnn.from_torch(self.sin[start:stop].reshape(1, 1, length, d.head_dim), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=self.device)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32),
                                            chunk_start)
            first, last = start // self.block_size, stop // self.block_size
            chunk_table = ttnn.from_torch(self.pool.table[user:user + 1, first:last].contiguous(), dtype=ttnn.int32,
                                          device=self.device)
            for index, layer in enumerate(self.layers):
                h = ttnn.rms_norm(x, epsilon=d.eps, weight=layer["attn_norm"], compute_kernel_config=self.compute)
                qkv = self._linear(h, layer["qkv"], memory_config=dram)
                ttnn.deallocate(h)
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qkv, num_heads=d.heads, num_kv_heads=d.kv_heads, transpose_k_heads=False, memory_config=dram)
                ttnn.deallocate(qkv)
                q2 = ttnn.experimental.rotary_embedding_hf(q, cos, sin)
                k2 = ttnn.experimental.rotary_embedding_hf(k, cos, sin)
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                for cache, value in ((self.k_cache[index], k2), (self.v_cache[index], v)):
                    if self.kv_dtype != ttnn.bfloat16:
                        cast = ttnn.typecast(value, self.kv_dtype)
                        ttnn.experimental.paged_fill_cache(cache, cast, chunk_table, batch_idx=0)
                        ttnn.deallocate(cast)
                    else:
                        ttnn.experimental.paged_fill_cache(cache, value, chunk_table, batch_idx=0)
                ttnn.deallocate(k2)
                ttnn.deallocate(v)
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q2, self.k_cache[index], self.v_cache[index], user_table, chunk_start_idx_tensor=chunk_start,
                    scale=scale, program_config=self.prefill_sdpa, compute_kernel_config=self.compute_sdpa)
                ttnn.deallocate(q2)
                merged = ttnn.experimental.nlp_concat_heads(attn, memory_config=dram)
                ttnn.deallocate(attn)
                o = self._linear(merged, layer["o"], memory_config=dram)
                ttnn.deallocate(merged)
                residual = ttnn.add(x, o, memory_config=dram)
                ttnn.deallocate(o)
                ttnn.deallocate(x)
                x = self._mlp(residual, layer, grid=False, memory_config=dram)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(chunk_table)
            if start <= n - 1 < stop:
                row = n - 1 - start
                tile = (row // TILE) * TILE
                last_tile = ttnn.slice(x, (0, 0, tile, 0), (1, 1, tile + TILE, d.hidden))
                h = ttnn.rms_norm(last_tile, epsilon=d.eps, weight=self.final_norm, compute_kernel_config=self.compute)
                ttnn.deallocate(last_tile)
                out = self._linear(h, self.head, "head", memory_config=dram)
                ttnn.deallocate(h)
                logits = ttnn.to_torch(out).float().reshape(TILE, -1)[row - tile, : d.vocab]
                ttnn.deallocate(out)
            ttnn.deallocate(x)
        ttnn.deallocate(user_table)
        ttnn.deallocate(chunk_start)
        return logits

    # -- decode -------------------------------------------------------------

    def _user_height_sharded(self, cols: int):
        """One user per core on an 8x4 rectangle (a split grid breaks concat_heads)."""

        ttnn = self.ttnn
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
        return ttnn.create_sharded_memory_config(
            shape=(TILE, cols), core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )

    def decode_step(self, x, positions_rm, positions, active: int = DECODE_BATCH):
        """One decode step for all slots; returns uint32 next-token ids on device.

        ``x`` is the bf16 ``[1, 1, 32, hidden]`` embedding of each slot's
        current token, ``positions_rm`` uint32 row-major ``[1, 32]`` and
        ``positions`` int32 ``[32]``.  Build them with :meth:`decode_inputs`.

        ``active`` is the number of leading slots that attend.  Flash-decode
        divides the cores between the query rows it is given, so a single long
        sequence among 32 rows gets ~3 of 110 cores: measured at 16k context,
        3.4 ms per layer with 32 rows against 0.13 ms with one.  Slots at and
        beyond ``active`` still run the projections (their rows are padding of
        the same tile) but produce meaningless tokens.
        """

        ttnn = self.ttnn
        d = self.d
        batch = DECODE_BATCH
        if not 1 <= active <= batch:
            raise ValueError(f"active must be in 1..{batch}, got {active}")
        if active < batch:
            table = ttnn.slice(self.page_table, (0, 0), (active, self.max_blocks_per_user))
            cur = ttnn.slice(positions, (0,), (active,))
        else:
            table, cur = self.page_table, positions
        cos_rows = ttnn.reshape(ttnn.embedding(positions_rm, self.cos_table, layout=ttnn.TILE_LAYOUT),
                                (1, batch, 1, d.head_dim))
        sin_rows = ttnn.reshape(ttnn.embedding(positions_rm, self.sin_table, layout=ttnn.TILE_LAYOUT),
                                (1, batch, 1, d.head_dim))
        cos = sin = None
        scale = 1.0 / math.sqrt(d.head_dim)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        for index, layer in enumerate(self.layers):
            h = ttnn.rms_norm(x, epsilon=d.eps, weight=layer["attn_norm"], compute_kernel_config=self.compute)
            qkv = self._linear(h, layer["qkv"], "qkv")
            ttnn.deallocate(h)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=d.heads, num_kv_heads=d.kv_heads, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
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
            ttnn.experimental.paged_update_cache(self.k_cache[index], k, update_idxs_tensor=positions,
                                                 page_table=self.page_table)
            ttnn.experimental.paged_update_cache(self.v_cache[index], v, update_idxs_tensor=positions,
                                                 page_table=self.page_table)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if active < batch:
                q_rows = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(q)
                q = ttnn.slice(q_rows, (0, 0, 0, 0), (1, active, d.heads, d.head_dim))
                ttnn.deallocate(q_rows)
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, self.k_cache[index], self.v_cache[index], page_table_tensor=table,
                cur_pos_tensor=cur, scale=scale, program_config=self.decode_sdpa,
                compute_kernel_config=self.compute_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q)
            if active < batch:
                padded = ttnn.pad(attn, [(0, 0), (0, batch - active), (0, 0), (0, 0)], 0.0)
                ttnn.deallocate(attn)
                attn = padded
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
            x = self._mlp(residual, layer, grid=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        if active < batch:
            ttnn.deallocate(table)
            ttnn.deallocate(cur)
        h = ttnn.rms_norm(x, epsilon=d.eps, weight=self.final_norm, compute_kernel_config=self.compute)
        ttnn.deallocate(x)
        logits = self._linear(h, self.head, "head", memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        rows = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        ids = ttnn.argmax(rows, dim=-1)
        ttnn.deallocate(rows)
        return ids

    def decode_inputs(self, tokens, positions, *, on_device: bool = False):
        """Host (or device) tensors for :meth:`decode_step` from per-slot ids and positions."""

        ttnn = self.ttnn
        positions = torch.as_tensor(positions, dtype=torch.int32).reshape(1, self.users)
        x = self._embed_rows(torch.as_tensor(tokens)).reshape(1, 1, self.users, self.d.hidden)
        host = (
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.from_torch(positions, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.from_torch(positions.reshape(self.users), dtype=ttnn.int32),
        )
        return [ttnn.to_device(t, self.device) for t in host] if on_device else host

    def capture(self, tokens, positions, active: int = DECODE_BATCH):
        """Capture :meth:`decode_step` as a trace (needs a trace region on the device)."""

        ttnn = self.ttnn
        self.release()
        device_inputs = self.decode_inputs(tokens, positions, on_device=True)
        ttnn.deallocate(self.decode_step(*device_inputs, active=active))  # compile outside the trace
        device_inputs = self.decode_inputs(tokens, positions, on_device=True)
        trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        out = self.decode_step(*device_inputs, active=active)
        ttnn.end_trace_capture(self.device, trace, cq_id=0)
        self._trace = (trace, device_inputs, out)
        self.active = active

    def replay(self, tokens=None, positions=None, *, blocking: bool = False):
        """Load new inputs into the captured tensors (optional) and replay the step."""

        ttnn = self.ttnn
        if self._trace is None:
            raise RuntimeError("capture() a trace before replay()")
        trace, device_inputs, out = self._trace
        if tokens is not None:
            for host, device in zip(self.decode_inputs(tokens, positions), device_inputs):
                ttnn.copy_host_to_device_tensor(host, device)
        ttnn.execute_trace(self.device, trace, cq_id=0, blocking=blocking)
        return out

    def read_ids(self, ids) -> torch.Tensor:
        return self.ttnn.to_torch(ids).reshape(-1)[: self.users].to(torch.int64)

    def release(self):
        if self._trace is not None:
            self.ttnn.release_trace(self.device, self._trace[0])
            self._trace = None

    # -- generation -------------------------------------------------------------

    def generate(self, prompts, max_new_tokens: int, *, eos_token_ids=(), use_trace: bool = True):
        """Greedy-generate for up to 32 prompts: chunked prefill, then batched decode."""

        if not prompts or len(prompts) > self.users:
            raise ValueError(f"need 1..{self.users} prompts, got {len(prompts)}")
        eos = {int(t) for t in eos_token_ids}
        generated = [[] for _ in prompts]
        done = [False] * len(prompts)
        tokens = [0] * self.users
        positions = [0] * self.users
        for user, prompt in enumerate(prompts):
            if len(prompt) + max_new_tokens > self.max_context:
                raise ValueError(f"prompt {user}: {len(prompt)} + {max_new_tokens} exceeds {self.max_context}")
            first = int(self.prefill(user, prompt).argmax())
            generated[user].append(first)
            done[user] = first in eos or max_new_tokens <= 1
            self.pool.reserve(user, len(prompt) + max_new_tokens)
            tokens[user], positions[user] = first, len(prompt)
        self._sync_page_table()
        active = next(b for b in (1, 2, 4, 8, 16, 32) if b >= len(prompts))
        if use_trace:
            self.capture(tokens, positions, active)
        while not all(done):
            if use_trace:
                ids = self.read_ids(self.replay(tokens, positions))
            else:
                ids = self.read_ids(self.decode_step(*self.decode_inputs(tokens, positions, on_device=True),
                                                     active=active))
            for user in range(len(prompts)):
                if done[user]:
                    continue
                token = int(ids[user])
                generated[user].append(token)
                done[user] = token in eos or len(generated[user]) >= max_new_tokens
                tokens[user], positions[user] = token, positions[user] + 1
        if use_trace:
            self.release()
        for user in range(len(prompts)):
            self.pool.release(user)
        self._sync_page_table()
        return generated


__all__ = [
    "DECODE_BATCH",
    "KVBlockPool",
    "kv_pool_geometry",
    "LlamaDims",
    "P300C_DECODE_GRIDS",
    "PackedBlockfloat",
    "TTLlama",
    "hf_rope_tables",
    "llama_dims",
    "load_tt_llama_weights",
    "validate_llama_config",
]
