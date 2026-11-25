# LeapQuant recurrent-state quantization

[Documentation index](README.md) · [Project overview](../README.md)

LeapQuant is an opt-in runtime quantizer for the fixed-size recurrent state of
delta-rule linear attention. It is **not** a weight format: model weights,
checkpoints, and `quant_method` are unchanged. LeapQuant composes with weight
quantization in the supported GPT-QModel ports.

Reference: *LeapQuant: Efficient Linear Attention with Accurate Recurrent State
Quantization*, [arXiv:2609.38166](https://arxiv.org/abs/2609.38166).

## Why the state

A Gated DeltaNet (GDN) or Kimi Delta Attention (KDA) layer compresses the
context into a state matrix `S` of shape `[key_dim, value_dim]`. Every decoded
token reads that matrix, applies a diagonal decay and one rank-one delta-rule
update, and writes it back. The state is both a memory cost and a per-token
traffic cost, so weight-only quantization does not address it.

LeapQuant uses three training-free mechanisms:

1. **Per-window quantization.** The packed checkpoint stays fixed for a window
   of tokens. Each token's `(decay, key, update)` record is buffered in higher
   precision, and the state is re-quantized only at the window boundary.
2. **Compensator Tokens.** A rank-`R` high-precision outer-product term captures
   the state's dominant structure. Only the residual is quantized, and the term
   is replayed additively at readout.
3. **Residual smoothing.** Key rows are rescaled by the square root of their
   mean residual magnitude before symmetric quantization, then rescaled back at
   readout.

## Usage

```python
from gptqmodel import GPTQModel, LeapQuantConfig

model = GPTQModel.load(
    "Kimi/Kimi-K3",
    state_quant=LeapQuantConfig(bits=8, window=16, rank=4),
)
out = model.generate("...", max_new_tokens=128)
```

The option is off by default. It can be enabled or disabled after loading, but
the policy must be set before the first forward; an existing recurrent-state
cache keeps the policy it was created with.

```python
model.enable_state_quant(LeapQuantConfig(bits=8))
model.enable_state_quant(None)  # use the unquantized path for new caches
```

| Field | Default | Meaning |
| --- | --- | --- |
| `bits` | `8` | Symmetric code width: `4`, `6`, or `8`. 8 is the production point. |
| `window` | `16` | Tokens between state quantizations. `1` is per-token quantization. |
| `rank` | `4` | Compensator Tokens. `0` disables them; `6` is recommended for 4/6-bit. |
| `smooth` | `True` | Key-row residual smoothing. |
| `buffer_dtype` | `bfloat16` | In-window key/update buffer dtype (`float16`, `bfloat16`, or `float32`). |
| `lowrank_iters` | `2` | Power iterations for the randomized Compensator fit. |
| `lowrank_oversample` | `4` | Oversampling columns for that fit. |

## Supported ports

| Family | Port | Status |
| --- | --- | --- |
| Kimi K3 with KDA layers | `models/ports/kimi_k3` | Supported |
| Qwen3.5 dense (GDN) | `models/ports/native/qwen3_5` | Supported |
| GLM-5.3 (KDA) | `models/ports/native/glm5_next` | Supported |

The quantizer applies to the delta-rule family whose update is a diagonal decay
followed by one rank-one correction. It does not apply to softmax attention or
to Mamba-style selective SSMs.

GDN permits fewer key heads than value heads. The quantizer repeats each key
head across its value-head group in the same order as the port's delta-rule
kernel, and each value head keeps an independent packed state.

## Flow

`generate()` prepares the cache before the first forward. Qwen3.5 and GLM-5.3
resolve the decoder policy in `LeapQuantGenerationMixin`. The Kimi K3 port
creates its own cache in `KimiLinearModel.forward`. A caller-provided cache
retains the policy selected when it was created.

```mermaid
flowchart TD
    A[Generate with caching] --> B{Caller supplies cache}
    B -->|Yes| C[Keep the caller cache]
    B -->|No| D{Decoder enables LeapQuant}
    D -->|No| E[Build the port's dense cache]
    D -->|Yes| F[Build the port's LeapQuant cache]
    C --> G[Model forward]
    E --> G
    F --> G
    G --> H{Packed recurrent state}
    H -->|No| I[Run the dense recurrence]
    H -->|Yes| J{Unpadded and unpacked input}
    J -->|No| K[Raise before updating cache]
    J -->|Yes| L[Run packed prefill or decode]
```

```mermaid
flowchart TD
    A[Decode token] --> B{State policy}
    B -->|off| C[Existing delta-rule kernel]
    C --> D[Dense recurrent state]
    B -->|LeapQuant| E[Packed checkpoint + window buffer]
    E --> F[Replay buffered records]
    F --> G[Compute output and new update]
    G --> H{Window boundary}
    H -->|no| I[Buffer record]
    H -->|yes| J[Fit Compensator and quantize residual]
    J --> E
```

### Before and after

| Area | Before | After (LeapQuant enabled) |
| --- | --- | --- |
| Generation cache | The generation framework creates the default dense cache for Qwen3.5 and GLM-5.3. | The decoder's LeapQuant policy selects a packed recurrent cache before the first forward. |
| Decode state | `KimiDeltaAttention`, `Qwen3_5GatedDeltaNet`, and `Glm5NextTextLinearAttention` read and rewrite a dense recurrent tensor through the cache layer. | The cache layer owns a `LeapQuantState`; decode replays the packed checkpoint and buffered records, then re-quantizes at the window boundary. |
| Cache ownership | A Transformers `LinearAttentionLayer` or `KimiDeltaCacheLayer` stores the dense recurrent tensor. | `LeapQuantLinearAttentionLayer` or `KimiDeltaCacheLayer(state_quant=...)` stores int8 codes, scales, Compensator factors, and the window buffer; the dense slot stays empty. |
| Prefill | The port kernel writes the final dense state. | The same dense kernel runs, then `LeapQuantState.quantize_` checkpoints the final state and clears the window. |
| Invalid cache inputs | Ports process input masks and cache operations for their dense recurrence. | The packed path rejects padding, packed sequence boundaries, crop, and live batch expansion before updating the cache. |
| Default | Dense FP32 recurrent state. | Unchanged: `state_quant=None` returns the original cache and kernel path. |

Prefill uses the port's existing FP32 delta-rule kernel: the packed checkpoint
is materialized for the call, the prompt runs in the model dtype, and the final
state is checkpointed and the window reset. Decode uses the packed path above.
The portable decode path replays the buffered window into a transient dense
FP32 state for each step and releases it immediately; it is never retained.
No dense state is retained after the call.

`LeapQuantState.prefill()` processes a prompt on a temporary copy of the packed
checkpoint and window buffer. It commits that copy once all outputs are ready.
If a later token fails, the caller can correct the prompt and retry it against
the original state. The model ports use their own prefill kernels.

## Storage

The persistent representation is:

```
int8 codes [B, H, K, V]
float32 key scale [B, H, K]
float32 value scale [B, H, V]
float16 compensator factors [B, H, K, R] and [B, H, V, R]
window keys/updates [B, H, L, K/V] in the configured buffer dtype
window decay records [B, H, L, K] for KDA or [B, H, L, 1] for GDN, FP32
```

For `K=V=128`, `L=16`, `R=4`, the 8-bit checkpoint is about 1.19 bytes per
state element before the window buffer; the buffer adds about 8 KiB per head
for GDN and about 16 KiB per head for KDA. The buffer is part of
`storage_bytes()` and is never omitted from memory accounting.

In this release the 4-bit and 6-bit modes use the same int8 code container as
8-bit. Their accuracy is available, but their packed-storage benefit is not;
packed 4/6-bit containers are a follow-up. 8-bit is the recommended mode.

## Constraints

The implementation fails closed instead of silently degrading:

- Use unpadded input with an optional 2D all-ones mask. The ports check this
  before masking or changing the cache, including during single-token decode.
- The packed path rejects `cu_seqlens` and other packed sequence boundaries.
- Use the default dynamic cache for generation with LeapQuant. A cache can
  reorder live sequences at the same batch size and rejects live expansion.
- The packed cache supports ordinary generation and cached continuation.
  Operations that require recurrent-state rollback raise.
- Training and autograd are not supported.
- The prefill path uses a transient dense state; it is not a memory-saving
  claim.

## Validation

The tests are:

```bash
python -m pytest tests/test_leapquant_state.py tests/test_leapquant_models.py -q
```

`test_leapquant_state.py` compares the quantizer against a hand-written FP32
delta-rule recurrence, covers 4/6/8-bit, window boundaries, Compensator rank,
smoothing, storage, reorder, and non-finite rejection. `test_leapquant_models.py`
checks Kimi K3, Qwen3.5, and GLM-5.3 prefill/decode parity, packed-state
ownership, cache-layer selection, padded-mask and `cu_seqlens` rejection, and
the public toggle. The regressions also cover `generate()`, cache continuation
after offload and reorder, and retries after an invalid packed restore or a
quantization failure.

The portable Torch path is a correctness and memory baseline, not a
throughput-optimized kernel. Measure it with:

```bash
python scripts/benchmark_leapquant_state.py --device cuda
```

On one accelerator, with `K=V=128`, 8 heads, `window=16`, `rank=4`, 8-bit, and
model-shaped inputs:

| Step | Time |
| --- | ---: |
| Dense FP32 delta-rule step | 64-70 µs |
| LeapQuant decode step | 724 µs |
| Compensator fit (once per window) | 1175 µs |

Decode is about 10x the dense step, and the fit dominates the per-token cost
through its one-per-window amortization. Two costs remain structural:

- The step is dispatch-bound. Roughly 40 small eager kernels per token, plus
  one host synchronization for the finiteness check, cost more than the
  arithmetic.
- The fit is limited by the accelerator linear-algebra drivers. The current
  form uses a Cholesky-factor orthonormalization and a Gram-matrix
  eigendecomposition because a batched QR plus a wide SVD of the sketch cost
  several times more on the same hardware.

The paper's speedups come from specialized kernels. A native kernel is a
separate follow-up and is not claimed here.
