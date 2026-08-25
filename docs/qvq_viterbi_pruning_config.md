# QVQ `ViterbiPruningConfig`

`ViterbiPruningConfig` is the public policy for QVQ's exact Viterbi
survivor pruning. It selects, per quantization run, whether the exact
norm-band contiguous-band recurrence added in PR #45 is used for the CUDA V2
segmented grid dispatch, or whether the unmodified baseline recurrence runs.

> **Stacked change.** This configuration sits on top of PR #45
> (`polly/norm-rank-kernel`), which added the norm-band kernel itself. Nothing
> here changes the kernel's math, its eligibility set, or its measured
> performance; it only decides whether the existing dispatch is taken and
> whether an ineligible call is allowed to fall back silently.

## Schema

```python
from gptqmodel.quantization import QVQConfig, ViterbiPruningConfig
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `mode` | `"auto" \| "off" \| "required"` | `"auto"` | Policy selector |
| `strategy` | `"norm_band"` | `"norm_band"` | Pruning family; only the exact band exists |
| `exact` | `bool` | `True` | `False` is rejected until an approximate strategy exists |
| `fallback` | `"baseline" \| "error"` | `"baseline"` | What `auto` does with an ineligible call |

`QVQConfig.viterbi_pruning` is a sibling of `QVQConfig.yaqa`: it defaults via
`default_factory=ViterbiPruningConfig`, accepts a plain dict, and is validated
in `QVQConfig.__post_init__`.

### Mode semantics

- **`auto`** reproduces today's automatic behavior exactly. Eligible
  unconstrained, unweighted, FP16-codebook W2.5/W3 two-bank/four-bank grid
  calls use the norm band; everything else uses the exact baseline recurrence.
- **`off`** deterministically suppresses norm-band dispatch. Every call uses
  the baseline recurrence.
- **`required`** uses the norm band for eligible calls and raises a clear
  error, naming the disqualifying condition, before any silent fallback.

### `fallback` applies to `auto` only

`fallback` decides what `auto` does with a call that cannot prune:
`"baseline"` keeps the exact baseline recurrence, `"error"` raises.
`off` and `required` already determine their own behavior completely, so
pairing either with `fallback="error"` is redundant or contradictory and is
rejected at construction time:

```python
ViterbiPruningConfig(mode="off", fallback="error")       # ValueError
ViterbiPruningConfig(mode="required", fallback="error")  # ValueError
```

The four legal combinations map one-to-one onto the native policy codes in
`gptqmodel/quantization/qvq_pruning.py` (`0` auto, `1` off, `2` auto+error,
`3` required), which are what the CUDA op actually receives.

### Eligibility is not broadened

The eligible set is exactly PR #45's benchmark-supported set and is unchanged:
grid-parallel, non-cooperative, non-midpoint-only, unconstrained, unweighted,
non-family-batched, FP16 codebooks, `bank_count` 2 with `segment_steps` 16 or
`bank_count` 4 with `segment_steps` 32, and `transition_bits` 5 (W2.5) or 6
(W3). W1.5/W2/W3.5 stay on the baseline under `auto` and are rejected under
`required` and under `auto` + `fallback="error"`.

## Precedence over `GPTQMODEL_QVQ_DISABLE_OCTET_GRID`

Explicit configuration is authoritative. The legacy
`GPTQMODEL_QVQ_DISABLE_OCTET_GRID` variable survives only as a deprecated A/B
escape hatch, and only under `mode="auto"`. The variable is re-read on every
dispatch, so mutating it between calls in the same process is observed
deterministically — there is no cached process-global policy:

| `mode` | `fallback` | env set to a disabling value | Result |
| --- | --- | --- | --- |
| `auto` | `baseline` | no | norm band on eligible calls |
| `auto` | `baseline` | yes | baseline everywhere (legacy A/B) |
| `auto` | `error` | no | norm band on eligible calls |
| `auto` | `error` | yes | raises: the call cannot prune |
| `off` | `baseline` | either | baseline everywhere; variable ignored |
| `required` | `baseline` | either | norm band; variable ignored |

## Examples

```python
from gptqmodel.quantization import QVQConfig, ViterbiPruningConfig
from gptqmodel.quantization.config import FORMAT

# Default: unchanged automatic behavior.
QVQConfig(bits=3.0, format=FORMAT.QVQ_V2B2_P32, bank_count=2)

# Deterministic A/B baseline, no environment variable needed.
QVQConfig(
    bits=3.0,
    format=FORMAT.QVQ_V2B2_P32,
    bank_count=2,
    viterbi_pruning=ViterbiPruningConfig(mode="off"),
)

# Strict: fail loudly rather than quietly quantizing on the slow path.
QVQConfig(
    bits=3.0,
    format=FORMAT.QVQ_V2B2_P32,
    bank_count=2,
    viterbi_pruning={"mode": "required"},
)

# Diagnose: keep automatic selection, but refuse to fall back.
QVQConfig(
    bits=3.0,
    format=FORMAT.QVQ_V2B2_P32,
    bank_count=2,
    viterbi_pruning={"mode": "auto", "fallback": "error"},
)
```

## Serialization and compatibility

`QVQConfig._update_output_payload` writes a nested `viterbi_pruning` object
and `from_quant_config` round-trips it. A checkpoint written before this
change has no `viterbi_pruning` key at all; it deserializes to the `auto`
default, which is exactly the pre-policy dispatch. Older readers ignore the
unknown key. There is no `FORMAT` enum change, no weight-layout or packing
change, and no model-format version change.

## Runtime plumbing

The policy is threaded, not merely stored:

```
QVQConfig.viterbi_pruning
  -> QVQProcessor quantization kwargs
  -> quantize_qvq_linear(viterbi_pruning=...)
  -> yaqa_inner / yaqa_inner_v2b2_p32 / yaqa_inner_v2b4_p64
     block_ldlq_inner_v2b2_p32 / block_ldlq_inner_v2b4_p64
  -> _tail_biting_v2_banked_quantize / _batched_v2_banked_viterbi_quantize
  -> viterbi_v2_segment_grid{,_trusted} / _tail_trusted / _family_grid_trusted
  -> qvq_viterbi_v2_segment_banked_cuda_impl(pruning_policy=...)
```

Every new argument defaults to `auto`, in both Python and the native op
schema (`int pruning_policy=0`), so direct low-level callers are unchanged.

The strict policies are enforced at both layers. The native CUDA op refuses
before kernel selection for calls that reach it, and the Python recurrence in
`_batched_v2_banked_viterbi_quantize` refuses before any of its own dispatch
decisions — so a CPU/MPS call, a non-FP32 working dtype, non-contiguous
tensors, or a pre-`sm_80` device raise under `mode="required"` and
`mode="auto"`+`fallback="error"` instead of silently using the baseline
recurrence. `auto`+`fallback="baseline"` and `off` keep every one of those
fallbacks.

### Caveat: `required` is strict by design

Because `required` must never fall back silently, it rejects any call that
cannot use the band, including calls that are structurally ineligible on the
current path rather than "unsupported" in a user-visible sense:

- The second pass of the two-pass tail-biting recurrence is always
  constrained (it is given the wrap-around overlap), so it is never band
  eligible.
- On `sm_80`, the two-bank tail pass runs family-batched, which is also not
  band eligible.

`required` is therefore a diagnostic and A/B tool for direct kernel-level
callers, not a production quantization mode. Production runs should use
`auto` (the default), or `off` for a deterministic pristine comparison.
