# NVIDIA Hymba source provenance

This is GPT-QModel's package-owned adaptation of the actual Hymba hybrid
attention/state-space architecture. It does not delegate to Mamba, Jamba,
checkpoint Python, or a dynamically downloaded module.

The authoritative source is
[`nvidia/Hymba-1.5B-Instruct` at `bc2d2b43bc73d62ad8bfc9e6a3dcc26ab0795e77`](https://huggingface.co/nvidia/Hymba-1.5B-Instruct/tree/bc2d2b43bc73d62ad8bfc9e6a3dcc26ab0795e77).
The model card links [NVlabs/hymba](https://github.com/NVlabs/hymba); its inspected
HEAD was `df5b2a95b9905534eb162305a5414bc43e37e84b`. The checkpoint's own pinned
configuration/modeling files are the behavioral source, rather than a moving
repository branch. They were inspected as text, never imported or executed.

| Source | SHA-256 |
| --- | --- |
| `configuration_hymba.py` | `22b7283f1f12a71c987f126a4fadfcf15ef890a229fc68d5ee2faf443114accd` |
| `modeling_hymba.py` | `4393c3e2097b58b6e1e137dad0ee6289481dcaab7db5b3415f8cfd5b7f0e45cf` |
| `config.json` (retained as `upstream_config.json`) | `5cf3090f104443f7a276453c02d4095539d2e8854bed084a1d8062949f0b0c54` |
| Official NVIDIA June 2024 license PDF | `396d7de220f6b0e6fcfe33836f1b99a9943769b97c56330444f8efe1b441f607` |
| Original safetensors JSON header | `dd30d34a3c0d89e727ad4d747cd54cd39562facb3dd88a557a81dff6ff659a01` |

The model card specifies the **NVIDIA Open Model License Agreement, June 2024**;
the unmodified official PDF is included as `LICENSE.pdf`. `NOTICE` also credits
the Hugging Face Transformers/Apache 2.0 origins of attention, normalization
and MLP conventions. This port does not relabel NVIDIA's source as Apache-only.

`checkpoint_shapes.json` contains all 611 tensor names and shapes obtained by an
HTTP byte-range read of the pinned `model.safetensors` header. No trained tensor
payload was downloaded. `test_published_1_5b_state_dict_matches_every_checkpoint_key_and_shape`
builds the full 32-layer configuration on meta and compares every tensor and
shape, accounting explicitly for the tied `lm_head.weight` alias omitted from
the safetensors file. No synthesized model tensors are treated as checkpoint
ownership obligations.

Implementation adaptations preserve fused projection row ordering, grouped
query/value widths, per-layer convolution/SSM states, Q/K norms, rotary/NTK
conventions, shared-KV producers, memory-token registers, residual paths and
MLP tensor names. The portable recurrence is the Mamba selective-scan equation,
including delta bias/softplus, input-dependent B/C, D skip and SiLU output gate.
It replaces unavailable compiled kernels with ordinary Torch operations.

The adaptation deliberately fixes upstream implementation hazards: mutation of
`dt_proj.bias` during forward, multi-token continuation, interspersed memory insertion at continuation
boundaries, and continuation after a partially failed forward. Original
execution hints (`flex`, `use_mamba_kernels`) remain configuration data; this
port executes Torch SDPA plus the reference recurrence. See `COVERAGE.md` for
qualification and unsupported operations.
