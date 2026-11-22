<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GPTQ Hessian accumulation

`GPTQ.add_batch` records new Hessian partials. `GPTQ.materialize_global_hessian`
combines those partials with the already normalized `H` when capture resumes
after a previous finalization. `GPTQ.finalize_hessian` returns that accumulated
matrix to the quantizer. `GPTQ.adopt_hessian_from` copies the leader's sample
count as well as its matrix. This flow was corrected in PR #3721.

```mermaid
flowchart TD
    A["GPTQ.add_batch records new partials"] --> B["GPTQ.finalize_hessian"]
    AA["GPTQ.adopt_hessian_from copies H and sample count"] --> B
    B --> C["GPTQ.materialize_global_hessian"]
    C --> D{"_hessian_dirty?"}
    D -- No --> E["Return existing H"]
    D -- Yes --> F["total_samples = _hessian_total_samples + new_samples"]
    F --> G{"total_samples == 0?"}
    G -- Yes --> Z["Store zero H and zero sample count"]
    G -- No --> H["Weight previous H by previous_samples / total_samples"]
    H --> I["Add each device partial with 2 / total_samples"]
    I --> J["Store H, nsamples, and _hessian_total_samples"]
    classDef empty fill:#fdd,stroke:#c33
    class Z empty
```

| Mechanism | Before | After |
| --- | --- | --- |
| Sample count | `gptqmodel/quantization/gptq.py:1141` counted only new partials | `materialize_global_hessian` includes `_hessian_total_samples` |
| Adopted sample count | `gptqmodel/quantization/gptq.py:1290` copied `H` and `nsamples` but omitted `_hessian_total_samples` | `adopt_hessian_from` copies the cumulative count |
| Reused output | `gptqmodel/quantization/gptq.py:1155` zeroed the previous `H` | Previous normalized `H` contributes with its sample weight |
| Normalization | `gptqmodel/quantization/gptq.py:1219` scaled only the new sum | Each new partial contributes `2 / total_samples` before final storage |

When there are no prior or new samples, `materialize_global_hessian` stores a
zero matrix. When no new samples are dirty, it returns the existing normalized
matrix. The regression is `tests/test_gptq_hessian_accumulation.py`.

## Snapshot ownership

`snapshot_hessian` is the read path for the durable EXL3 capture frontier: a
writer may hold the returned tensor while capture continues. The value must
therefore own its storage, because `materialize_global_hessian` mutates `self.H`
in place on the next finalization and the embedding path accumulates into
`self._H_diag` the same way.

```mermaid
flowchart TD
    A["snapshot_hessian(target_device)"] --> B{"embedding capture?"}
    B -- Yes --> C{"_hessian_dirty?"}
    C -- No --> D["result = _H_diag.detach().to(target)"]
    D --> E{"same storage as _H_diag?"}
    E -- Yes --> F["result = result.clone()"]
    E -- No --> G["defer the source until the copy completes"]
    B -- No --> H{"_hessian_dirty?"}
    H -- No --> I["result = H.detach().to(target)"]
    I --> J{"same storage as H?"}
    J -- Yes --> K["result = result.clone()"]
    J -- No --> L["defer the source until the copy completes"]
    C -- Yes --> M["merge finalized + pending into a fresh tensor"]
    H -- Yes --> M
    F --> N["return a storage-independent snapshot"]
    G --> N
    K --> N
    L --> N
    M --> N
    style F fill:#dfd,stroke:#060
    style K fill:#dfd,stroke:#060
```

`Tensor.to(device=..., dtype=torch.float32)` is a no-op that returns the same
storage when the source already has the requested device and dtype, and
`.detach()` shares storage rather than copying it. Both fast paths therefore
aliased the live capture:

| Mechanism | Before | After |
| --- | --- | --- |
| Matrix fast path (`gptqmodel/quantization/gptq.py:1449`) | `H.detach().to(device, float32)` returned a view of `H` | clone when the move is a no-op |
| Embedding fast path (`gptqmodel/quantization/gptq.py:1398`) | `_H_diag.detach().to(device, float32)` returned a view of `_H_diag` | clone when the move is a no-op |
| Merge path | already built a fresh tensor | unchanged |

Measured on CPU (torch 2.16.0.dev20261102+cu130), same-device FP32 snapshots:

| Capture | Snapshot at first call | Snapshot after `add_batch` + `finalize_hessian` | Live value after |
| --- | --- | --- | --- |
| `nn.Linear(2, 2)` | `[[2, 0], [0, 0]]` | `[[2, 0], [0, 0]]` (was `[[1, 0], [0, 4]]`) | `[[1, 0], [0, 4]]` |
| `nn.Embedding(5, 3)` | `[0.667, 1.333, 0, 0]` | `[0.667, 1.333, 0, 0]` (was `[0.4, 0.8, 0.4, 0.4]`) | `[0.4, 0.8, 0.4, 0.4]` |

The snapshot is also writable: a caller may keep, transform, or release it
without the capture observing the change. Tests:
`tests/test_gptq_hessian_accumulation.py::test_gptq_snapshot_hessian_survives_later_finalization`,
`::test_gptq_snapshot_hessian_embedding_survives_later_finalization`, and
`::test_gptq_snapshot_hessian_is_writable_without_touching_the_capture`.

## desc_act sink-row cap

`desc_act=True` disables `act_group_aware`, so a few massive-activation
("attention sink") rows own the largest Hessian diagonal and `desc_act`
saturates the columns correlated with them (issue #4742).
`GPTQ.resolve_sink_row_cap_threshold` bounds each row's norm at
`min(p99.9, HessianConfig.sink_row_cap_median_multiple x median)` and
`GPTQ.compute_hessian_xtx` scales the row before the fp32 `X^T X` GEMM.
`add_batch` runs inside the layer forward hooks, so the whole decision is
tensor-only: a `bool()`/`float()` there drains the CUDA queue on every batch
(issue #4950).

```mermaid
flowchart TD
    A["compute_hessian_xtx(matrix)"] --> B["resolve_hessian_chunk_size"]
    B --> C["resolve_sink_row_cap_threshold(matrix, chunk_size)"]
    C --> D{"sink_row_cap_median_multiple()"}
    D -- None --> E["no cap"]
    D -- "N > 0" --> F["_sink_row_norms: fp32 row L2, chunk-staged"]
    F --> G["torch.sort(norms) once"]
    G --> H["threshold = min(ordered[p99.9], N x ordered[median])"]
    H --> I{"every norm finite and threshold > 0?"}
    I -- No --> J["threshold = +inf, so every scale is 1"]
    I -- Yes --> K["keep threshold"]
    K --> L["_sink_row_cap_scale_from_threshold: clamp(threshold / norm, max=1)"]
    J --> L
    L --> M{"chunk_size is None?"}
    M -- Yes --> N["fp32 fast path: (X * s)^T (X * s)"]
    M -- No --> O["chunk lease: (X_c * s_c)^T (X_c * s_c) per chunk"]
    classDef bad fill:#fdd,stroke:#c33
    class J bad
```

| Mechanism | Before (#4909) | After |
| --- | --- | --- |
| Cap decision | four host reads per `add_batch` (`bool(isfinite(...))`, `float(threshold)`) | tensor-only `torch.where`; a non-finite or non-positive threshold becomes `+inf` |
| Capped batch layout | left the fp32 `matmul` fast path for the chunk-workspace lease (`chunk_size = rows`) | folded into the fp32 fast-path GEMM |
| Quantiles | two `kthvalue` calls | one `torch.sort` |
| Numerics | per-batch `min(p99.9, N x median)` | unchanged: bitwise `H`, same PPL |

Caching the threshold on the first batch (the cheapest way to skip the sort) is
rejected: it decides later batches against a stale bound, and on
Qwen2.5-0.5B-Instruct it moved wikitext-2 PPL from 20.32 to 20.78. The
regression is `tests/test_descact_cap_sync_4950.py`; the quality contract stays
`tests/test_descact_sink_4742.py`.

## Shared-input dedup for calibrated quantizers

`SharedInputPlan` already groups sibling projections that read the same
activation (`:in=<tag>`). `GPTQProcessor` turned that plan into a leader/
follower capture; `SharedInputCaptureMixin` gives QQQ, VPTQ, MACaQ and EXL3 the
same election, and GPTAQ/FOEM now participate in `GPTQProcessor`'s election
instead of being excluded.

```mermaid
flowchart TD
    A["stage_subset: subset_names"] --> B["processor.begin_shared_input_capture"]
    B --> C["_resolve_shared_input_plan(model)"]
    C --> D{"SharedInputPlan has explicit groups?"}
    D -- No --> Z["No followers; every module captures"]
    D -- Yes --> E["elect_shared_input_leaders(signature_for, enabled_for)"]
    E --> F{"signatures equal?"}
    F -- No --> G["warn; follower captures independently"]
    F -- Yes --> H["_shared_input_leaders = follower -> leader"]
    H --> I["pre_process_fwd_hook: follower hook is a no-op"]
    I --> J["end_shared_input_capture"]
    J --> K["follower.adopt_hessian_from(leader)"]
    K --> L["copy leader H (+ dXXT for GPTAQ/FOEM)"]
    classDef fail fill:#fdd,stroke:#c33
    classDef ok fill:#dfd,stroke:#060
    class G fail
    class L ok
```

The quantizer owns the copy because each keeps a different statistic:
`QQQ` a token-normalized float32 mean (`gptqmodel/quantization/qqq.py:705`),
`VPTQ` a float64 mean (`gptqmodel/quantization/vptq.py:414`), `MACaQ` a
float64 sum (`gptqmodel/quantization/macaq.py:740`), and `GPTAQ`/`FOEM` an
`H` plus `dXXT` (`gptqmodel/quantization/gptaq.py:310`,
`gptqmodel/quantization/foem.py:384`). `EXL3Processor` exposes its plain
`GPTQ` capture tasks through `_shared_input_task_enabled`
(`gptqmodel/looper/exllamav3_processor.py:2298`) and excludes `w3` aliases
already deduplicated by `share_hessian_state_from`.

| Mechanism | Before | After |
| --- | --- | --- |
| QQQ / VPTQ / MACaQ election | no plan; every sibling accumulated `H` | `SharedInputCaptureMixin.begin_shared_input_capture` (`gptqmodel/looper/shared_input.py:153`) |
| EXL3 election | plain `GPTQ` capture per projection | same mixin over `tasks[name]["capture"]` |
| GPTAQ / FOEM election | excluded by `type(task) is GPTQ` | `_hessian_capture_signature` (`gptqmodel/looper/gptq_processor.py:435`) admits them with `dXXT` |
| QQQ GSQ | n/a | excluded: its moments, not `H`, drive quantization |
| Follower capture | one `add_batch`/`addmm_` per sibling | hook is a no-op; leader copy is bitwise equal |

Measured on the synthetic 2-layer Llama fixture used by
`tests/test_shared_input_hessian_dedup_methods.py`, `batch_size=2`:

| Method | Captures per layer before | Captures per layer after | Checkpoint vs `origin/main` |
| --- | --- | --- | --- |
| QQQ | 7 (`q,k,v,o,g,u,d`) | 4 (`q,o,g,d`) | bitwise equal, 36 tensors |
| VPTQ | 7 | 4 | bitwise equal, 49 tensors |
| MACaQ | 7 | 4 | bitwise equal, 35 tensors |
| GPTAQ | 7 `H` + `dXXT` | 4 | bitwise equal, 63 tensors |
| FOEM | 7 `H` + `dXXT` | 4 | bitwise equal, 63 tensors |

The two follower matrices per group are the same storage the independent path
would have held (leader statistic plus the follower's private copy), so peak
allocated memory does not grow with the group size. CUDA peak measured on the
same fixture stayed within 0.02% of `origin/main` for every method (allocator
noise; no directional regression).

## Hessian release before the block pass

`GPTQ.quantize` needs the full fp32 `columns x columns` Hessian only to build
`Hinv` and `P`. The column loop that follows reads neither `H` nor the
permuted weight copy, but three paths kept a second reference alive for the
whole module (issue #5717):

- `desc_act` and `act_group_aware` reorder into a local `H_permuted`, assign it
  to `self.H`, and then never rebind the local, so the same storage had two
  owners through the block pass.
- `GPTAQ` and `FOEM` pass their local `H` to `hessian_inverse`; the inverse
  closures (retry, delta correction) capture that cell, so the local must be
  rebound after the inverse for the tensor to become unreachable.

```mermaid
flowchart TD
    A["materialize H: columns x columns fp32"] --> B{"desc_act or act_group_aware?"}
    B -- Yes --> C["H_permuted = reorder(H)"]
    C --> D["self.H = H_permuted"]
    D --> E["del H_permuted, W_permuted"]
    B -- No --> F["self.H = H"]
    E --> G["hessian_inverse(H) -> Hinv, P"]
    F --> G
    G --> H{"GPTAQ / FOEM?"}
    H -- Yes --> I["H = None after the inverse"]
    H -- No --> J["block pass reads Hinv/P only"]
    I --> J
    J --> K["Hessian unreachable before the first column"]
    classDef ok fill:#dfd,stroke:#060
    class E,I,K ok
```

| Mechanism | Before | After |
| --- | --- | --- |
| `desc_act` ordering | `gptqmodel/quantization/gptq.py:3247` kept `H_permuted` bound next to `self.H` | `del H_permuted, W_permuted` after the assignment (`gptq.py:3249`) |
| `act_group_aware` ordering | `gptqmodel/quantization/gptq.py:3285` kept the same local | same release (`gptq.py:3290`) |
| GPTAQ local Hessian | `gptqmodel/quantization/gptaq.py:635` stayed captured by the inverse closures | rebind `H = None` after the inverse |
| FOEM local Hessian | `gptqmodel/quantization/foem.py:696` stayed captured by the inverse closures | rebind `H = None` after the inverse |

The release changes lifetime only. On the synthetic `512x64` module in
`tests/test_hessian_block_pass_release_5717.py`, the weakref spy saw the
Hessian alive during 512/512 per-column `Quantizer.quantize` calls before and
0/512 after for `desc_act`, `act_group_aware`, GPTAQ, and FOEM; natural GPTQ
was already 0/512. That second buffer is `columns ** 2 x 4` bytes: 1 MiB at
512 columns, 64 MiB at 4096, 784 MiB at 14336, and 3.06 GiB at 28672. The
packed outputs are bitwise identical to `origin/main` for all five orderings
(same fixture, same seed), because the same tensors feed the inverse and only
their lifetime changes; CUDA peak VRAM was not measured on this host.

The regression is `tests/test_hessian_block_pass_release_5717.py`. The
`desc_act` and `act_group_aware` branches also release the permuted weight
local: `W = W_permuted` is the only surviving reference the block pass needs.
