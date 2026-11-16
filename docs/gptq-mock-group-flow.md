<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GPTQ mock quantization group flow

`GPTQ.quantize` selects the mock path when its configuration requests it or a
loss recovery retries without a Hessian inverse. For dynamic groups, it uses
group-sized blocks so every parameter set and output slice cover the same
columns. For static groups, it resolves each reordered column back to its
original group before quantizing. Corrected in PR #3720.

```mermaid
flowchart TD
    A["GPTQ.quantize enters mock path"] --> B{"static_groups?"}
    B -- No --> C{"group_size positive?"}
    C -- Yes --> D["Set mock_blocksize to group_size"]
    D --> E["find_params for each group and quantize its columns"]
    C -- No --> F["Quantize ungrouped block"]
    B -- Yes --> G["Resolve original column via perm or final_perm"]
    G --> H["Select groups by original group index"]
    H --> I["Quantize each reordered column"]
    E --> J["Reorder Q and group metadata for export"]
    F --> J
    I --> J
```

```mermaid
flowchart LR
    A["Old fixed-size mock block"] --> B{"Group crosses block end?"}
    B -- Yes --> X["Q1 index out of bounds"]
    B -- No --> C["Quantize with per-group metadata"]
    classDef failure fill:#fdd,stroke:#c33
    class X failure
```

| Mechanism | Before | After |
| --- | --- | --- |
| Dynamic group boundary | `gptqmodel/quantization/gptq.py:2210` used the requested block size, while `:2223` extended a group past it | `mock_blocksize` is the group size for dynamic groups |
| Static group lookup | `gptqmodel/quantization/gptq.py:2232` used the current column unless `desc_act` was enabled | The group-aware `final_perm` is also resolved to the original column |
| Output and metadata | A crossing group could overrun `Q1` before export | Each group writes only its own columns and one metadata entry |

The boundary and group-aware regressions live in
`tests/test_gptq_mock_group_order.py`. A final short group remains within its
last block; no extra scale or zero-point entry is emitted.
The packed roundtrip in `tests/test_gptq_static_group_order.py` also runs in
mock mode after PR #3719, checking the combined group reordering, metadata,
packed words, and decoded weight against an independent affine oracle.

## Retry ordering snapshot

The NaN-loss retry re-enters the mock path with `mock_quantization = True`.
`GPTQ._quantize_impl` captures `retry_hessian_diag` from the natural-order
Hessian diagonal before `desc_act` / `act_group_aware` permutes `H` and `W`
(`gptqmodel/quantization/gptq.py:2311`). The retry restores
`H = torch.diag(retry_hessian_diag)`, and because it clones `W` in natural
column order it recomputes the same permutation as the primary path.

```mermaid
flowchart TD
    A["finalize_hessian"] --> B["capture retry_hessian_diag (natural order)"]
    B --> C["neutralize dead columns"]
    C --> D["desc_act / act_group_aware permutation"]
    D --> E["primary blockwise update"]
    E --> F{"avg_loss is NaN"}
    F -- yes --> G["H = diag(natural-order diagonal)"]
    G --> H["mock retry re-permutes W and H identically"]
    H --> I["g_idx and column order match the primary path"]
    style I fill:#d1e7dd,stroke:#0f5132,color:#0f5132
```

| Mechanism | Before | After |
| --- | --- | --- |
| Retry diagonal source | `gptq.py:2397` captured `self.H.diagonal()` after permutation | `gptq.py:2311` captures it before permutation |
| Retry permutation | `argsort` of an already-sorted diagonal was the identity, so the retry quantized in natural order | The retry recomputes the primary permutation from the same natural-order diagonal |
| `g_idx` | Not act-ordered on retry | Equal to the primary path's `g_idx` |
