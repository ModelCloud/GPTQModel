# NF12 HIP device and split-K workspace flow

`nf12_ad_hip.linear_dispatch` selects the NF12 matrix-core route for 16 to 256
tokens on gfx1200/gfx1201. Selection queries the **current HIP device**, so the
binding first checks that `x`, `packed`, and `escapes` share a device, then holds
`device_guard(packed)` through selection, allocation, and launch. The guard
restores the caller's current device when the call returns.

```mermaid
flowchart TD
    A["linear_dispatch receives x, packed, escapes"] --> B{"All three on the same HIP device?"}
    B -- No --> X["TORCH_CHECK raises"]
    B -- Yes --> C["device_guard(packed)"]
    C --> D{"nf12_ad_wmma_supported(m) on packed device?"}
    D -- No --> E["Fused small-M or decode_dispatch fallback"]
    D -- Yes --> F["nf12_ad_wmma_workspace_bytes from wmma_launch_plan"]
    F --> G{"Split-K needs FP32 partial planes?"}
    G -- Yes --> H["torch::empty on packed device and current stream"]
    G -- No --> I["Pass null workspace"]
    H --> J["nf12_ad_linear_wmma validates workspace size"]
    I --> J
    J -- Too small --> Y["hipErrorInvalidValue"]
    J -- Valid --> K["WMMA and optional reduce on current_stream"]
    K --> L["Local workspace returns to PyTorch caching allocator"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,Y failure;
```

`wmma_launch_plan` supplies both the workspace size and the launch's tile,
M-block, and split counts. Split-K uses one FP32 partial plane per split. The
workspace is a local PyTorch tensor, so separate calls on different streams or
devices receive separate live storage. The allocator can reuse the storage
after the call's queued work finishes on its stream. No decoded weight is kept.

| Mechanism | Before | Current |
| --- | --- | --- |
| WMMA device selection | `nf12_ad_wmma_supported` read the ambient device before a guard (`58b00cc:gptqmodel_ext/nf12_hip/nf12_ad_torch.cpp:265`; `58b00cc:gptqmodel_ext/nf12_hip/nf12_ad_wmma.hip:346-350`) | `linear_dispatch` validates tensor devices and guards `packed` before routing (`gptqmodel_ext/nf12_hip/nf12_ad_torch.cpp:197-203`) |
| Split-K scratch | One process-wide `g_partial` pointer was reused by all calls and streams (`58b00cc:gptqmodel_ext/nf12_hip/nf12_ad_wmma.hip:353-376`) | `linear_dispatch` owns a local allocator tensor, sized by the shared launch plan (`gptqmodel_ext/nf12_hip/nf12_ad_torch.cpp:285-298`; `gptqmodel_ext/nf12_hip/nf12_ad_wmma.hip:412-449`) |
| Invalid inputs | WMMA could launch with tensors on different devices, or reuse scratch from another device | A tensor-device mismatch raises before dispatch; an undersized workspace returns `hipErrorInvalidValue` |

The independent Torch FP32 oracle in `tests/test_nf12_ad_codec.py` checks the
multi-device and concurrent-stream results on suitable RDNA4 hosts. The
`m <= 15` fused route and the `m > 256` decode-then-GEMM route retain their
existing math and packed-weight storage.

Introduced in #3635.

## Activation layout and output shape

Every HIP route indexes rows of a contiguous `[m, k]` matrix: the fused
small-M kernel, the matrix-core kernel, and the decode-then-GEMM slices all
take a row pointer plus `k`. A sliced activation (`storage[:m, :k]` of a wider
allocation) or a 3-D activation therefore cannot be forwarded as-is. The
op layer normalizes once at entry and restores the caller's leading
dimensions on every return.

```mermaid
flowchart TD
    A["linear_dispatch receives x of any shape"] --> B["k = x.size(-1), m = x.numel() / k"]
    B --> C["x2 = x.contiguous().reshape({m, k})"]
    C --> D["out_sizes = x.sizes(); out_sizes.back() = n"]
    D --> E{"m < 16 and fused small-M supported?"}
    E -- Yes --> F["nf12_ad_linear_fused on x2"]
    F --> F1["return out.reshape(out_sizes)"]
    E -- No --> G{"nf12_ad_wmma_supported(m)?"}
    G -- Yes --> H["nf12_ad_linear_wmma on x2 with caller-sized workspace"]
    H --> H1["return out.reshape(out_sizes)"]
    G -- No --> I["decode_dispatch per row slice"]
    I --> J["at::matmul / at::addmm on x2"]
    J --> K["cat slices, then return dense.reshape(out_sizes)"]
    classDef entry fill:#e8f0ff,stroke:#003f8a,color:#111;
    class C,D entry;
```

| Mechanism | Before | Current |
| --- | --- | --- |
| Fused small-M eligibility | Required `x.dim() == 2 && x.is_contiguous()`, so a 3-D or sliced input silently took a slower route (`08cf3857b:gptqmodel_ext/nf12_hip/nf12_ad_torch.cpp:214`) | Route decision uses only `m`, after `x2 = x.contiguous().reshape({m, k})` (`gptqmodel_ext/nf12_hip/nf12_ad_torch.cpp:218-221`) |
| Kernel input pointer | Passed `x.data_ptr()` to the fused and WMMA kernels (`08cf3857b:...:240`, `:290`) | Passes `x2.data_ptr()`, which is contiguous by construction (`...:246`, `:296`) |
| GEMM operand | `at::matmul(x, ...)` / `at::addmm(bs, x, ...)` (`08cf3857b:...:338`) | `at::matmul(x2, ...)` / `at::addmm(bs, x2, ...)` (`...:339-341`) |
| Output shape | Returned `[m, n]`, flattening `[2, 8, 256]` to `[16, 256]` (`08cf3857b:...:245`, `:296`, `:338`) | Returns `out.reshape(out_sizes)` / `dense.reshape(out_sizes)` (`...:251`, `:302`, `:345`) |

Failure path: a shape the kernels cannot honour (`n % 16 != 0`, `k % 64 != 0`,
`packed` not contiguous) still raises `TORCH_CHECK` before any launch, so the
normalization never converts a rejection into a wrong answer. The raw C ABI
keeps rejecting a padded `ldx` explicitly; only the torch op normalizes.

`tests/test_nf12_ad_codec.py` asserts the 3-D shape round-trip, the padded
row-slice result against `F.linear`, and the `NF12ADLinear` module shape on a
RDNA4 host.
