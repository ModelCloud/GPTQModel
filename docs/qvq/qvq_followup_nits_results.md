# QVQ follow-up nits results

## Item 1: MPS symmetric-Gram packing guard

- **MEASURED:** No production change was made for Item 1.
- **MEASURED:** This host uses PyTorch `2.13.0+cpu` at commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`; `torch.backends.mps.is_built()`
  and `torch.backends.mps.is_available()` both return `False`.
- **MEASURED:** Reading the pinned PyTorch source at the exact installed commit
  shows that `equal` is registered as a data-dependent operation returning C++
  `bool`, dispatches MPS to `mps_equal`, and is implemented as
  `at::mps::eq(self, src).all().item().to<bool>()`. Evidence:
  [native_functions.yaml](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/native_functions.yaml#L10079-L10086)
  and
  [Equal.cpp](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mps/operations/Equal.cpp#L15-L23).
- **INFERRED:** The `.item()` device-result-to-host-scalar read forces an
  MPS-to-host synchronization rather than producing an asynchronous Python
  boolean.
- **MEASURED (cross-vendor source verification):** The independent reviewer
  traced both synchronization paths in the same pinned PyTorch source.
  [`MPSHooks.mm:73-74`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/mps/MPSHooks.mm#L73-L74)
  maps `torch.mps.synchronize()` to
  `getDefaultMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT)`, while
  [`Copy.mm:158`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mps/operations/Copy.mm#L158)
  makes the MPS-to-host scalar copy backing `.item()` issue
  `stream->synchronize(SyncType::COMMIT_AND_WAIT)`. The stream's
  [`commitAndWait()`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/mps/MPSStream.mm)
  fully drains its serial command stream.
- **INFERRED:** Because both calls perform the same `COMMIT_AND_WAIT` operation
  on the same serial stream and nothing is enqueued between them, the guard
  moves the mandatory wait roughly four lines earlier and the later bridge
  synchronization degenerates to a no-op. The residual cost is the guard's
  equality/reduction work, not another drain of all preceding work.
- **INFERRED:** The cross-vendor review estimates that residual GPU work is
  roughly 1-5% of the adjacent Gram formation: both are O(N^2), but the guard
  is bandwidth-bound while Gram formation is compute-bound O(B*N^2), and the
  guard is of the same order as the D2H blit already in the path. The fraction
  should shrink as batch size grows. This is a reasoned estimate made without
  MPS hardware, not a measurement or speedup claim.
- **INFERRED:** Folding the predicate into `accumulate_gradient` is not
  semantics-identical. An accumulator-side guard sees each generated update,
  while the current consumer-side guard validates the exact contiguous tensor
  passed to the packer. It could also detect an error only after a corrupt
  triangle had already been packed.
- **INFERRED:** Moving an on-device predicate into the packer and reading it
  after the bridge synchronization could preserve the exact-matrix check, but
  it would be an unmeasured production-path rewrite whose benefit is uncertain
  given the already-mandatory synchronization. That speculation does not
  justify changing the MPS bridge on this host.
- **MEASURED:** The existing guard remains unchanged, including its NaN
  behavior. The MPS test fixture containing `[[1, nan], [nan, 4]]` still
  collects and skips for lack of MPS, exactly as it did at baseline.
- **INFERRED:** On MPS, that unchanged `torch.equal` guard rejects NaN because
  elementwise equality with NaN is false before the `all()` reduction.
- **INFERRED:** A pre-existing backlog issue remains deliberately out of scope:
  `gptqmodel/utils/qvq_mlx.py` says the matrix must be "finite and exactly
  symmetric", but `torch.equal` rejects NaN rather than all non-finite values.
  A symmetric matrix containing matching `+inf` or `-inf` entries can therefore
  pass the guard despite the message promising finiteness. Production is
  untouched by this PR.
- **MEASURED:** No MPS speedup was measured or claimed, and the packed MPS path
  was not run on hardware.

## Item 2: lifecycle damping assertion

- **MEASURED:** The parametrization remains `(1, 1.5)`, so rate coverage is
  unchanged.
- **MEASURED:** The assertion now uses the explicit mapping
  `{1: 0.1, 1.5: 0.1}` and a comment identifies
  `YAQA_DEFAULT_RATE_REGULARIZATION` in
  `gptqmodel/quantization/qvq_yaqa.py` as the source of truth.
- **MEASURED:** The focused test passes both cases with production unchanged:
  `2 passed`.
- **MEASURED:** Temporarily removing the production assignment to
  `qcfg_clone.yaqa.regularization` made both cases fail with exit code 1:

  ```text
  E       assert 0.0001 == 0.1
  FAILED ...[1]
  FAILED ...[1.5]
  ============================== 2 failed in 3.25s ===============================
  ```

- **MEASURED:** The temporary production mutation was restored; the final
  production source has no diff.

## Required gates

- **MEASURED:** Baseline gate 1 on unmodified `origin/main` at `df80f33e`:
  exit 0, `803 passed, 260 skipped`; cgroup CPU `usage_usec` delta `610502210`.
- **MEASURED:** After gate 1: exit 0, `803 passed, 260 skipped`; cgroup CPU
  `usage_usec` delta `468021771`.
- **MEASURED:** Baseline gate 2 on unmodified `origin/main` at `df80f33e`:
  exit 0, `182 passed, 17 skipped`; cgroup CPU `usage_usec` delta `31466103`.
- **MEASURED:** After gate 2: exit 0, `182 passed, 17 skipped`; cgroup CPU
  `usage_usec` delta `14000767`.
- **MEASURED:** Gate counts after the change exactly match their baselines.
- **MEASURED:** All nine cases in `tests/test_qvq_yaqa_mps.py` skipped before
  and after because this Linux x86-64 host has no MPS device.
- **MEASURED:** Direct imports of `capture_yaqa_sketch_b` and
  `qvq_mlx_pack_symmetric_gram_from_torch_mps` succeeded with exit code 0.
- **MEASURED:** `git diff --check` completed with exit code 0 and no output.
- **MEASURED:** No `OMP_PLACES` override was used.

## Cross-vendor review follow-up gates

- **MEASURED:** Gate 1 after the review nits: exit 0,
  `803 passed, 260 skipped`; cgroup CPU `usage_usec` delta `480160268`.
- **MEASURED:** Gate 2 after the review nits: exit 0,
  `182 passed, 17 skipped`; cgroup CPU `usage_usec` delta `12207708`.
- **MEASURED:** These review-follow-up counts exactly match both the original
  baseline and the first after-change run.
