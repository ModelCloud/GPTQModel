# QVQ follow-up nits results

## Item 1: MPS symmetric-Gram packing guard

- **MEASURED:** No production change was made for Item 1.
- **MEASURED:** This host uses PyTorch `2.13.0+cpu` at commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`; `torch.backends.mps.is_built()`
  and `torch.backends.mps.is_available()` both return `False`.
- **INFERRED:** `torch.equal` forces an MPS-to-host synchronization rather than
  being lowered to an asynchronous Python boolean. The exact installed PyTorch
  commit registers `equal` as a data-dependent operation returning C++ `bool`,
  dispatches MPS to `mps_equal`, and implements that kernel as
  `at::mps::eq(self, src).all().item().to<bool>()`. The `.item()` is the explicit
  device-result-to-host-scalar read. Evidence:
  [native_functions.yaml](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/native_functions.yaml#L10079-L10086)
  and
  [Equal.cpp](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/mps/operations/Equal.cpp#L15-L23).
- **INFERRED:** The guard's synchronization cost is much smaller in practice
  than an additional drain of all preceding work: the same-thread call enters
  `_TORCH_MLX_BRIDGE_LOCK` a few host instructions later and unconditionally
  calls `torch.mps.synchronize()` before exposing the Torch storage to MLX.
  Therefore the preceding MPS work must finish at that point even without the
  guard. The guard advances the unavoidable wait and adds equality/reduction
  work; this Linux host cannot measure the residual command-buffer cost.
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
- **MEASURED:** No MPS speedup was measured or claimed, and the packed MPS path
  was not run on hardware.

## Item 2: lifecycle damping assertion

- **MEASURED:** The parametrization remains `(1, 1.5)`, so rate coverage is
  unchanged.
- **MEASURED:** The assertion now uses the explicit mapping
  `{1: 0.1, 1.5: 0.1}` and a comment identifies `config.py`'s default
  rate-regularization table as the source of truth.
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
