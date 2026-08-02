# 5/6/7-bit GPTQ support plan (GPT-QModel-Ultra)

## Verdict: yes, it's feasible

Nothing in the GPTQ math blocks it: the quantizer is bit-agnostic (`maxq = 2**bits - 1`), and
`BaseQuantizeConfig.bits` already declares `choices=[2, 3, 4, 5, 6, 8]`
(`gptqmodel/quantization/config.py:2693`) — the config layer accepts 5/6 today; 7 only needs
adding to that `choices` list. What's missing
is everything downstream: packing, serialization layout, and kernels. The repo also already
solved the "bits don't divide 32" problem once, for 3-bit, so all the required patterns exist:

- Continuous bit-stream pack/unpack with cross-word carries (3-bit "10-1-10-1-10" scheme in
  `nn_modules/qlinear/__init__.py`, native parity in `gptqmodel_ext/pack_block_cpu.cpp`).
- A dedicated native kernel (`trilin` for 3-bit) + Triton fused fallback (`triton_utils/three_bit.py`).
- The **expand-and-run-Marlin trick**: `prepare_marlin_3bit` expands exact 3-bit codes to
  `uint4b8` and runs the stock 4-bit Marlin kernel (`three_bit.py:542-599`).
- GGUF already ships 5/6-bit (`q5_k`, `q6_k`) using 4-bit planes + high-bit plane — the
  precedent for the storage layout recommended below.

## The divisibility problem, and the two layout options

None of 5, 6, 7 divides 32, so a `pack_factor = 32 // bits` layout doesn't exist. Two options:

### A. Continuous bit-stream (what 3-bit does today)
- 5-bit: lcm(5, 32) = 160 → 32 codes per five 32-bit words; codes straddle word boundaries.
- 6-bit: lcm(6, 32) = 96 → 16 codes per three 32-bit words.
- 7-bit: lcm(7, 32) = 224 → 32 codes per seven 32-bit words (worst carry pattern of the three).
- Pro: densest possible, one qweight tensor, mirrors the existing 3-bit scheme.
- Con: cross-word carries make GPU dequant branchy and misaligned; this is exactly why the
  3-bit path needed a bespoke native kernel (trilin) to be fast.

### B. Split-plane / bit-plane packing (recommended)
Store the code as two aligned planes that each individually pack cleanly:
- **5-bit = 4 + 1**: a standard 4-bit plane (`qweight_lo`, pack_factor 8) + a 1-bit high plane
  (`qweight_hi`, pack_factor 32). Kernel: `q = lo | (hi << 4)`.
- **6-bit = 4 + 2**: 4-bit plane + 2-bit plane (pack_factor 16). `q = lo | (hi << 4)`.
- **7-bit = 4 + 2 + 1**: 4-bit plane + 2-bit plane + 1-bit plane.
  `q = lo | (mid << 4) | (hi << 6)`. Three planes, but every plane is word-aligned and reuses
  existing 4/2/1-bit pack code; the extra plane costs exactly its 1 bit/weight of bandwidth.
  (Alternative 4+3 rejected: a 3-bit plane reintroduces continuous-packing carries.)

Pros:
- Every load is word-aligned; no carries, no tail special cases beyond existing 4/2/1-bit ones.
- Reuses the mature, vectorized 2/4-bit pack/unpack machinery on CPU and GPU nearly verbatim.
- GPU dequant stays branch-free: the fast 4-bit LOP3/bit-twiddle dequant sequences apply to the
  low plane; the high plane adds one extra 32-bit load per 32 (or 16) codes + shift/or — the
  extra bandwidth is exactly the 1 or 2 bits/weight, so you keep the full 5/6-bit memory-BW win.
- Same trick GGUF q5_k/q6_k and several ExLlama layouts use; known-good on GPUs.

Con: two qweight tensors per layer (or one concatenated tensor with an offset) — a new
serialized layout. Since no upstream GPTQ ecosystem (AutoGPTQ, vLLM, Marlin) defines a 5/6-bit
GPTQ layout, we're free to define it; suggest gating it behind `FORMAT.GPTQ_V2` metadata
(e.g. store planes as `qweight` + `qweight_hi`) so v1 loaders fail loudly rather than misread.

Decision needed: A vs B. Recommendation is **B** — layout A's only advantage is a single
tensor, and the 3-bit experience shows continuous layouts push all the complexity into the
kernels.

## Why GPU inference can still be fast

Weight-only quant decode is memory-bandwidth-bound. 5/6/7-bit cut bytes vs fp16 by 56–69% and
beat 8-bit by 12–37%; the dequant ALU cost (mask/shift/or + LOP3 + fma) is hidden under the
loads as long as accesses stay aligned — which split-plane guarantees. Concretely:

1. **Day-1 correct + already-fast path (small work): expand-to-8-bit Marlin.**
   Copy `prepare_marlin_3bit`: at `post_init`, expand exact 5/6/7-bit codes to `uint8b128`
   (8-bit) and run the existing 8-bit Marlin kernel (`SUPPORTS_BITS = [4, 8]` in `marlin.py`).
   Disk/VRAM-at-rest stays 5/6-bit; the runtime cache runs at 8-bit Marlin speed (already much
   faster than fp16). Zero new CUDA. Same trick works for Machete (4/8-bit).
2. **Native speed (larger work): split-plane dequant in kernels.**
   - Triton first: extend `TritonV2Linear`'s dequant kernel with the two-plane load
     (`q = (lo >> s_lo) & 0xF | (((hi >> s_hi) & mask) << 4)`); fully expressible in Triton,
     no cross-word logic. This gives a real 5/6-bit-bandwidth GPU path cheaply.
   - Then, if benchmarks justify it, a native CUDA kernel modeled on trilin/Marlin: keep the
     4-bit fragment pipeline (cp.async + LOP3 dequant + mma) and add the high-plane fetch.
     This is the only step that needs serious kernel engineering.
3. BitBLAS is a possible extra: it supports arbitrary `uint` widths in principle, but the repo
   pins `BITBLAS_SUPPORTED_BITS = [1, 2, 4, 8]`; treat as optional follow-up.

## Work plan (phased, each phase independently shippable)

**Phase 1 — quantizer + config (small)**
- Verify `find_params`/GPTQ solve with `bits=5,6,7` (should already work; `maxq = 2**b - 1`).
- Config: bits 5/6 already in `choices`; add 7 to the list; add validation/serialization round-trip tests
  (`tests/qcfg/`). Decide and record the format contract (GPTQ_V2-only, sym/asym, group sizes).

**Phase 2 — packing + reference backend (the core)**
- Implement split-plane pack in all four packers with bitwise parity, per
  `quantization_packing.md` invariants (saturate-then-pack, never mask-as-clamp):
  Python `pack_block` / original packer in `nn_modules/qlinear/__init__.py`, GPU pack path,
  and native `gptqmodel_ext/pack_block_cpu.cpp`. Same treatment for `qzeros`.
- Extend `TorchLinear` dequant (`torch.py`) with the 5/6/7-bit branches → reference backend,
  `SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]`.
- Gates to update: `moe_dispatch.py:283` (`bits not in (2,3,4,8)`), `awq_processor` guard,
  importer capability maps (auto via `SUPPORTS_*`).
- Tests: extend `tests/test_pack.py`, `test_pack_block_cpu.py`, `test_packing_matrix.py`
  adversarial endpoint matrix to 5/6/7; save/load/generate smoke test.

**Phase 3 — fast GPU inference**
- 3a: expanded-Marlin runtime cache (copy the 3-bit pattern) → immediate production-quality
  speed on Ampere+.
- 3b: TritonV2 multi-plane dequant kernel (two planes for 5/6, three for 7) → true native bandwidth.
- 3c (optional): native CUDA kernel (trilin-style module, e.g. `pentlin`/`hexlin`) if decode
  benchmarks show the Marlin-8-bit cache leaving >15% on the table. Note 7-bit gains the
  least here (only 12.5% below 8-bit), so the expanded-Marlin cache may be its permanent path.
- Benchmarks in `scripts/` per AGENTS rules (decode + prefill shapes, vs fp16 + 4-bit + 8-bit).

**Phase 4 — quality & ecosystem**
- Perplexity/lm-eval ladder 4 vs 5 vs 6 vs 7 vs 8 bit on a small model to document the
  accuracy/size trade-off (near-8-bit quality at well below 8-bit size); confirm 7-bit earns
  its place vs 8-bit before investing in a native 7-bit kernel.
- Docs + `test_bits.py` matrix; explicitly leave vLLM/sglang export unsupported with a clear
  error until/unless the layout is upstreamed.

## Open decisions for you
1. Layout A (continuous, like 3-bit) vs **B (split-plane, recommended)**.
2. Serialized tensor naming for the high plane (`qweight_hi` vs concatenated single tensor).
3. Is the Phase-3a expanded-8-bit Marlin cache acceptable as the initial "fast" path, with the
   native kernel deferred until benchmarked?
