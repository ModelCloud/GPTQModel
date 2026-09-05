# Algebraic reference implementations

These are CPU reference prerequisites, not completed GPU experiments or model-quality claims.

- Experiments 21/22/24/25/26: compact affine scan, symbolic all-start transfer,
  multi-symbol, bit-sliced, and GF(2) state recovery. Actual P32 has 16 state bits
  (65,536 states), not 16 states. Symbolic all-start transfer avoids enumerating
  every start state; it is not the proposed redundant GPU all-start implementation.
  The existing direct circular-window decoder remains the performance comparator.
- Experiments 19/20/29: rank-constrained fits to actual callable output residuals,
  alternating native-quantizer callback, sparse correction, and storage interfaces.
  Integration must supply the deployed native operator and verified calibration data.
- Experiments 15/16/27/28/30: additive/tensor-product codebooks, signed/ternary bases,
  and local sparse Walsh references. Storage estimates describe proposed payloads,
  not deployed checkpoint serializers or complete model BPW.

Validation: 138 CPU tests passed; the opt-in snapshot check was then run separately
and passed on 282 first/middle/last tiles across all 94 P32 projections. Each of five
state-recovery candidates matched canonical states and banked decoded values exactly.
The historical F6 seed-7 snapshot was read only. No GPU speed or downstream quality
conclusion follows from these checks. Ruff and whitespace checks passed.

The model harness now supports window inner-kernel replacement while preserving
production transforms/output boundaries, with configurable minimum row count and
explicit runtime cache-byte reporting. The layer harness now honors `--module`;
older profiler reports contain the worker's three assigned modules and must be read
by their recorded rows, not attributed to the previously ignored requested module.
