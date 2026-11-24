<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 22–24 audit

28 PRs merged to main between 2026-11-22 01:00 and 2026-11-24 01:00 UTC.
The [previous audit](2026-11-23-regressions.md) covers the overlapping
measurement, model-port and packing changes. The examples below come from
landed numerical fixes. [Audited revision](https://github.com/ModelCloud/GPTQModel/commit/38be95ec7900d01a31b9d25e31c46121057645a2)

## The grid must survive its consumer

| Fixed path | Fixture and observable result | Test family |
| --- | --- | --- |
| FP16 replay in [RTN #6001](https://github.com/ModelCloud/GPTQModel/pull/6001) and [GPTQ #6012](https://github.com/ModelCloud/GPTQModel/pull/6012) | Use symmetric INT4 rows `[-a, a, 0, …]`. At `a = 61408` the negative lane keeps code 0 and replays as −65504. At the next FP16 value, 61440, code 0 would give −65536. Require code 1 and replay −57344. Drive each solver and inspect the real packer's lane. A finiteness check alone permits dropping the valid endpoint. | [RTN](../../../../tests/test_rtn_narrowed_endpoint.py) · [GPTQ](../../../../tests/test_gptq_narrowed_endpoint_6000.py) |
| [AWQ symmetric staging #6002](https://github.com/ModelCloud/GPTQModel/pull/6002) and [GGUF headers #5999](https://github.com/ModelCloud/GPTQModel/pull/5999) | For AWQ GEMM at 65504, require stored scale 9352 and signed code 7. Round their product in the consumer dtype. For GGUF Q4_0/Q8_0, choose a block whose scale magnitude rounds from 65520 to FP16 infinity and require an error. At 65504 inspect the valid two-byte scale header. Each format specifies its own outcome. | [AWQ staging to pack](../../../../tests/test_awq_symmetric_grid_finite.py) · [GGUF block bytes](../../../../tests/test_gguf_all_formats.py) |
| [QQQ activation #5998](https://github.com/ModelCloud/GPTQModel/pull/5998) and [GSQ channel scales #6009](https://github.com/ModelCloud/GPTQModel/pull/6009) | With `u = 2**-149`, QQQ maps zero to (code 0, scale 1) and `7*u` to (code 7, scale `u`). For a channelwise weight scale of `9 * 2**-24` stored in FP16, widening before division by 16 changes the result. Use nibble 1 to compare the GSQ candidate and actual packed decode against the original scale in FP64. | [Activation rows](../../../../tests/test_gsq_qqq_activation_contract.py) · [GSQ adapter and packer](../../../../tests/test_gsq_method_adapters.py) |

RTN and GPTQ still derive replay bounds separately. A weight fit on an FP32
copy can omit the source dtype's limit. The design proposal makes that
information mandatory at the weight-fit boundary. [Proposal #6020](https://github.com/ModelCloud/GPTQModel/issues/6020)

## A finite score can still select the wrong candidate

- An MSE shrink search at norm 16 kept candidate 0 when every powered residual
  underflowed on a row scaled by 0.001. The FP64 objective over the same FP32
  grid selects candidate 6. In MACaQ, rescale a 32-lane block with one unit
  peak and 31 BF16 values of 0.13 by `2**-80` and `2**80`. The winning offset
  stays 0, the stored exponent shifts by the applied power, and undoing the
  rescale recovers the same decoded weight. [MSE #5997](https://github.com/ModelCloud/GPTQModel/pull/5997) · [MSE tests](../../../../tests/test_find_params_mse_grid.py) · [MACaQ #6010](https://github.com/ModelCloud/GPTQModel/pull/6010) · [MACaQ tests](../../../../tests/test_macaq_exponent_screen_scale.py)
- For NVFP4 least-squares refinement, multiply the block **and its global
  scale** by `2**124`. The stored E4M3 local scale should stay 144. The
  seed-only recipe returns 208 at both magnitudes, making a skipped refinement
  observable even when all output values are finite. [NVFP4 #6008](https://github.com/ModelCloud/GPTQModel/pull/6008) · [NVFP4 tests](../../../../tests/test_nvfp4_ls_refine_scale_invariance.py)
- VPTQ's expanded distance rounded a positive distance gap to a tie. Put a
  vector of ones beside an exact centroid and a centroid with one lane one ULP
  higher. Try both orders and require the exact match and zero residual. RCO's
  reference-plus-delta mix lost the selected endpoint even with one-hot
  weights. Select each of −60000 and +60000 with unit input on an FP16 layer
  and require its exact output. [VPTQ #6018](https://github.com/ModelCloud/GPTQModel/pull/6018) · [VPTQ tests](../../../../tests/test_vptq.py) · [RCO #6005](https://github.com/ModelCloud/GPTQModel/pull/6005) · [RCO tests](../../../../tests/test_rco.py)

**Historical check.** On the [pre-fix revision](https://github.com/ModelCloud/GPTQModel/commit/d86a200e1907c360e0523febee0a0d345b458597),
six of eight CPU cases failed their numerical assertions. RTN and GPTQ
produced −inf at 61440, MACaQ missed the oracle offset at both extreme
magnitudes, and NVFP4 returned 208 instead of 144. The RTN 61408 and MACaQ
unit-scale controls passed. All eight passed on the audited revision.
