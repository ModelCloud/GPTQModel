# SLQ: statistical fidelity and nonuniform allocation

## Sources and finding

Helcig, Kurtic and Alistarh,
[Statistically-Lossless Quantization of Large Language Models, v2](https://arxiv.org/abs/2605.02404v2).

The paper distinguishes task-level preservation within sampling variation from
preservation of the next-token distribution. Expected Acceptance Rate (EAR)
measures maximal agreement under optimal coupling; it is not greedy top-1
agreement or exact generated-trajectory survival. The paper also studies
quantization-noise variance and nonuniform bit allocation.

“Statistically lossless” does not mean bitwise lossless compression or identical
weights. Paper thresholds and empirical bit rates require their stated
assumptions and evaluation protocols.

## Repository evidence

[gptqmodel/quantization/slq/__init__.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/slq/__init__.py) exposes metrics, variance-law
helpers, allocation and calibrators as composable utilities.
[gptqmodel/quantization/slq/config.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/slq/config.py) builds dynamic-bit settings.
[gptqmodel/quantization/sensitivity/profiler.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/sensitivity/profiler.py) separately profiles
candidate rates. This is a utility layer, not a METHOD.SLQ enum or automatic
certification of every resulting model.

## Recovery implications

Keep EAR, KL, top-k agreement and task scores separate. Do not transfer an
asymmetric scalar-quantization variance conclusion unchanged to a trellis or
NVFP4 codebook without checking the assumptions.

Allocate bits or correction rank against a declared storage budget, then
evaluate the installed model on disjoint inputs. Count scales and correction
payloads in that budget. A successful allocation solve does not prove the
target fidelity threshold after serialization or full-model propagation.
