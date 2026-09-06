# GAR: group-aware reordering

## Sources

[Dual Precision Quantization for Efficient and Accurate Deep Neural Networks
Inference](https://arxiv.org/abs/2505.14638), cited by the repository's
[GAR implementation](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/gar.py).

## Repository finding

The implementation constructs local group permutations, a global permutation,
their composition, tail extension and inverse. It is a grouping/reordering
enhancement, not a separate METHOD enum or an additive correction branch.
The source's citation does not establish that the complete dual-precision
paper pipeline is implemented by this helper.

## Recovery implications

A permutation is reversible before quantization if all dependent tensors and
index maps are updated consistently. Changing group membership changes which
weights share scales and therefore changes the quantization problem.

Validate inverse mappings, incomplete groups, ordering stability and packed
group indices. Recompute scales and recovery factors as required by the new
operator; do not attach factors fitted against an old ordering.

Measure the improvement against a matched no-GAR quantization and include
any runtime gather/reordering cost. An exact permutation round-trip is a
correctness property, not evidence of post-quantization quality improvement.
