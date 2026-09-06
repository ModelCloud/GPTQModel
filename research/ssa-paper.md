# Cytron et al.: static single assignment and control dependence

## Primary reference

Ron Cytron, Jeanne Ferrante, Barry K. Rosen, Mark N. Wegman and F. Kenneth Zadeck,
[Efficiently Computing Static Single Assignment Form and the Control Dependence
Graph](https://doi.org/10.1145/115372.115320), TOPLAS 13(4), 1991, pp. 451–490.
[IBM publication record](https://research.ibm.com/publications/efficiently-computing-static-single-assignment-form-and-the-control-dependence-graph).

## Finding and scope

This foundational work concerns efficient construction of program
representations for dataflow and control-dependence analysis. SSA makes value
definitions explicit, helping later optimization reason about uses and merges.
It is infrastructure for optimization, not a GPU-specific algebraic optimizer.

## QVQ implication (proposed)

Use [SSA/IR inspection](ssa-sass.md) when asking whether repeated expressions
can be merged or constants propagated. SASS inspection answers what survives
lowering. Neither representation alone proves that floating-point algebra
permits a rewrite or that the result is faster.

For activation-scale folding, state numeric and control-flow preconditions
before changing the graph. Preserve clipping, rounding, side effects and alias
dependencies. This note adds background literature, not an SSA pass to QVQ.
