# Phase 58: rejected H100 16-bit PGC multiply-add

Phase 58 tested whether the P32 pseudo-random generation checksum (PGC)
multiply-add should use a 16-bit PTX instruction after the preceding XORs
have reduced the state to its low 16 bits. The candidate was mathematically
exact but made every complete-MLP benchmark cell slower. All candidate CUDA
source was reverted; production remains byte-identical to Phase 56.

## Exact candidate math

The accepted decoder computes

\[
u=(state\oplus high\_byte\oplus bank\_mask)\bmod 2^{16}
\]

followed by

\[
p=(40503u+17011)\bmod 2^{16}.
\]

The production source expresses the multiply-add as `mad.lo.u32` and returns
the low 16 bits. The candidate first narrowed `u` to `uint16_t` and expressed
the same operation as `mad.lo.u16`. Because only the low 16 product bits feed
the following lookup, both expressions are identical for every input state.
No selector, level, accumulation, transform, or rounding order changed.

Twenty real Llama 3.2 1B grouped-Hopper shape cases passed exact output,
repeatability, dense-oracle tolerance, and CUDA Graph stability before the
performance sweep.

## Complete Llama 3.2 1B MLP result

Timing used the physical H100 only, 30 warmups, 200 CUDA-event samples, and
50 warmed CUDA Graph replays per sample. Marlin and Machete are figurative W4
baselines; ratios above one mean QVQ is faster. `Better` compares the
candidate with the accepted Phase-56 production matrix.

| W | MKN: gate/up x2; down | Candidate QVQ | vs Marlin W4 | vs Machete W4 | vs last | Better |
|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 48.673 us | 0.597x | 1.013x | 0.944x | No |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 48.997 us | 0.635x | 1.009x | 0.938x | No |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 49.710 us | 0.630x | 0.999x | 0.941x | No |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 50.148 us | 0.584x | 0.991x | 0.941x | No |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 51.383 us | 0.634x | 0.969x | 0.944x | No |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 55.728 us | 0.522x | 0.884x | 0.849x | No |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 55.852 us | 0.557x | 0.885x | 0.849x | No |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 56.470 us | 0.554x | 0.880x | 0.852x | No |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 57.119 us | 0.513x | 0.870x | 0.850x | No |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 58.193 us | 0.559x | 0.856x | 0.855x | No |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 49.272 us | 0.590x | 1.000x | 0.945x | No |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 49.436 us | 0.629x | 1.000x | 0.949x | No |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 49.948 us | 0.627x | 0.995x | 0.951x | No |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 50.513 us | 0.580x | 0.984x | 0.953x | No |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 51.555 us | 0.631x | 0.966x | 0.954x | No |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 49.667 us | 0.585x | 0.992x | 0.962x | No |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 49.890 us | 0.624x | 0.991x | 0.962x | No |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 50.375 us | 0.621x | 0.986x | 0.963x | No |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 50.794 us | 0.576x | 0.979x | 0.965x | No |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 51.915 us | 0.627x | 0.959x | 0.965x | No |

Geometric speed versus Phase 56 was **0.94151x** for W2, **0.85088x**
for W2.5, **0.95036x** for W3, and **0.96378x** for W3.5. The all-rate
result was **0.92553x with 0/20 wins**. The candidate's all-rate geometric
ratios were **0.59254x versus Marlin W4** and **0.95904x versus Machete W4**.

## Hopper machine-code analysis

`cuobjdump` was run on matched production and candidate extension binaries.
The production binary was
`5991c90fe1183941/gptqmodel_qvq_wgmma_ops.so`; the candidate was
`f2eecdfe9551859b/gptqmodel_qvq_wgmma_ops.so`. The W2.5 N128 kernel's
register count fell from 64 to 55, but that apparent resource improvement did
not represent cheaper execution.

Static opcode totals across the matched cubins changed as follows:

| Opcode family | Production | Candidate | Change |
|:--|--:|--:|--:|
| `IMAD` | 11,305 | 9,005 | -2,300 |
| `IMAD.SHL.U32` | 0 | 2,526 | +2,526 |
| `LOP3.LUT` | 7,105 | 11,837 | +4,732 |
| `PRMT` | 7,320 | 7,320 | unchanged |
| `LDS.U16` | 4,608 | 4,608 | unchanged |
| `LDS` | 4,160 | 4,160 | unchanged |

Hopper did not lower the PTX 16-bit multiply-add into a beneficial packed
integer operation. Instead, `ptxas` introduced substantially more masking,
packing, and shifted-multiply machinery. W2.5 suffered most because that
decoder executes the affected path at high frequency. The nine-register
reduction could not offset the enlarged dynamic instruction dependency path.

## Decision

- The candidate is rejected and no candidate CUDA/runtime source remains.
- Quantization bytes, persistent VRAM, graph topology, and mathematical output
  are unchanged.
- Phase-56 depth-three N128 remains the latest production implementation.
- Compilation was capped at four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.
- Future PGC work must be judged by generated Hopper machine code, not PTX
  operand width. A narrower C++/PTX type is not evidence of a narrower or
  cheaper SM90 instruction sequence.
