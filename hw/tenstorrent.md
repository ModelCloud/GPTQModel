# Tenstorrent Blackhole — architecture and measured behaviour

How a Blackhole p300c actually executes a kernel, what that implies for GPTQ
alignment, and the numbers measured on the qualified host. The canonical GPTQ
implementation reference is [Tenstorrent XLA GPTQ execution](../docs/tenstorrent.md);
this page is the hardware model behind it.

Facts marked **documented** come from Tenstorrent's own repository
(`tech_reports/`, `METALIUM_GUIDE.md`, `tt-isa-documentation`). Facts marked
**measured** were taken on `tt-quietbox` and are reproducible with the scripts
described at the end.

## 1. The Tensix core

A Blackhole chip is a 2D array of **Tensix** tiles plus memory, Ethernet and
control tiles. Each Tensix tile is not a GPU shader and not a systolic array; it
is a small self-contained computer (**documented**, `METALIUM_GUIDE.md`,
`tt-isa-documentation`):

| Component | Role |
| --- | --- |
| **BRISC** (RISCV_0) | data movement: reads tiles from DRAM or another core's L1 through the read NoC endpoint |
| **NCRISC** (RISCV_1) | data movement: writes finished tiles out through the write NoC endpoint |
| **TRISC0** | unpack: L1 → matrix engine input registers |
| **TRISC1** | math: drives the matrix engine |
| **TRISC2** | pack: matrix engine output registers → L1 |
| **Matrix engine** | 16x16-native multiply, plus reduce and elementwise |
| **L1** | ~1464 KiB (1.5 MB) of **explicitly managed SRAM**, per core |
| **NoC endpoints** | one read, one write, into the chip network |

Two properties matter more than the rest:

1. **L1 is not a cache.** It is working memory the program places data into. The
   compiler/runtime decides what lives there; nothing is brought in implicitly.
   On Blackhole L1 additionally gained a small (4 x 16 B lines) write-through
   data cache (**documented**, `BlackholeBringUpProgrammingGuide.md`).
2. **Movement is decoupled from compute.** The two data-movement RISCs run
   concurrently with the three compute RISCs, which is why a kernel is three
   co-operating programs (reader, compute, writer) rather than one loop.

The canonical flow is: NoC → L1 → unpack → matrix engine → pack → L1 → NoC
(**documented**, `METALIUM_GUIDE.md`).

## 2. Tiles and faces — the alignment that matters

Data reaching the matrix engine is not a flat array (**documented**,
`tech_reports/tensor_layouts/tensor_layouts.md`,
`tech_reports/matrix_engine/matrix_engine.md`):

- A **tile** is `32x32` elements (hardware also supports `16x32`, `4x32`,
  `2x32`, `1x32`; TT-Metal exposes `32x32` generally and `16x32` for a few ops).
- A tile is stored as **four faces of `16x16`**, laid out `face0→face1→face2→face3`,
  because **the matrix engine natively multiplies 16x16 matrices** and a tile
  multiply decomposes into face multiplies.
- Faces are not contiguous in the logical sense — a tile is a 2D page.

So the natural alignment unit is **32** for a tensor dimension, and **16** for
the engine's inner step.

### Matrix engine throughput

Per cycle the engine performs `8x16 × 16x16 = 8x16`, i.e. 4096 FLOPs, giving
(**documented**):

| Fidelity | TFLOPS per engine at 1 GHz |
| --- | --- |
| LoFi | 4 |
| HiFi2 | 2 |
| HiFi3 | 1.33 |
| HiFi4 | 1 |

Two consequences that shape kernel design:

- **Small `M` is wasteful.** If the engine is fed a `1x16` operand it still
  executes `8x16 × 16x16`, so effective throughput falls to **1/8**. Decode
  shapes (`M` = 1 token) are the worst case; `M` should be padded toward 8, and
  tile-aligned (`32`) for full efficiency.
- **Fidelity is a direct multiplier on FLOPs.** HiFi4 costs 4x LoFi. The native
  GPTQ kernel chooses HiFi4 only because measurement showed it removes a real
  accuracy floor; it is not free.

## 3. Native dtypes on the p300c

Only the FPU formats are multiply-accelerated. Measured on this card
(`ttnn.linear` on a `256x256` tile tensor, plus elementwise multiply):

| dtype | tile alloc | matrix engine (`ttnn.linear`) | elementwise mul |
| --- | --- | --- | --- |
| `bfloat16` | ok | **yes** (0.085 ms) | yes |
| `bfloat8_b` | ok | **yes** (0.111 ms) | yes |
| `bfloat4_b` | ok | **yes** (0.095 ms) | yes |
| `float32` | ok | **yes** (0.086 ms) | yes |
| `int32` | ok | **no** | yes |
| `uint32` | ok | **no** | yes |
| `uint16` | ok | **no** | yes |
| `uint8` | ok | **no** | **no** |

What that means, and where the trap is:

- **The FPU computes on `bfloat16` (Float16_b) or `float32`** (**documented**,
  `tech_reports/matrix_engine`). `float32` accumulation is opt-in via
  `fp32_dest_acc_en` and halves destination-register capacity, so it costs tile
  throughput.
- **`bfloat8_b` / `bfloat4_b` are storage formats, not compute formats.** The
  matrix engine does not multiply them natively; they are unpacked to BF16 by
  TRISC0 before math (**documented**, `tech_reports/data_formats`). `ttnn.linear`
  accepting them means the conversion happens for you, not that the multiply is
  free. They halve/quarter memory traffic and L1 footprint at a precision cost.
- **Integer dtypes cannot use the matrix engine at all.** `int32`, `uint32`,
  `uint16` and `uint8` allocate as tiles and support elementwise work, but
  `ttnn.linear` fails outright. Integer work runs on the **SFPU**, which is a
  vector unit, not the wide multiply array.
- **`uint8` is the weakest**: even elementwise multiply fails.

This is the mechanism behind the GPTQ dequant cost measured in section 7. The
packed weight words are `uint32`, and extracting nibbles
(`logical_right_shift` + `bitwise_and`, both measured `ok`) is inherently SFPU
work. So the dequant is not merely "many ops" -- it is many **integer** ops on
a unit that is not the fast path, followed by a BF16 GEMM that is dwarfed by
them. Doing that expansion once per load instead of once per forward removes
~85 % of a layer's cost; doing it in `uint32` on every token is the worst case.

Rule for TT kernels here: **keep hot compute in `bfloat16`.** Use block floats
only to cut memory traffic, use integer tiles only for bit manipulation, and
never put an integer dtype on the multiply path.

## 4. Memory hierarchy

| Level | Size | Notes |
| --- | --- | --- |
| L1 per Tensix | ~1464 KiB | explicitly managed; holds kernels, circular buffers, tiles |
| DRAM | **8 banks x ~4 GB** (Blackhole) | one bank per DRAM channel |
| DRAM bandwidth | 512 GB/s class | GDDR6, 256-bit bus (**documented**; vendor + independent measurement) |

Allocation rules that leak into correctness and performance (**documented**,
`tech_reports/memory/allocator.md`):

- Buffers are **lockstep-allocated** across banks: every bank reserves the same
  space for a buffer even if only some banks hold its data. Address arithmetic
  stays trivial; the cost is internal fragmentation.
- Allocations are **`DRAM_ALIGNMENT`-aligned** so DRAM→L1 reads do not violate
  NoC alignment. A page or shard size that ignores it is padded.
- **Interleaved** layout round-robins pages across banks; **sharded** layout
  places chosen shards in chosen banks. Sharding is how a tensor is kept in
  L1 across the core grid instead of streaming from DRAM.

## 5. NoC (network-on-chip)

(**documented**, `BlackholeBringUpProgrammingGuide.md`, `tt-isa-documentation`)

| Property | Wormhole | Blackhole |
| --- | --- | --- |
| NoC clock | 1 GHz | **1.35 GHz** |
| Flit width | 256 bit | **512 bit** |
| Address width | 36 bit | 64 bit |
| DRAM align (read / write) | 32 B / 16 B | **64 B / 16 B** |
| PCIe align (read / write) | 32 B / 16 B | **64 B / 16 B** |
| L1 align (read / write) | 16 B / 16 B | 16 B / 16 B |
| Multicast | rectangular | rectangular, **strided, L-shaped** |

The **64 B read alignment** is the one that bites host-facing copies: a transfer
that is not a multiple of 64 B pays extra transactions. Multicast is what lets a
weight tile reach many cores without re-reading DRAM per core, and strided /
L-shaped patterns exist to match how a tensor is actually sharded.

## 6. Chip topology and mesh

Tensix, Ethernet, DRAM and PCIe tiles are arranged on a 2D grid with a torus-like
NoC. Tiles are addressable by coordinate, so "put this shard on that core" is a
placement decision, not a DMA descriptor (**documented**).

Multiple chips form a **mesh**; `TT-Fabric` programs the inter-chip links and
collectives (`tech_reports/TT-Fabric/`). For GPTQ this is what makes tensor
parallelism possible: `all_gather` across a `1x4` mesh.

## 7. Measured on this host

`tt-quietbox`: 4x p300c, firmware 19.17.1, KMD 2.12.0, TT-Metal plugin
`1.5.0.dev20260831`.

### Silicon exposed

```text
compute grid            11x10 = 110 Tensix cores
AICLK idle / busy       800 MHz / 1350 MHz
GDDR throttle limit     90 C        (THM_LIMIT_THROTTLE)
GDDR shutdown limit     110 C
board TDP limit         125 W       (also the maximum; larger requests clamp)
```

The p300c exposes **110** compute cores, fewer than the 13x10 in the architecture
table because defective columns are harvested out. Any kernel sizing must read
the grid at runtime rather than assume 130 or 140.

### PCIe cost per crossing (1 device, BF16, tile layout)

| Payload | host→device | device→host | round trip |
| --- | --- | --- | --- |
| 12 KiB (`8x768`) | 0.03 ms | 0.09 ms | **0.12 ms** |
| 192 KiB (`128x768`) | 0.04 ms | 0.09 ms | 0.13 ms |
| 1.5 MiB (`1024x768`) | 0.18 ms | 0.65 ms | 0.84 ms |

Crossings are **cheap at decode sizes**. This contradicts the intuition that a
per-layer round trip is what makes small-batch inference slow, and it changed
the design conclusion below.

### Where a GPTQ layer's time actually goes (batch 8, `768x768`, HiFi4)

| Phase | Time | Share |
| --- | --- | --- |
| **dequant (nibble planes → BF16)** | **1.27 ms** | **85 %** |
| `ttnn.linear` (the actual GEMM) | 0.06 ms | 4 % |
| host→device upload | 0.11 ms | 7 % |
| device→host readback | 0.05 ms | 3 % |

**The dequantization is ~20x the cost of the matrix multiply it feeds.** Every
forward re-expands the packed nibbles into a dense BF16 tile, and that
expansion — eight nibble planes, a stack, a permute, two `embedding` gathers, a
subtract and a multiply — is the real bottleneck, not PCIe and not the GEMM.

Consequence for the residency work: keeping activations on device removes the
0.16 ms of transfers, i.e. about **1.13x** on a warmed 4-layer chain (5.93 ms →
5.26 ms, identical output). An earlier 8.6x figure was an artefact of timing
first calls, which include a one-off weight upload, and should be disregarded.

## 8. What this implies for GPTQ alignment

Ordered by the measured payoff:

1. **Stop dequantizing per forward.** Materialize the BF16 weight once (at load,
   on device) and run the GEMM against it. For OPT-125M the expanded weights are
   ~1.2 MB per `768x768` layer, ~85 MB for the model — trivial against ~32 GB of
   DRAM, and it removes ~85 % of the per-layer cost. This trades the packed-only
   invariant for speed, so it must be opt-in and documented as such.
2. **Pad `M` to the engine's granularity.** `M` below 8 wastes up to 8x of the
   matrix engine (`1/8` throughput at `1x16`); pad to 8 as a floor and 32 for
   full tiles. This is the single biggest lever for decode.
3. **Keep `K` and `N` tile-aligned.** GPTQ `group_size` 128 is 4 tiles (8
   faces) and aligns naturally; sizes that are not multiples of 32 force padding
   and extra NoC traffic.
4. **Choose layout deliberately.** Weights in L1-sharded layout, activations
   interleaved, and multicast for weight broadcast. This is where DRAM
   contention is avoided.
5. **Pick fidelity by measurement, not default.** HiFi4 costs 4x LoFi; the
   native kernel pays it to remove a measured accuracy floor on real data.
6. **Batch beats round trips.** Since a crossing is 0.12 ms and a dequant is
   1.27 ms, reducing round trips buys far less than removing the dequant.
7. **Keep the hot path in `bfloat16`.** Integer tiles have no matrix engine, so
   an integer op between two GEMMs costs a format change and runs on the SFPU.
   Section 3 is the support matrix; do not put `int32`/`uint32` on the compute
   path, only on the bit-manipulation path.

## 9. Qualified host

| Item | Validated value |
| --- | --- |
| Host | `tt-quietbox`, ASRock B850M-C |
| OS / kernel | Ubuntu 26.04.1, `7.0.0-43-generic` |
| Accelerators | 4x Blackhole p300c (`1e52:b140`), BDF `01:00.0` through `04:00.0` |
| PCIe | Gen4 x4 per card; BIOS bifurcation is **x4x4x4x4** |
| Firmware | `19.17.1.0`; cm `0.39.1.0`, eth `1.13.1`, dm_app `0.33.1.0`, gddr `2.17` |
| Kernel module | `tenstorrent-dkms 2.12.0` |
| System tools | `tenstorrent-tools 1.4.1`, `sfpi 7.69.0` |
| Python | CPython 3.12.16 from pyenv, GIL enabled |
| Runtime | CPU Torch `2.11.0+cpu`, Torch/XLA `2.9.0+gited8a445`, p300c PJRT plugin |
| Management | Tailscale running; Tailscale SSH disabled; OpenSSH active |

All four cards report operating DRAM, passing GDDR training and BIST, and zero
corrected or uncorrected GDDR errors. `tt-smi 6.8.1` enumerates four p300c
boards once PCIe bifurcation is set. No firmware update was required.

## 10. Runtime identity

```text
torch                    2.11.0+cpu
torch-xla                2.9.0+gited8a445
pjrt-plugin-tt           1.5.0.dev20260831000501
PJRT device type         TT
addressable devices      4
plugin commit            873c53c4ff84bd91112a1f43e788cfed75aaa914
tt-mlir commit           e2b21a721955b2b68540d9e6879a215ce3dcfae5
tt-metal commit          d5f9bc4a43b1287523458f1af6a470c6b2adc180
```

The plugin wheel bundles TTNN and TT-Metal under `pjrt_plugin_tt`; do not put a
separate TT-Metal checkout on `PYTHONPATH`. The bundled TT-Metal requires the
matching SFPI version (`7.69.0` for this build). This plugin is the newest
published release; `tenstorrent/tt-xla` releases stop at 2026-08-31 and its
workflows currently fail with `startup_failure`.

Note the version skew that explains several surprises: the local tt-metal source
tree is `v0.83.0-dev20261108`, while the installed plugin bundles tt-metal from
**2026-08-20**. Features present in the source tree (for example
`TT_METAL_TDP_LIMIT_WATTS`) do not exist in the installed binaries.

## 11. Verification

The opt-in hardware test executes the real `XLALinear` on each card and compares
it with an independent CPU unpack/dequantize oracle:

```bash
GPTQMODEL_TEST_TT_XLA=1 python -m pytest -q -s \
  tests/test_tenstorrent_support.py::test_actual_ttxla_forward_on_all_four_cards
```

Validated result: `1 passed`. Device dequantized weights are bit-identical to the
CPU oracle; across `xla:0`–`xla:3` PCC is ~`0.999999`, normalized error ~`3e-4`,
and material outputs are within one BF16 ULP. The persistent layout stays packed;
expanded BF16 codes exist only while a forward executes.

The focused CPU and contract suite reports `402 passed, 1 skipped` before the
opt-in hardware gate is enabled.

### Reproducing the measurements on this page

```bash
/home/ttuser/tt-reboot/hw_probe.py           # grid size and PCIe crossing cost
/home/ttuser/tt-reboot/phase_breakdown.py    # per-phase layer breakdown
/home/ttuser/tt-reboot/resident_fair.py      # warmed host vs resident chain
```

Run them under the thermal watchdog, which kills at 88 °C:
`TTWATCH_MAX_C=88 python /home/ttuser/tt-reboot/ttwatch.py -- python <script>`.

## 12. Firmware tooling

`tt-flash` writes a board firmware bundle or SPI image. It belongs in the
maintenance toolchain, not the inference loop. The p300c bundle format is
`p300c.fwbundle`; changing firmware requires an explicit maintenance decision, a
verified power state, and a recovery plan.

## Sources

Documented architecture:

- `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md` — grid, L1, NoC
  clocks/flits, alignments, multicast patterns
- `tech_reports/matrix_engine/matrix_engine.md` — 8x16x16 per cycle, fidelity
  scaling
- `tech_reports/tensor_layouts/tensor_layouts.md` — tile/face layout, interleaved
  vs sharded
- `tech_reports/memory/allocator.md` — lockstep allocation, alignment
- `METALIUM_GUIDE.md` — Tensix anatomy and data flow
- `tenstorrent/tt-isa-documentation` — per-core RISC-V roles, NoC upgrade details

External:

- [TT-Metalium for Beginners](https://github.com/tenstorrent/tt-tutorial/blob/main/tt-metalium/TT-Metalium_For_Beginners.md)
- [TT-Metalium: Tensix Core to Multi-Chip Scale-Out (HPC Asia '25)](https://riscv.epcc.ed.ac.uk/assets/files/hpcasia25/Tenstorrent.pdf)
- [Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking (ASPLOS)](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [NoC documentation, tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/README.md)
