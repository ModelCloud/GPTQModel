# A100 / GA100 Hardware-Software Discovery Notes

Date: 2026-04-13

This note captures the hardware and software facts gathered before continuing SM80 kernel tuning for MentaRay. It separates:

- Official NVIDIA-public A100 / GA100 facts
- Local host-observed facts for GPU `0`
- Inferences that appear likely but are not backed by a public NVIDIA spec for this exact OEM board

## Executive Summary

- NVIDIA's public `A100` documentation describes a `GA100`-based shipping A100 with `108 SMs`.
- NVIDIA's public `GA100 full-die` documentation describes `128 SMs`.
- The local board on this host, GPU `0`, reports as an Ampere `sm80` board with `98304 MiB` memory and `124 SMs`.
- Therefore, the local board is materially different from NVIDIA's standard public 40GB/80GB A100 documentation and should be treated as a near-full-die `GA100`-class `sm80` target.

Inference:
The local `96 GB / 124 SM` board appears to be an OEM / non-standard A100-family implementation that is closer to full GA100 than to the public 108-SM A100. I did not find a public NVIDIA spec sheet for this exact SKU.

## Official NVIDIA-Public Architecture Facts

Primary source:
- NVIDIA A100 / GA100 architecture whitepaper: <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf>

Public GA100 full-die organization from the whitepaper:
- `8 GPCs`
- `8 TPCs / GPC`
- `2 SMs / TPC`
- `128 SMs` for the full GA100 die
- `64 FP32 CUDA cores / SM`
- `4` third-generation Tensor Cores / SM
- `6` HBM2 stacks and `12 x 512-bit` memory controllers

Public A100 implementation from the same whitepaper:
- `108 SMs`
- `432` third-generation Tensor Cores
- `5` HBM2 stacks and `10 x 512-bit` memory controllers

Relevant microarchitectural facts for kernel work on `sm80`:
- Ampere A100 supports FP16 and BF16 tensor-core math at the same advertised dense Tensor Core rate.
- Each SM has `192 KB` combined L1 data cache / shared memory.
- Ampere raises configurable shared memory up to `164 KB` per SM.
- Ampere adds asynchronous global-to-shared copy (`cp.async`-style pipeline support) and asynchronous barriers.
- Public A100 adds a `40 MB` L2 cache with residency controls and improved bandwidth over V100.

Why this matters for MentaRay:
- This is an `sm80` kernel problem, not a Hopper `wgmma` / `tma` problem.
- Tensor-core kernels should be designed around `mma.sync`, `ldmatrix`, async copy staging, and aggressive shared-memory pipelining.
- Launch heuristics should be tuned to the actual observed SM count, not blindly to public 108-SM A100 assumptions.

## Public Chip Diagrams

Official diagrams are in the whitepaper above:

- Full-chip diagram: `Figure 6`, "GA100 Full GPU with 128 SMs (A100 Tensor Core GPU has 108 SMs)"
- SM diagram: `Figure 7`, "GA100 Streaming Multiprocessor (SM)"

These are the reference diagrams to use for kernel planning.

## Local Host-Observed Hardware Facts

All commands below were run with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

### GPU 0 inventory

Command:

```bash
nvidia-smi -i 0 -q
```

Observed key fields for GPU `0`:
- Product name: `NVIDIA PG506-230`
- Architecture: `Ampere`
- PCI Bus ID: `00000000:22:00.0`
- Memory total: `98304 MiB`
- MIG mode: `Disabled`
- VBIOS: `92.00.4F.00.01`
- Board part number: `699-2G506-0230-500`
- Device ID: `0x20B610DE`

### GPU 0-3 SM count

Command:

```bash
python - <<'PY'
import torch
for i in range(4):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.multi_processor_count, p.total_memory, (p.major, p.minor))
PY
```

Observed:
- GPU `0`: `124 SMs`, `95.15 GiB`, `sm80`
- GPU `1`: `124 SMs`, `95.15 GiB`, `sm80`
- GPU `2`: `124 SMs`, `95.15 GiB`, `sm80`
- GPU `3`: `124 SMs`, `95.15 GiB`, `sm80`

### PCI-order GPU names

Command:

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total,driver_version,vbios_version,compute_cap --format=csv,noheader
```

Observed for GPUs `0-3`:
- `0, 00000000:22:00.0, NVIDIA PG506-230, 98304 MiB, 595.58.03, 92.00.4F.00.01, 8.0`
- `1, 00000000:28:00.0, NVIDIA PG506-232, 98304 MiB, 595.58.03, 92.00.53.00.02, 8.0`
- `2, 00000000:5C:00.0, NVIDIA PG506-230, 98304 MiB, 595.58.03, 92.00.4F.00.01, 8.0`
- `3, 00000000:61:00.0, NVIDIA PG506-230, 98304 MiB, 595.58.03, 92.00.4F.00.01, 8.0`

## Local Host-Observed Software Facts

Command:

```bash
python - <<'PY'
import torch, triton
import gptqmodel
print(torch.__version__)
print(torch.version.cuda)
print(triton.__version__)
print(getattr(gptqmodel, "__version__", "unknown"))
PY
```

Observed:
- Driver: `595.58.03`
- CUDA driver/runtime reported by `nvidia-smi`: `13.2`
- PyTorch: `2.11.0+cu130`
- Torch CUDA toolkit version: `13.0`
- Triton: `3.6.0+git9844da95`
- GPT-QModel: `6.1.0-dev`
- Transformers (during import banner): `5.6.0.dev0`

Operational notes from the local environment:
- GPT-QModel auto-sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, which is the correct mode for deterministic GPU targeting.
- MIG is disabled on GPU `0`, so full-board resources are available to a single process.

## Public Documentation on PG506 Family

Useful NVIDIA reference:
- NVIDIA data center driver release notes (example branch documenting `PG506` HGX A100 family VBIOS entries): <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-517-88/index.html>

What that confirms:
- `PG506` is used in NVIDIA HGX A100 family documentation.

What it does not confirm:
- It does not provide a public `96 GB / 124 SM` spec matching the local `PG506-230` / `PG506-232` boards observed on this host.

## What Is Known vs Unknown

Known from public NVIDIA documents:
- Full `GA100` is `128 SM`.
- Public `A100` is `108 SM`.
- A100 / GA100 `sm80` supports the Tensor Core, shared-memory, async copy, and L2 behaviors that matter for Marlin-style kernels.

Known from this host:
- GPU `0` is an Ampere `sm80` board with `124 SM` and `96 GB` memory.

Unknown from public docs:
- Exact official SKU naming and official memory-controller / bandwidth spec for this `96 GB`, `124 SM`, `PG506-230/232` board.

## Tuning Implications For MentaRay

The kernel should be tuned as an `sm80 near-full-GA100` target:

- Prefer `mma.sync` / `ldmatrix` / async-copy design patterns.
- Exploit Ampere shared-memory capacity and multistage pipelines.
- Treat `124` observed SMs as the occupancy target for launch policy on this host.
- Re-check any heuristic inherited from Marlin that implicitly assumes standard public `108 SM` A100 behavior.
- Avoid importing Hopper-only ideas that depend on architecture features missing on `sm80`.

## Sources

- NVIDIA A100 / GA100 architecture whitepaper: <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf>
- NVIDIA A100 product page: <https://www.nvidia.com/en-us/data-center/a100/>
- NVIDIA A100 80GB public datasheet: <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf>
- NVIDIA data center driver release notes with `PG506` family references: <https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-517-88/index.html>
