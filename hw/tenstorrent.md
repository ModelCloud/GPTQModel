# Tenstorrent Blackhole and Torch/XLA

This page records the validated GPT-QModel Tenstorrent path and the hardware
environment used to qualify it. The short version is: a standard CPU PyTorch
build talks through Torch/XLA and Tenstorrent's PJRT plugin to the bundled
TTNN/TT-Metal runtime. GPT-QModel adds an explicit `DEVICE.XLA` family and a
packed 4-bit `XLALinear`; it does not use a Tenstorrent-specific PyTorch
build.

The canonical implementation reference is
[Tenstorrent XLA GPTQ execution](../docs/tenstorrent.md). This hardware page keeps
the host inventory and release-level measurements that are useful when
reproducing the result on another QuietBox.

## Qualified host

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
corrected or uncorrected GDDR errors in the qualification run. `tt-smi 6.8.1`
enumerated four p300c boards after PCIe bifurcation was corrected. A future
firmware update was not required for this software validation and is not part
of the installation script.

## Runtime identity

The PJRT plugin reports:

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

The plugin wheel bundles TTNN and TT-Metal under `pjrt_plugin_tt`; do not
install or put a separate TT-Metal checkout on `PYTHONPATH`. The bundled
TT-Metal requires the matching SFPI version. For this build that is `7.69.0`.

## Verification

The opt-in hardware test executes the real `XLALinear` on each card and
compares it with an independent CPU unpack/dequantize oracle:

```bash
GPTQMODEL_TEST_TT_XLA=1 \
python -m pytest -q -s \
  tests/test_tenstorrent_support.py::test_actual_ttxla_forward_on_all_four_cards
```

Validated result: `1 passed`. Device dequantized weights are bit-identical to
the CPU oracle. Across `xla:0` through `xla:3`, measured PCC is approximately
`0.999999`, normalized error is approximately `3e-4`, and material outputs are
within at most one BF16 ULP. The persistent layout remains packed and on the
target card; the expanded BF16 codes exist only while a forward is executing.

The focused CPU and contract suite currently reports `402 passed, 1 skipped`
before the opt-in hardware gate is enabled. The skip is the hardware test, not
an unqualified fallback.

## Firmware tooling

`tt-flash` writes a board firmware bundle or SPI image. It belongs in the
maintenance toolchain, not the inference loop. The p300c bundle format is
`p300c.fwbundle`; changing firmware requires an explicit maintenance decision,
verified power state, and recovery plan. It is unrelated to installing or
upgrading `XLALinear`.

Package pins and the exact installation procedure live in the
[canonical runtime guide](../docs/tenstorrent.md#install-and-versions).
