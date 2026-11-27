# The `gptqmodel` command line

```console
$ gptqmodel doctor
area       check                      status  detail
---------  -------------------------  ------  -----------------------------------------------------------
runtime    python                     OK      CPython 3.13.3 (no GIL)
runtime    platform                   INFO    Linux x86_64 (WSL)
torch      version                    OK      2.15.0+cu132
torch      cuda build                 OK      CUDA 13.2
torch      safety advisories          OK      none
gpu        cuda:0                     OK      NVIDIA GeForce RTX 5090 Laptop GPU sm_120, 22.6/23.9 GiB free
deps       transformers               OK      5.23.0
deps       tokenicer                  OK      0.0.21
deps       torch                      OK      2.15.0+cu132
extension  native kernels             OK      1 JIT artifact(s) in the torch extension cache (allspark_spike)
env        PYTORCH_ALLOC_CONF         INFO    expandable_segments:True

result: ok
```

`python -m gptqmodel` runs the same entry point without installing the console
script.

## `gptqmodel doctor`

Answers the questions a support thread otherwise asks for one at a time:

| area | what it covers |
| --- | --- |
| `runtime` | Python version, implementation, platform, WSL, whether the GIL is disabled |
| `torch` | torch build, CUDA/ROCm version, safety advisories and whether checks are bypassed |
| `gpu` | per-device name, architecture (`sm_120` on CUDA, `gfx1201` on ROCm, labelled `hip:N`), free/total VRAM |
| `deps` | required packages (a missing one is `FAIL`), optional backends (missing is `INFO`), and supported version ranges |
| `extension` | compiled kernels: ones shipped next to the package, installed under `gptqmodel_ext`, JIT-built into the torch extension cache, or present in the NF12 source-keyed cache |
| `env` | flags that change behaviour: allocator, device order, offline mode, safety bypass |

Supported version ranges come from `requirements.txt` in a source checkout and
from the installed package metadata in a wheel. An installed version outside its
range is a `WARN` (the tool still runs; the combination is untested), while a
missing required package is a `FAIL`. Versions are read from package metadata,
so `doctor` never imports an optional backend just to print its version.

### `--model PATH`

Adds the checkpoint's config - architecture, quantization summary, and the
KV-cache estimate from `gptqmodel.utils.kv_cache` - without loading any weights.

Checkpoint Python is **not** executed by default: a config with an `auto_map`
entry fails with a hint instead of importing its module. To inspect such a
checkpoint, review its code and pass both consents:

```console
$ GPTQ_BYPASS_SAFETY_CHECKS=1 gptqmodel doctor --model ./custom-checkpoint --trust-remote-code
```

Without the environment variable the flag is refused and the config is not
imported.

### Machine-readable output

`--json-out FILE` writes the same rows as JSON to `FILE` and keeps the table on
stdout:

```console
$ gptqmodel doctor --json-out doctor.json
$ jq '.failing, .warning' doctor.json
```

There is no `--json` stdout mode: the package prints its banner and environment
logs to stdout at import time, before any CLI code runs, so stdout is not a
reliable machine-readable channel. Use the file.

Exit codes: `0` when nothing failed, `1` on `FAIL` (`--fail-on warn` also exits
`1` on `WARN`).

## Adding a subcommand

Each subcommand lives in its own module under `gptqmodel/cli/`; `main.py` only
parses and dispatches, so `inspect` and `quantize` can be added without touching
the existing ones.
