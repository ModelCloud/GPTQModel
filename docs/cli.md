# The `gptqmodel` command line

```console
$ gptqmodel doctor
area       check                      status  detail
---------  -------------------------  ------  -----------------------------------------------------------
runtime    python                     OK      CPython 3.12.14
runtime    platform                   INFO    Linux x86_64 (WSL)
torch      version                    OK      2.14.1+cu132
torch      cuda build                 OK      CUDA 13.2
torch      safety advisories          INFO    GHSA-v32g-mr63-3j83 (checks bypassed)
gpu        cuda:0                     OK      NVIDIA GeForce RTX 5090 Laptop GPU sm_120, 22.6/23.9 GiB free
deps       transformers               OK      5.24.0
deps       tokenicer                  OK      0.0.21
deps       triton                     OK      3.8.0
env        PYTORCH_ALLOC_CONF         INFO    expandable_segments:True

result: ok
```

`python -m gptqmodel` runs the same entry point without installing the console
script.

## `gptqmodel doctor`

Answers the questions a support thread otherwise asks for one at a time:

| area | what it covers |
| --- | --- |
| `runtime` | Python version, implementation, platform, WSL |
| `torch` | torch build, CUDA/ROCm version, safety advisories and whether checks are bypassed |
| `gpu` |每 device name, compute capability (`sm_120`), free/total VRAM |
| `deps` | required packages (a missing one is `FAIL`), optional backends (missing is `INFO`), and `tokenicer.ernie_vl` - transformers 5.24 needs it and only fails at the first model load |
| `extension` | compiled kernels shipped next to the package |
| `env` | flags that change behaviour: allocator, device order, offline mode, safety bypass |

`--model PATH` adds the checkpoint's config - architecture, quantization summary,
and the KV-cache estimate from `gptqmodel.utils.kv_cache` - without loading any
weights.

Exit codes: `0` when nothing failed, `1` on `FAIL` (`--fail-on warn` also exits
`1` on `WARN`). `--json` prints JSON (the startup banner may precede it), while
`--json-out FILE` writes a clean file and keeps the table on stdout.

## `gptqmodel measure` (after #5967 lands)

It will delegate to `python -m gptqmodel.measure`, so the measurement arguments
and their validation stay authoritative in one place:

```console
gptqmodel measure --model <quant> --reference <fp16> --text docs.txt --kld-pass "<=0.05"
```

## Adding a subcommand

Each subcommand lives in its own module under `gptqmodel/cli/`; `main.py` only
parses and dispatches, so `inspect` and `quantize` can be added without touching
the existing ones.
