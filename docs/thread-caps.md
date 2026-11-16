<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Torch thread caps

Torch sizes its intra-op (OpenMP/MKL) pool from the CPU count the process can
see. On a container host that advertises hundreds of CPUs, one test or CLI
process can therefore create a thread per CPU and spend its time in the
OpenMP barrier instead of doing work. Measured on a 320-CPU container: a
single `pytest` process reported `torch.get_num_threads() == 205` and held
268 OS threads after one 512x512 matmul, while doing very little work.

Adding cores does not help: the pools grow to fill whatever the process is
allowed to see.

## What the repository does

The pytest harness caps the pools in `tests/conftest.py`. The environment
variables must be assigned **before** `numpy`/`torch` are imported because the
OpenMP/MKL/OpenBLAS runtimes read them when they load, so the cap block sits
above those imports:

```python
_DEFAULT_TEST_THREADS = min(8, os.cpu_count() or 1)
_TEST_THREADS = int(os.environ.get("OMP_NUM_THREADS", _DEFAULT_TEST_THREADS))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", ...):
    os.environ.setdefault(name, str(_TEST_THREADS))

import numpy
import torch

torch.set_num_threads(_TEST_THREADS)
```

`torch.set_num_threads` is called explicitly because the environment is not
authoritative for Torch's own pool once the runtime has loaded.
`torch.set_num_interop_threads` raises after parallel work has started, so it
is best-effort.

Library and CLI code can opt in through
`gptqmodel.utils.threads.cap_torch_threads()`:

```python
from gptqmodel.utils.threads import cap_torch_threads

applied = cap_torch_threads()          # min(8, cpu_count), or $OMP_NUM_THREADS
applied = cap_torch_threads(4, interop=2)
```

`import gptqmodel` does **not** call it. Silently changing a user's global
thread count on import would change numerics and performance without consent.
`optimize/` command-line tools and `scripts/dequantize_model.py` opt in at
their entry points.

## Precedence

1. An explicit argument to `cap_torch_threads(n)` wins.
2. Otherwise a positive integer `OMP_NUM_THREADS` wins.
3. Otherwise the capped default `min(8, os.cpu_count() or 1)` is used.

Every environment assignment uses `setdefault`, so an explicit
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` or `NUMEXPR_NUM_THREADS` is never overwritten.

## Opting out for a real performance measurement

The cap is a default, not a ceiling. A benchmark that deliberately controls
its pool keeps doing so: `tests/benchmark/*` call
`torch.set_num_threads(args.threads)` and are not changed by this harness.
For a timing run, set the pool explicitly and state it with the numbers:

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
  python tests/benchmark/benchmark_qqq_cpu.py --threads 32
```

`OMP_NUM_THREADS` also overrides the pytest default, for example
`OMP_NUM_THREADS=2 pytest tests/test_vptq.py`. Use a small value when several
jobs share the host; the default exists to keep one process from consuming
the whole machine.

## Verifying the cap

```bash
OMP_NUM_THREADS=2 PYTHONPATH=$PWD python -c \
  "import os, torch; print(torch.get_num_threads(), len(os.listdir('/proc/self/task')), os.cpu_count())"
OMP_NUM_THREADS=2 PYTHONPATH=$PWD python -m pytest tests/test_thread_caps.py -q
```

The intra-op count should be single-digit (or the explicit `OMP_NUM_THREADS`
value) and the process thread count should not grow with `os.cpu_count()`.
