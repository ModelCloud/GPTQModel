---
name: gpu-allocator-cli
description: Request and release NVIDIA GPUs through the standalone GPU allocator using its CLI or Python client. Use whenever the task needs one or more GPUs, must avoid GPU contention with other sessions, or should run a command under an explicit lease.
---

# GPU allocator CLI workflow

The `gpu_allocator/` program is a standalone, thread-safe, free-threading-compatible GPU lease manager. Always acquire GPUs right before CUDA work and release them immediately after.

## Start the server (one per host)

The server must already be running. If it is not, start it:

```bash
python -m gpu_allocator.server --host 0.0.0.0 --port 17351
```

The default URL for all client commands is `http://127.0.0.1:17351`. Override with `--base-url` or by setting `GPU_ALLOCATOR_URL`.

## Request GPUs from the CLI

The safest style is `uuid` because it is unambiguous regardless of how CUDA orders devices:

```bash
python -m gpu_allocator.cli --base-url http://127.0.0.1:17351 acquire -n 2 --style uuid --format shell
```

Other valid styles:

- `pci_bus_id` — full PCI address (e.g. `00000000:25:00.0`).
- `pci_order_index` — requires `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the consumer.

Output formats:

- `json` (default) — full lease record including `lease_id` and GPU metadata.
- `shell` — `export` statements for `CUDA_VISIBLE_DEVICES`, `GPU_ALLOCATOR_LEASE_ID`, and `GPU_ALLOCATOR_GPU_STYLE`.
- `ids` — only the comma-separated GPU identifiers.

## Run a command under a temporary lease

Use `run` so the allocator automatically releases the GPUs when the command exits:

```bash
python -m gpu_allocator.cli --base-url http://127.0.0.1:17351 run -n 2 --style uuid -- python bench.py
```

This sets `CUDA_VISIBLE_DEVICES`, `GPU_ALLOCATOR_LEASE_ID`, and `CUDA_DEVICE_ORDER` (when needed) in the child environment.

## Release a lease explicitly

If you used `acquire` instead of `run`, release the lease by id:

```bash
python -m gpu_allocator.cli --base-url http://127.0.0.1:17351 release --lease-id <lease_id>
```

## Request modes

- `acquire` defaults to `timeout=0`: returns immediately if no GPUs are free.
- `-t 60` blocks up to 60 seconds.
- Omit `-t` or pass a negative value to block indefinitely.
- Default is `--exclusive`. Use `--shared` for work that can overlap with other shared leases on the same GPUs.

## Python API alternative

```python
from gpu_allocator import acquire

with acquire(2, base_url="http://127.0.0.1:17351", exclusive=True) as lease:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = lease.as_cuda_visible_devices("uuid")
    # GPU work here; lease is released on context exit
```

## Important rules

1. Always prefer `uuid` for `CUDA_VISIBLE_DEVICES` unless a benchmark explicitly requires PCI-ordered indices.
2. Never hold GPUs while doing long CPU-only work. Acquire immediately before GPU execution and release right after.
3. Pass the Devin session id so the allocator can attribute leases and clean them up if the session dies:
   - `DEVIN_OUTPOST_SESSION_ID` (preferred) or `DEVIN_SESSION_ID`
   - `--session-id devin-...`
4. If the allocator returns no GPUs (`timeout=0`), either wait with a longer timeout or retry later; do not silently fall back to an unmanaged GPU.

## Server-side zombie cleanup

When the server is started with `DEVIN_API_TOKEN` and `DEVIN_ORG_ID` (or `--devin-api-token` and `--devin-org-id`), the allocator periodically queries the Devin API for each `devin-*` session id. If a session is `exit`, `error`, or `suspended`, its leases are released automatically. This prevents zombie allocations when a Devin agent terminates without releasing GPUs.
