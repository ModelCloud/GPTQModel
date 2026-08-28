# Xcode/MLX Metal workflow

## Preflight

Run these checks from the repository root and record the output in the profiling report:

```bash
xcode-select -p
xcodebuild -version
xcrun --find xctrace
xctrace list templates | rg -i 'Metal|GPU'
xctrace list instruments | rg -i 'Metal|GPU'
```

For MLX, record the device and backend explicitly:

```bash
python3 -c 'import mlx.core as mx; print(mx.default_device()); print(mx.metal.device_info())'
pmset -g batt
pmset -g custom
```

Use a dedicated artifact directory such as `/tmp/qvq-metal-profile-<date>-<commit>`. Do not use a repository path for
large capture bundles.

## MLX GPU capture

MLX exposes a bounded Metal capture API. Complete warmup first, then capture only the representative calls:

```python
import mlx.core as mx

# Build inputs and compile/warm the exact workload before this point.
for _ in range(warmup):
    y = workload()
    mx.eval(y)
    mx.synchronize()

mx.metal.start_capture("/tmp/qvq-metal-profile/run.gputrace")
for _ in range(active_calls):
    y = workload()
    mx.eval(y)
    mx.synchronize()
mx.metal.stop_capture()
```

If capture fails with `Capture layer is not inserted`, retry the launched process with `MTL_CAPTURE_ENABLED=1`. Keep
the capture bounded; `.gputrace` directories can become very large because resources are included.

Open the resulting bundle in Xcode GPU Frame Capture. The command-line `xctrace export` tool does not generally parse an
MLX `.gputrace` bundle; use Xcode for per-dispatch counters/resource inspection and retain the bundle path in the
report.

## Metal System Trace

Use an absolute interpreter path and a separate output bundle:

```bash
xctrace record \
  --template 'Metal System Trace' \
  --output /tmp/qvq-metal-profile/system.trace \
  --launch -- \
  /absolute/path/to/python3 scripts/your_workload.py \
  --warmup 50 --active-calls 10
```

For a short target process, the recording ends when the process exits. Export the table of contents and the relevant
interval tables:

```bash
xctrace export --input /tmp/qvq-metal-profile/system.trace \
  --toc --output /tmp/qvq-metal-profile/toc.xml
xctrace export --input /tmp/qvq-metal-profile/system.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-application-intervals"]' \
  --output /tmp/qvq-metal-profile/application.xml
xctrace export --input /tmp/qvq-metal-profile/system.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output /tmp/qvq-metal-profile/gpu.xml
```

Use Metal Application intervals for command-buffer/encoder ordering and Metal GPU intervals for coarse GPU activity.
The default Metal System Trace configuration may show `Counter Set: (null)` and `Shader Timeline: Disabled`; that is
valid scheduling evidence, not a hardware-counter report.

## GPU counters and failure handling

Try the Metal GPU Counters template only for a bounded, warmed workload. If the trace reports that the selected counter
profile is unsupported on the target device, OS, or Xcode combination:

1. preserve the error text in the report;
2. do not substitute guessed occupancy, register, bandwidth, or stall values;
3. use GPU Frame Capture if it produces a valid bundle;
4. use source-backed barrier/load topology and synchronized A/B timing;
5. label the result as structural profiling, not counter profiling.

## Source-level audit

Before proposing an optimization, inspect the exact generated source/template:

```bash
rg -n 'threadgroup_barrier|simd_shuffle|simd_broadcast|simdgroup_matrix|atomic|threadgroup|constant' \
  gptqmodel/utils/qvq_mlx.py
```

Build a small table:

| Phase | Producer/consumer | Synchronization | Idle work | Candidate |
|---|---|---|---|---|
| decode | decoder lanes → compute lanes | barrier or SIMD shuffle | sibling SIMD groups | cooperative decode |
| metadata | selector/codeword loads → lanes | none/SIMD shuffle | duplicate loads | lane-0 broadcast |
| reduction | split partials → output | MLX graph/kernel boundary | output waits | fixed fused reduction |
| epilogue | GEMV → transform/scale/bias | data dependency | producer/consumer boundary | fused epilogue |

Do not infer a stall from a barrier token alone: first establish how many SIMD groups are in the threadgroup and whether
the producer and consumers have a true dependency.

## Final A/B protocol

Run baseline and candidate in the same process when possible, with identical random seed/payload, shape order, warmup,
active samples, synchronization, and power state. Report median and p95 at both levels:

```text
inner kernel:  baseline p50/p95, candidate p50/p95, baseline/candidate
full module:  baseline p50/p95, candidate p50/p95, baseline/candidate
```

Use profiler traces to explain a difference, not to replace the synchronized benchmark. If a specialization is faster
only for one shape, gate it explicitly by shape/device and retain the tested fallback.

## References

- [Apple Metal tools](https://developer.apple.com/metal/tools/)
- [Capturing a Metal workload in Xcode](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-in-xcode)
- [Analyzing Apple GPU performance using counter statistics](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
- [Porting Metal code to Apple silicon](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
