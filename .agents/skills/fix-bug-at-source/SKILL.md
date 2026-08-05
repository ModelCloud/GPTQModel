---
name: fix-bug-at-source
description: Trace a bug symptom to its upstream root cause before patching the immediate failure point. Use whenever a local failure looks like it should be fixed by adding information or state that the local site does not naturally own.
---

# Fix bugs at the source, not at the symptom

Use this skill when you see a failure inside a low-level function and your first instinct is to patch that function with a local fallback, heuristic, or extra lookup. Stop and trace *why* the local function does not have the information it needs.

## When to use

- A leaf utility (`LazyTurtle`, a kernel wrapper, a logger, etc.) fails because a string/path/index is missing or wrong.
- The only obvious local fix is to derive the missing value from global state (`named_modules()`, `sys._getframe()`, file-system probes, regex on class names).
- The failure reappears in a sibling call site or requires the same lookup to be duplicated.
- You find yourself adding a comment like "we do not know the real X here, so guess/infer/scan for it".

## PR 204 case study

### Symptom

`DeepseekV4HashRouter.forward` raised:

```
AttributeError: 'DeepseekV4HashRouter' object has no attribute 'tid2eid'
```

during the first layer's calibration input capture. The buffer `tid2eid` is a real checkpoint tensor under `model.layers.0.mlp.gate.tid2eid`.

### First local patch

`StageInputsCapture.cache_inputs` was using the layer's **class name** (`DeepseekV4DecoderLayer`) as `module_path` when it called `shell_module_materialize`. `LazyTurtle` therefore searched for `DeepseekV4DecoderLayer.mlp.gate.tid2eid`, found no match, and silently dropped the buffer. The hotpatch was to scan `self.gptq_model.model.named_modules()` to recover the dotted path (`model.layers.0`).

### Upstream root cause

The dotted path was already known: `ModuleLooper._loop_impl` calls `get_layers_with_prefixes(...)` and receives `layers` **and** `layer_names`. It passed `layers` into `cache_inputs` but discarded `layer_names`. `StageInputsCapture` had no way to know the correct `module_path` without reverse-engineering it from the model tree.

### Correct fix

1. Add `layer_names` as a parameter to `ModuleLooper.cache_inputs` and `StageInputsCapture.cache_inputs`.
2. Use `layer_names[0]` as the `module_path` for the first layer.
3. Keep the old `full_name` / `named_modules()` fallbacks only as safety nets.

This removes the need for the local scan and makes the data flow explicit: the caller that knows the path passes it down.

## Workflow

1. **Reproduce the failure with a minimal case.**
   - Write a focused test or run the smallest script that triggers the bug.
   - Capture the exact function, line, and variable that is wrong or missing.

2. **Add telemetry before you patch — do not fly through a state blackhole.**
   - Log the values that the failing function actually has and the values it expected.
   - For long-running quantization or training jobs, emit periodic state snapshots (layer index, expert index, tensor names/devices, free VRAM, active PID) so a later failure can be rewound to the last known good state.
   - Snapshot module state before materialization: `named_modules()` keys, `state_dict()` keys, `device`/dtype, and any `LazyTurtle` metadata that maps runtime names to checkpoint keys.
   - Prefer targeted logging or cheap NVTX markers over noisy per-element prints; see `$gptqmodel-telemetry` for lightweight instrumentation patterns.
   - Do not skip this step to save time — without telemetry you cannot distinguish "fixed" from "did not reach the bug region".

3. **Ask: where should this value come from?**
   - Is it a configuration constant, a model architecture constant, or derived from user input? It should be passed in.
   - Is it already computed by an upstream caller? It should be forwarded.
   - Is it truly local (e.g., a cache key internal to this function)? A local derivation is okay.

4. **Check the call chain for discarded information.**
   - Trace every function from the failure site back to the user-facing API.
   - Look for lists/tuples/dicts that are unpacked or filtered, dropping context.
   - Look for hard-coded fallbacks like `type(layer).__name__`, `__class__.__name__`, or `str(mod)`.

5. **Prefer plumbing over guessing.**
   - Add the needed parameter to the affected function signature.
   - Update each caller to supply the correct value.
   - If a value is optional, use `None` as the default and fall back to the old heuristic only as a last resort.

6. **Add tests that enforce the contract.**
   - Test that the correct value is forwarded.
   - Test each fallback path (correct value, `full_name`, scan, class-name) independently.
   - Do not test the symptom alone; test that the *correct source data* reaches the leaf.

7. **Document the anti-pattern.**
   - Leave a comment where the fallback lives explaining *why* it can be missing and what the upstream should have provided.
   - If the fallback remains, file or reference a follow-up issue to remove it.

## Key files from PR 204

- `gptqmodel/looper/module_looper.py` — caller that had `layer_names` but did not pass it.
- `gptqmodel/looper/stage_inputs_capture.py` — leaf that needed the dotted path.
- `tests/test_stage_inputs_capture.py` — micro-tests proving the three resolution paths.

## Anti-patterns

- Adding a local `named_modules()` scan, `__class__.__name__` fallback, or regex when the caller already knows the right value.
- Duplicating the same derivation in multiple leaf functions instead of computing it once upstream.
- Writing tests that only exercise the symptom and do not assert the data flow.
- Treating the first working patch as the final fix without asking "who should have told me this?".

## Checklist

- [ ] The failure is reproduced with a focused test or minimal script.
- [ ] Telemetry/state snapshots are captured before the patch.
- [ ] The missing/wrong value is identified.
- [ ] The call chain is traced back to where the value is first known.
- [ ] The value is plumbed to the failure site through signatures, not inferred.
- [ ] Old heuristics remain only as explicit fallbacks, not primary logic.
- [ ] Tests cover the new contract and each fallback.
- [ ] A follow-up note exists if the fallback cannot be removed in this PR.
