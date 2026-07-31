---
name: run-failing-cli-with-telemetry
description: When the user reports a bug by pasting the exact CLI command that produced it, run that command, capture full telemetry, and use the output to locate the root cause instead of guessing.
---

# Run failing CLI with telemetry

## When to use

- The user provides a complete CLI command that triggers a bug.
- The failure is easier to diagnose from runtime output than from static source inspection.
- You are tempted to grep/read source before reproducing the failure.

## Steps

1. **Run the exact command as-is.**
   Do not paraphrase, simplify, or pre-filter arguments. If it needs secrets, a specific working directory,
   or environment variables, resolve those first, then execute the command exactly as given.

2. **Attach telemetry before the first run.**
   Capture at minimum: the full command, current working directory, environment (`env`, `pip list`, package versions),
   `nvidia-smi` for GPU issues, stdout/stderr, and exit code. For performance or CUDA bugs, wrap the run in
   `nsys profile` or `torch.profiler` covering the failing region.

3. **Reproduce minimally only after the original fails.**
   If the full command is too large or slow, isolate the smallest still-failing invocation while preserving the
   same environment and arguments.

4. **Read source after the failure is confirmed.**
   Use the error text, stack trace, and telemetry to identify the relevant file and lines. Avoid broad grep loops;
   grep only the symbol or function surfaced by the traceback.

5. **Report what the telemetry revealed.**
   Cite the exact failing command, the error text, the environment, and the root-cause file/line. Do not present
   unsubstantiated hypotheses.

## Anti-patterns

- Do not "pre-investigate" by reading unrelated code.
- Do not replace the user's exact command with a guessed minimal reproducer.
- Do not add telemetry only after several speculative edits; add it on the first run.
