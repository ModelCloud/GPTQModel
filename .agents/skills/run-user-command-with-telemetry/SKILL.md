---
name: run-user-command-with-telemetry
description: When the user reports a bug or discrepancy by providing the exact command or execution path, run it as-is, attach telemetry, and use the output/logs to locate the source instead of guessing.
---

# Run user command with telemetry

## When to use

- The user provides a complete command, script, or execution path that shows an error, warning, or discrepancy.
- The issue is visible in the output or logs rather than requiring speculative source inspection.
- You are tempted to grep or read source before reproducing or capturing the evidence.

## Scope

- This is not limited to "CLI" commands. It applies to any reproducible command or workflow the user provides
  (script, benchmark, notebook cell, desktop/automation runner, etc.).
- The target does not need to be a full crash or failure. A clear error or warning in the logs, an unexpected
  metric, or any discrepancy the user points out is enough to trigger this skill.

## Steps

1. **Run the exact command or execution path as-is.**
   Do not paraphrase, simplify, or pre-filter arguments. If it needs secrets, a specific working directory,
   environment variables, or a different runtime context, resolve those first, then execute exactly what the
   user provided.

2. **Capture full output and telemetry before analysis.**
   Capture at minimum: the exact command/workflow, current working directory, environment (`env`, `pip list`,
   package versions), `nvidia-smi` when GPU is involved, complete stdout/stderr, and exit code. Also collect
   any log files the command writes. For performance or CUDA issues, wrap the run in `nsys profile`,
   `torch.profiler`, or an equivalent to record the affected region.

3. **Preserve logs showing errors, warnings, or discrepancies.**
   Do not require a hard crash. If the user says "this log line looks wrong" or "this number is off",
   capture the surrounding context and timestamps. Pin the exact log line, warning, or metric that indicates
   the problem.

4. **Reproduce minimally only after the original behavior is confirmed.**
   If the full command is too large or slow, isolate the smallest invocation that still shows the same error,
   warning, or discrepancy while preserving the original environment and arguments.

5. **Read source only after telemetry identifies the failure site.**
   Use the error message, stack trace, log line, or profiler data to find the relevant file and lines. Avoid
   broad grep loops; grep only the symbol or function surfaced by the output.

6. **Report what the telemetry revealed.**
   Cite the exact command, the relevant log/error text, the environment, and the root-cause file/line. Do not
   present unsubstantiated hypotheses.

## Anti-patterns

- Do not "pre-investigate" by reading unrelated code.
- Do not replace the user's exact command with a guessed minimal reproducer.
- Do not require a crash before capturing telemetry; warnings and logged discrepancies are valid triggers.
- Do not add telemetry only after several speculative edits; add it on the first run.
