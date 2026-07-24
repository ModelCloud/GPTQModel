---
name: ai-agent-terminal
description: Environment and CLI tuning for running GPT-QModel from AI agents such as Devin, Codex, or Claude. Use when an agent session is seeing excessive progress-bar output, terminal-animation spam, or needs a quiet, parseable, reproducible run.
---

# AI Agent Terminal Environment Tuning

AI agents consume logs, not interactive terminals. The defaults of many Python libraries assume a live human CLI with animated spinners and frequent screen redraws. When an agent drives GPT-QModel, set the environment up front so output is quiet, deterministic, and easy to parse.

## 1. Disable LogBar terminal animation

GPT-QModel uses `logbar` for progress output. In a non-TTY/agent context this produces high-volume frame updates that are not useful to models and can flood the context window.

```bash
export LOGBAR_ANIMATION=0
export LOGBAR_PROGRESS_OUTPUT_INTERVAL=1000
export PYTEST_CURRENT_TEST=1
```

- `LOGBAR_ANIMATION=0` disables animated spinners and title scrolling.
- `LOGBAR_PROGRESS_OUTPUT_INTERVAL=N` renders progress only every N logical updates.
- `PYTEST_CURRENT_TEST=1` tells `gptqmodel.utils.logger.setup_logger().pb()` to return `_SilentProgress` when stdout is not a TTY.

Set these in the parent shell before launching benchmark or test subprocesses so the child inherits them.

## 2. Force deterministic, parseable subprocess output

```bash
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
```

- `PYTHONUNBUFFERED=1` avoids line-buffering surprises in child output.
- `PYTHONDONTWRITEBYTECODE=1` prevents `.pyc` churn when iterating in an agent session.

## 3. Pin GPU visibility to physical PCI bus order

Agents should never assume a fixed CUDA ordinal maps to a specific physical GPU. Set bus-order discovery first:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<physical_pci_bus_ordered_id>
```

Inside the process the visible device will usually be `cuda:0`, but the environment is constrained to the exact physical GPU requested. Record the physical ID, PCI bus ID, and UUID in the run log.

## 4. Disable interactive or color-dependent tooling

```bash
export TERM=dumb
export NO_COLOR=1
export ANSI_COLORS_DISABLED=1
```

Use `TERM=dumb` when a script checks `isatty()` before emitting progress bars. `NO_COLOR`/`ANSI_COLORS_DISABLED` strip ANSI escapes from tools that honor them.

## 5. Keep Hugging Face/tokenizer caches predictable

```bash
export HF_HOME=/monster/data/hf
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
```

- `HF_HOME` pins the cache to a known path when running shared infrastructure.
- `TOKENIZERS_PARALLELISM=false` avoids fork/parallelism warnings in subprocess-heavy agent runs.

## 6. Emit machine-readable summary lines from benchmarks

Benchmark scripts should print lines in a stable format that is easy for the agent to grep, for example:

```text
BENCH_LOAD_TIME: 1.290
BENCH_QUANT_TIME: 116.056
BENCH_PEAK_MEM_GB: 2.084
BENCH_FUSE: 1
```

Avoid rich tables, interactive curses, or frames. If a table is needed, print it once at the end.

## 7. One-shot template for agent runs

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export LOGBAR_ANIMATION=0
export LOGBAR_PROGRESS_OUTPUT_INTERVAL=1000
export PYTEST_CURRENT_TEST=1
export TERM=dumb
export NO_COLOR=1
export TOKENIZERS_PARALLELISM=false
```

Add these to the environment block of every `exec` call that runs GPT-QModel code from an agent session.
