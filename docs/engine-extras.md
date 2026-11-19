<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Engine extras: vLLM and SGLang pins

GPT-QModel's `vllm` and `sglang` extras install an inference engine into the
same environment as the core package. A pip/uv environment has one Torch and
one Transformers, so an extra is only installable when the engine release's
published pins intersect `requirements.txt`. Engine releases pin both
libraries exactly and lag the core floors, so most combinations have no
solution and the resolver fails before anything is installed.

## Core floors

| Interpreter | Torch | Transformers |
| --- | --- | --- |
| `python_version < "3.11"` | `>=2.13.0,<2.14.0` | `>=5.20.2` |
| `python_version >= "3.11"` | `>=2.15.0` | `>=5.23` |

## Published engine pins

| Engine | Release | Torch | Transformers | Intersects core |
| --- | --- | --- | --- | --- |
| vLLM | `0.30.0` | `==2.13.0` | `>=5.10.4` | 3.10 only |
| vLLM | `0.31.0` | `==2.13.0` | `>=5.10.4,<5.18.0` | no (Transformers floor) |
| vLLM | `0.32.0` | `==2.13.0` | `>=5.16.1,<5.20.0` | no (Torch floor) |
| vLLM | `0.33.0` | `==2.14.0` | `>=5.16.1,<5.21.0` | no (Torch floor) |
| vLLM | `0.34.0` | `==2.14.0` | `>=5.16.1,<5.23.0` | no (Torch and Transformers floors) |
| SGLang | `0.5.20`–`0.5.21` | `==2.13.0` | `==5.12.1` | no (Transformers floor) |
| SGLang | `0.5.22`–`0.5.23` | `==2.14.1` | `==5.19.0` | no (Torch and Transformers floors) |

`vllm>=0.32.0` also declares `Requires-Python >=3.11`, so those releases are
not candidates on 3.10 at all.

## Advertised extras

`pyproject.toml` advertises only what resolves:

```toml
vllm = [
    "vllm>=0.30.0,<0.31; python_version < \"3.11\"",
]
```

There is no `sglang` extra. SGLang publishes an exact Transformers pin on
every release (`5.12.1` or `5.19.0`), below the `5.20.2` and `5.23` core
floors, so no SGLang release can share an environment with this package. The
extra is restored when upstream publishes a release whose pins intersect the
core floors; until then SGLang integration tests run in a separate environment
whose Transformers pin comes from SGLang, as `.github/scripts/deps.yaml` does.

The remaining interpreter combinations are deliberately unadvertised:
`gptqmodel[vllm]` on 3.11+ installs the base package only, and
`gptqmodel[sglang]` is an unknown extra. Both are resolver-level outcomes of
the pins above, not a bug in the extra marker.

## Checking a combination

Compile the extra for the interpreter you intend to use. The command fails
with the conflicting requirement when no release intersects the core floors,
and prints the selected Torch and Transformers otherwise:

```bash
uv pip compile --python-version 3.10 \
    --python-platform x86_64-manylinux_2_35 --extra vllm pyproject.toml \
    | grep -E '^(vllm|torch|transformers)=='
```

`scripts/check_engine_extras.py` runs the same compilation for every
advertised extra and representative interpreter, and reports the selected
Torch and Transformers per lane:

```bash
python3 scripts/check_engine_extras.py
```

Re-run it after bumping an engine floor. A lane that starts resolving is a
signal to widen the extra marker; a lane that stops resolving means the
advertised range no longer intersects the core floors.
