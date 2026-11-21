<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Engine extras: vLLM and SGLang pins

An inference engine installed as a GPT-QModel extra shares the same Torch and
Transformers versions as the core package. An extra is installable when the
engine release's published pins intersect `requirements.txt`.

## Core floors

| Interpreter | Torch | Transformers |
| --- | --- | --- |
| Python 3.11 and later | `>=2.15.0` | `>=5.23.0,<5.24` |

## Published engine pins

| Engine | Release | Torch | Transformers | Intersects core |
| --- | --- | --- | --- | --- |
| vLLM | `0.30.0` | `==2.13.0` | `>=5.10.4` | no (Torch floor) |
| vLLM | `0.31.0` | `==2.13.0` | `>=5.10.4,<5.18.0` | no (Transformers floor) |
| vLLM | `0.32.0` | `==2.13.0` | `>=5.16.1,<5.20.0` | no (Torch floor) |
| vLLM | `0.33.0` | `==2.14.0` | `>=5.16.1,<5.21.0` | no (Torch floor) |
| vLLM | `0.34.0` | `==2.14.0` | `>=5.16.1,<5.23.0` | no (Torch and Transformers floors) |
| SGLang | `0.5.20`–`0.5.21` | `==2.13.0` | `==5.12.1` | no (Transformers floor) |
| SGLang | `0.5.22`–`0.5.23` | `==2.14.1` | `==5.19.0` | no (Torch and Transformers floors) |

## Advertised extras

The `vllm` and `sglang` extras are unavailable. A compatible engine release
must support Torch 2.15 or newer and Transformers 5.23 on Python 3.11 or newer.

## Checking a combination

`scripts/check_engine_extras.py` runs the same compilation for every
advertised extra and representative interpreter, and reports the selected
Torch and Transformers per lane.

```bash
python3 scripts/check_engine_extras.py
```

Run the script with a proposed extra declaration to verify its pins before
publishing it.
