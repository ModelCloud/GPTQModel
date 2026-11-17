<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kimi-K2.5 tokenizer qualification fixtures

These **test-only**, non-weight files are byte-for-byte copies from
[`moonshotai/Kimi-K2.5`](https://huggingface.co/moonshotai/Kimi-K2.5/tree/4d01dfe0332d63057c186e0b262165819efb6611),
pinned at `4d01dfe0332d63057c186e0b262165819efb6611`.
They preserve the accompanying Moonshot [Modified MIT License](LICENSE),
including its commercial display requirement, and [attribution](NOTICE).

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `tiktoken.model` | 2795286 | `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103` |
| `chat_template.jinja` | 3759 | `7c91e4b3578b518bfb388d2e3dc645daf0fbf843656e887db75d12891e0030ff` |
| `tokenizer_config.json` | 4862 | `12fcab43d2b6068f46769f5ff373960bf7c17a94d7abbc50e2491306b2f6cf58` |
| `LICENSE` | See file | `c524b41658421b8b143c5b50637ef008909da56013ad12ccaba0536139ee35b0` |

Tests copy only the three tokenizer assets into temporary synthetic checkpoints,
then add hostile metadata to those temporary copies; these source fixtures stay
unchanged. They exercise real token IDs, chat/tools/thinking semantics,
skip-special decoding, shared processor identity and quantized save/reload.
The scoped `.gitattributes` preserves the exact upstream template whitespace
and suppresses the large vocabulary diff.

No runtime code reads these fixtures. Tokenizer vocabularies and templates come
from the caller's checkpoint. `setup.py` excludes tests from packages and
`MANIFEST.in` prunes `tests/`, so these assets do not ship as runtime data.
See the [native binding provenance and qualification](../../../gptqmodel/models/ports/kimi_k25/README.md).
