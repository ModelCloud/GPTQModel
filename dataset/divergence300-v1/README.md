# Divergence-300 v1

This directory contains the immutable Divergence-300 JSONL datasets used by
the QVQ experiment ledger:

| File | SHA-256 |
| --- | --- |
| `divergence300-development.jsonl` | `701916fbf75844fd66a6ad294cd49c3e2f8bc909746b60c351edeaeb77ace5b2` |
| `divergence300-locked.jsonl` | `17151e98b2e34587c9af58a6736c875c82854564f45c763f027090f35b8e8f58` |
| `divergence300-manifest.json` | `c4e92346ee7b9b1de86651e7ae236f53404724dd5276f3b89ecde5453cddf83d` |

The development split contains 300 prompts and is the D300 diagnostic used by
the current Llama 3.2 1B QVQ YAQA best-configuration record. The locked split
is distributed for immutable provenance and contamination checks; it was not
used for that reported development score.
