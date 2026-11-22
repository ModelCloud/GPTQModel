<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GPT-QModel contracts

A contract is a fixed rule about how GPT-QModel components are called in real
execution. Code may assume it, tests must respect it, and nobody may "fix" a
component so that it tolerates a caller that breaks it.

Contracts outrank skills, docs, issues and PR descriptions. If one of those
conflicts with a contract, the contract wins and the other text is wrong. Fix
the text, or close the issue or PR.

## Index

| Area | Contract | Rule in one line |
| --- | --- | --- |
| Kernel / module | [Forward dtype](kernel/forward-dtype.md) | Kernel modules use one input dtype (output dtype = input dtype), and their weight dtypes are frozen after the first forward. Quantization may rewrite source weight values in their existing dtype, but never casts live modules. A violation is a caller bug and must raise. |
| Model / container | [Re-fusion](model/refusion.md) | A fused checkpoint tensor holds one quantization: only modules with an identical quantization signature may be re-fused into it, and never a quantized kernel module. A violation raises before any tensor is created. |

## Rules for agents and contributors

1. **Read before you touch.** Before you change, review, audit, or file an issue
   against any area in the index, read its contract.
2. **Never design for a contract violation.** Do not add code, caches, tests
   or guards whose only purpose is to make a violating call work. This includes
   "call-local" copies, per-dtype caches, and invalidation when the dtype changes.
3. **Reproductions must honour contracts.** If an issue's reproduction only
   fails when it breaks a contract, the issue is invalid. Do not claim it.
   Close it with a link to the contract. The same applies to PRs.
4. **Cite the contract.** When code relies on a contract, or a review rejects
   work because of one, link the contract file.
5. **Changing a contract** needs a maintainer-approved PR that edits the
   contract file first. Never relax a contract inside an unrelated fix.

## Adding a contract

Put each contract in `contracts/<area>/<name>.md` and add a row to the index
above. Each contract states:

- the rule
- what counts as a violation
- what stays allowed
- how code enforces it
- how tests must exercise it
