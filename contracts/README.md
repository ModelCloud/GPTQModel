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
| Kernel / module | [Forward dtype](kernel/forward-dtype.md) | The first forward sets the activation dtype. Every later forward uses the same dtype, and the output dtype matches the input dtype. After the first forward, weight and buffer dtypes are frozen. A different input dtype, or a later weight dtype change, is a caller bug and must raise. |

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
