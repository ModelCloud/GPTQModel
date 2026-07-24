---
name: gptqmodel-upstream
description: Prepare, audit, publish, or review changes for the public ModelCloud/GPTQModel repository. Use for upstream branches, commits, pull requests, issues, reviews, comments, release notes, or ports from another development tree. Enforce a strict disclosure boundary so upstream-visible content refers only to code, tests, and behavior available in GPTQModel.
---

# GPTQModel upstream

Treat the public GPTQModel repository as an independent disclosure boundary. Describe every change from the
public upstream source and never reveal where private or downstream development occurred.

## Work from public source

1. Use a separate clean clone or worktree based on the public upstream default branch.
2. Inspect the upstream implementation before editing. Do not assume another tree's symbols, features, tests,
   history, or configuration exist upstream.
3. Implement the smallest self-contained upstream change. Include only dependencies that are present in the
   public tree or are part of the same reviewed patch.
4. Explain the problem using public symbols and observable behavior only. Do not describe the change as a port
   from another repository or branch.

## Keep upstream-visible surfaces clean

Never expose any of the following in upstream branches, commits, code, tests, documentation, PRs, issues,
comments, reviews, check output, or uploaded artifacts:

- downstream repository, product, branch, commit, PR, issue, or internal feature names;
- comparisons against non-upstream implementations or features;
- local/private model, checkpoint, dataset, adapter, path, host, storage, log, or artifact identifiers;
- testing artifact names, model names, or run-specific quantization/evaluation configurations;
- private benchmark results, hardware allocation details, work-queue logs, or unreleased roadmap context.

Use only:

- symbols, files, features, and behavior present in the public GPTQModel tree;
- synthetic or generic fixtures committed with the upstream tests;
- public test paths, pass/fail results, and implementation-neutral correctness statements;
- generic descriptions such as role-specific kernel, packed endpoint, or per-module quantization contract.

## Run the disclosure gate

Before every upstream commit, push, PR update, issue, comment, or review:

1. Inspect `git status`, the complete staged diff, and every commit that will be published.
2. Audit branch names, commit messages, source comments, test names, fixtures, documentation, and generated
   artifacts for downstream or local identifiers.
3. Draft the upstream text separately and audit its title, body, links, code blocks, paths, tables, and logs.
4. Remove unexplained external provenance, private test details, and links outside public upstream context.
5. Confirm the description remains understandable from the public patch alone.

If an existing upstream-visible surface leaks restricted context, stop new publication work, sanitize every
editable surface immediately, and verify comments/reviews before continuing.

## Publish narrowly

- Push only the audited upstream branch.
- Keep PR validation statements tied to committed public tests or generic smoke behavior.
- Do not link downstream reports or use downstream commits as evidence.
- In the handoff, link only the public upstream PR/commit and summarize only public upstream behavior.
