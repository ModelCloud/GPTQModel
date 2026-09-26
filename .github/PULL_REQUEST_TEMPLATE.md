## Summary

Describe the bug, fix, or feature clearly and briefly.

## What Changed

- List the main code changes.
- List any API or behavior changes.
- List any follow-up work that is intentionally out of scope.

## Tests

Every working PR must include at least one new simple, fast, targeted unit test when the change affects behavior, a bug fix, or a regression path.

- [ ] Unit tests have been executed and passed, or the reason they are not applicable is documented.
- [ ] I added a new simple/fast unit test for this change, or documented why that is not applicable.

Paste the exact test commands and results here:

```bash
```

## Review Requirements

AI-assisted code is welcome. The PR must still be reviewed before it is opened as ready for review.

I or an AI agent (such as Codex or Claude) have:

- [ ] Reviewed this PR.
- [ ] Checked that the code matches existing project structure, APIs, and conventions.
- [ ] Avoided unnecessary monkeypatching and used the project's normal extension points where possible.
- [ ] Minimized and compacted the impacted code surface.
- [ ] Considered and eliminated potential regressions.

## Kernel Accuracy Requirements

- [ ] For kernel-related changes, accuracy drift was measured against an applicable independent Torch FP32/FP64 oracle, or the reason this is not applicable is documented.

## Notes

Add any migration notes, risks, compatibility concerns, or reviewer guidance here.
