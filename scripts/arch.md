# CI Architecture

## Naming

- `.github/scripts/ci_*.py` are the only workflow entrypoints.
- Shared data stays in `.github/scripts/*.yaml`.
- Shared logic lives in reusable modules instead of one-off CLIs.

## GPTQModel unit test flow

1. `check-vm`
- `ci_workflow.py check-vm` computes `ip`, `run_id`, `install_ts`, and matrix parallelism, then writes them to `GITHUB_OUTPUT`.

2. `list-test-files`
- `ci_workflow.py list-tests` scans `tests/`, filters ignored and regex-matched cases, splits them into torch/model/mlx buckets, and builds shared env matrices from `deps.yaml` and `test.yaml`.

3. `prepare`
- `ci_workflow.py prepare-common-envs` deduplicates shared env rows, creates or refreshes those uv envs in parallel, installs base requirements, and syncs the repo-scoped git dependencies.

4. `torch` and `torch-models`
- `ci_workflow.py activate-test-env` resolves `GPU_COUNT`, `HAS_SPECIFIC_DEPS`, `ENV_NAME`, and `UV_CACHE_DIR`.
- `ci_workflow.py setup-specific-env` applies per-test compiler/python settings and test-specific install/uninstall package rules from `deps.yaml` and `blacklist.yaml`.
- `ci_workflow.py install-package` serializes source installs with lock files so only one job populates a dedicated env at a time.
- `.github/scripts/ci_workflow.py` owns workflow-oriented commands: test discovery, per-test env resolution, and package-version matrix generation.
- `.github/scripts/ci_deps.py` owns dependency install/uninstall commands and their shared YAML/package logic.
- `.github/scripts/ci_gpu.py` owns allocator lease commands and the shared allocator client logic.
- `.github/scripts/ci_tests.py` owns pytest execution and failure-log extraction.
- `ci_tests.py check-log` prints the same failure excerpts the old shell step grepped from the test log.

## Config files

- `.github/scripts/test.yaml`: per-group and per-test runtime config, mainly Python version and GPU count.
- `.github/scripts/deps.yaml`: extra packages needed by individual tests or directories.
- `.github/scripts/blacklist.yaml`: packages that must be removed for specific tests.

## Maintenance rule

- Add new workflow behavior by extending an existing `ci_*` entrypoint first.
- Only keep shell in workflow files for minimal glue such as checkout, `source /opt/uv/setup_uv_venv.sh ...`, and GitHub expression wiring.
