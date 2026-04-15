import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from ci_catalog import (
    DEFAULT_DEPS_FILE,
    DEFAULT_TEST_CONFIG_FILE,
    build_env_name,
    build_shared_env_matrix,
    collect_prepare_env_rows,
    has_specific_dependencies,
    list_test_files,
    parse_test_config,
    resolve_install_packages,
    resolve_uninstall_packages,
)
from ci_common import append_github_env, append_github_output, load_json, run_command


def command_check_vm(args: argparse.Namespace) -> int:
    run_id = args.github_run_id
    install_ts = args.install_ts or str(int(time.time()))
    max_parallel = str(args.max_parallel or "20").strip()
    max_parallel_json = json.dumps({"size": int(max_parallel)})

    append_github_output("ip", args.runner_ip)
    append_github_output("run_id", run_id)
    append_github_output("install_ts", install_ts)
    append_github_output("max-parallel", max_parallel_json)

    print(f"ip: {args.runner_ip}")
    print(f"run_id={run_id}")
    print(f"install_ts={install_ts}")
    print(f"max-parallel={max_parallel_json}")
    return 0


def command_list_tests(args: argparse.Namespace) -> int:
    deps = DEFAULT_DEPS_FILE if args.deps_file is None else args.deps_file
    config = DEFAULT_TEST_CONFIG_FILE if args.config_file is None else args.config_file
    deps_data = load_yaml_path(deps)

    torch_files, model_files, m4_files = list_test_files(
        ignored_test_files=args.ignored_test_files,
        test_names=args.test_names,
        test_regex=args.test_regex or ".*",
        tests_root=args.tests_root,
    )
    torch_envs = build_shared_env_matrix(
        group="tests",
        tests=torch_files,
        deps=deps_data,
        config_file=config,
    )
    model_envs = build_shared_env_matrix(
        group="tests/models",
        tests=model_files,
        deps=deps_data,
        config_file=config,
    )

    outputs = {
        "torch-files": json.dumps(torch_files),
        "model-files": json.dumps(model_files),
        "m4-files": json.dumps(m4_files),
        "torch-envs": json.dumps(torch_envs),
        "model-envs": json.dumps(model_envs),
    }
    for name, value in outputs.items():
        append_github_output(name, value)

    print(f"Test files: {outputs['torch-files']}|{outputs['model-files']}|{outputs['m4-files']}")
    print(f"Torch Test files: {outputs['torch-files']}")
    print(f"Model Compat Test files: {outputs['model-files']}")
    print(f"MLX Test files: {outputs['m4-files']}")
    print(f"Torch envs: {outputs['torch-envs']}")
    print(f"Model envs: {outputs['model-envs']}")
    print(f"Ignored Test files: {args.ignored_test_files}")
    return 0


def load_yaml_path(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_prepare_env_script(
    *,
    env_name: str,
    python_version: str,
    cuda_version: str,
    torch_version: str,
    runner: str,
    repo_urls: list[str],
) -> str:
    install_urls = "\n".join(f'  "{url}" \\' for url in repo_urls[:-1])
    install_urls += f'\n  "{repo_urls[-1]}"'
    return textwrap.dedent(
        f"""\
        set -e
        echo "::group::updating env for {env_name}\t{python_version}"
        echo "preparing env={env_name} py={python_version}"
        source /opt/uv/setup_uv_venv.sh "{env_name}"

        python -V
        which python
        which pip || true

        bash /opt/env/init_compiler_no_env.sh "{cuda_version}" "{torch_version}" "{python_version}"

        uv pip uninstall gptqmodel || true
        /opt/env/install_torch.sh "{cuda_version}" "{torch_version}"
        uv pip install -r requirements.txt -i "http://{runner}/simple/" --trusted-host "{runner}" --extra-index-url https://pypi.org/simple
        for pkg in defuser pypcre tokenicer logbar evalution; do
          uv pip uninstall "$pkg" || true
        done
        for url in \\
{install_urls}
        do
          uv pip install "$url"
        done
        /opt/env/install_torch.sh "{cuda_version}" "{torch_version}"

        echo "::group::pip list"
        uv pip list
        echo "::endgroup::"
        echo "::endgroup::"
        """
    )


def command_prepare_common_envs(args: argparse.Namespace) -> int:
    torch_envs = load_json(args.torch_envs_json, default=[])
    model_envs = load_json(args.model_envs_json, default=[])
    env_rows = collect_prepare_env_rows(torch_envs, model_envs)

    for index, (env_name, python_version) in enumerate(env_rows):
        print(f"{index}: {env_name}\t{python_version}")

    if args.plan_only:
        for env_name, python_version in env_rows:
            print(
                build_prepare_env_script(
                    env_name=env_name,
                    python_version=python_version,
                    cuda_version=args.cuda_version,
                    torch_version=args.torch_version,
                    runner=args.runner,
                    repo_urls=args.repo_url,
                )
            )
        return 0

    processes: list[subprocess.Popen[str]] = []
    for env_name, python_version in env_rows:
        script = build_prepare_env_script(
            env_name=env_name,
            python_version=python_version,
            cuda_version=args.cuda_version,
            torch_version=args.torch_version,
            runner=args.runner,
            repo_urls=args.repo_url,
        )
        processes.append(subprocess.Popen(["bash", "-lc", script]))

    status = 0
    for process in processes:
        if process.wait() != 0:
            status = 1
    return status


def command_activate_test_env(args: argparse.Namespace) -> int:
    deps = load_yaml_path(args.deps_file)
    config = parse_test_config(args.config_file, args.group, args.test_name)
    env_name = build_env_name(args.test_name, deps)
    has_specific = "1" if has_specific_dependencies(args.test_name, deps) else "0"

    if args.fix_safe_directory:
        run_command(
            ["git", "config", "--global", "--add", "safe.directory", "/__w/GPTQModel/GPTQModel"]
        )

    if args.write_gpu_count:
        gpu_count = str(config["gpu"])
        append_github_env("GPU_COUNT", gpu_count)
        print(f"using gpu={gpu_count} specific={has_specific} uv env={env_name}")
    else:
        print(f"using specific={has_specific} uv env={env_name}")

    append_github_env("HAS_SPECIFIC_DEPS", has_specific)
    append_github_env("ENV_NAME", env_name)

    base_uv_cache_dir = os.environ.get("UV_CACHE_DIR", "")
    uv_cache_dir = f"{base_uv_cache_dir.rstrip('/')}/{env_name}"
    Path(uv_cache_dir).mkdir(parents=True, exist_ok=True)
    append_github_env("UV_CACHE_DIR", uv_cache_dir)
    print(f"using uv cache dir={uv_cache_dir}")

    if not args.plan_only:
        run_command(["/opt/uv/setup_uv_venv.sh", env_name])
    else:
        print(f"/opt/uv/setup_uv_venv.sh {env_name}")
    return 0


def install_packages(packages: list[str]) -> None:
    if not packages:
        return

    print("--- Installing deps with uv:")
    for package in packages:
        print(f"  - {package}")

    for package in packages:
        command = ["uv", "pip", "install", "--no-cache", package]
        print(f"installing: {command}")
        subprocess.check_call(command, shell=False)


def uninstall_packages(packages: list[str]) -> None:
    if not packages:
        return

    print("--- Uninstalling deps with uv:")
    for package in packages:
        print(f"  - {package}")

    for package in packages:
        command = ["uv", "pip", "uninstall", package]
        try:
            subprocess.check_call(command, shell=False)
        except Exception as exc:
            print(f"--- Unnstall failed: {exc}")


def command_setup_specific_env(args: argparse.Namespace) -> int:
    config = parse_test_config(args.config_file, args.group, args.test_name)
    python_version = str(config["py"])

    run_command(["python", "-V"])
    run_command(["which", "python"])
    run_command(["bash", "-lc", "which pip || true"], check=False)

    print(
        f"--- setting env... cuda={args.cuda_version} torch={args.torch_version} "
        f"python={python_version}"
    )
    run_command(
        [
            "bash",
            "/opt/env/init_compiler_no_env.sh",
            args.cuda_version,
            args.torch_version,
            python_version,
        ]
    )

    specific_install, common_install = resolve_install_packages(
        args.test_name,
        args.deps_file,
    )
    specific_uninstall, common_uninstall = resolve_uninstall_packages(
        args.test_name,
        args.blacklist_file,
    )

    if args.plan_only:
        print(json.dumps({"install": specific_install + common_install}))
        print(json.dumps({"uninstall": specific_uninstall + common_uninstall}))
        return 0

    print("--- installing required deps...")
    install_packages(specific_install)
    install_packages(common_install)

    print("--- uninstalling required deps...")
    uninstall_packages(specific_uninstall)
    uninstall_packages(common_uninstall)
    return 0


def command_install_package(args: argparse.Namespace) -> int:
    import fcntl

    lock_dir = Path(args.lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)

    lock_file = lock_dir / f"{args.env_name}.{args.install_key}.lock"
    unlock_file = lock_dir / f"{args.env_name}.{args.install_key}.unlock"

    if unlock_file.exists():
        print(f"gptqmodel already installed for {args.env_name} @ {args.install_key}, skipping")
        return 0

    try:
        with lock_file.open("w+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                owner = lock_file.read_text(encoding="utf-8").strip()
                if owner:
                    print(f"lock owner: {owner}")
                print(f"waiting for lock: {lock_file}")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

            if unlock_file.exists():
                print(f"gptqmodel already installed for {args.env_name} @ {args.install_key}, skipping")
                return 0

            handle.seek(0)
            handle.write(
                f"test={args.test_script} job={args.job} run={args.run_id} env={args.env_name}\n"
            )
            handle.flush()

            install_command = [
                "uv",
                "pip",
                "install",
                ".",
                "-i",
                f"http://{args.runner}/simple/",
                "--trusted-host",
                args.runner,
                "--extra-index-url",
                "https://pypi.org/simple",
            ]
            if args.no_build_isolation:
                install_command.insert(4, "--no-build-isolation")

            if args.plan_only:
                print("plan:")
                print("uv pip uninstall gptqmodel || true")
                print(" ".join(install_command))
                print(
                    "uv pip install torch torchvision torchaudio -U "
                    f"--torch-backend={args.uv_torch_backend}"
                )
                return 0

            run_command(["uv", "pip", "uninstall", "gptqmodel"], check=False)
            run_command(install_command)
            run_command(
                [
                    "uv",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "-U",
                    f"--torch-backend={args.uv_torch_backend}",
                ]
            )

            print("::group::pip list")
            run_command(["uv", "pip", "list"])
            print("::endgroup::")

        lock_file.replace(unlock_file)
        return 0
    finally:
        if lock_file.exists():
            lock_file.unlink()


def command_cleanup_install_lock(args: argparse.Namespace) -> int:
    if not args.env_name:
        return 0
    lock_file = Path(args.lock_dir) / f"{args.env_name}.{args.install_key}.lock"
    if lock_file.exists():
        print(f"removing leftover install lock: {lock_file}")
        lock_file.unlink()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_vm = subparsers.add_parser("check-vm")
    check_vm.add_argument("--runner-ip", required=True)
    check_vm.add_argument("--github-run-id", required=True)
    check_vm.add_argument("--max-parallel", default="")
    check_vm.add_argument("--install-ts", default="")
    check_vm.set_defaults(handler=command_check_vm)

    list_tests = subparsers.add_parser("list-tests")
    list_tests.add_argument("--ignored-test-files", default="")
    list_tests.add_argument("--test-names", default="")
    list_tests.add_argument("--test-regex", default="")
    list_tests.add_argument("--tests-root", default="tests")
    list_tests.add_argument("--deps-file", default=str(DEFAULT_DEPS_FILE))
    list_tests.add_argument("--config-file", default=str(DEFAULT_TEST_CONFIG_FILE))
    list_tests.set_defaults(handler=command_list_tests)

    prepare_envs = subparsers.add_parser("prepare-common-envs")
    prepare_envs.add_argument("--torch-envs-json", default="[]")
    prepare_envs.add_argument("--model-envs-json", default="[]")
    prepare_envs.add_argument("--cuda-version", required=True)
    prepare_envs.add_argument("--torch-version", required=True)
    prepare_envs.add_argument("--runner", required=True)
    prepare_envs.add_argument("--repo-url", action="append", required=True)
    prepare_envs.add_argument("--plan-only", action="store_true")
    prepare_envs.set_defaults(handler=command_prepare_common_envs)

    activate_env = subparsers.add_parser("activate-test-env")
    activate_env.add_argument("--group", required=True)
    activate_env.add_argument("--test-name", required=True)
    activate_env.add_argument("--deps-file", default=str(DEFAULT_DEPS_FILE))
    activate_env.add_argument("--config-file", default=str(DEFAULT_TEST_CONFIG_FILE))
    activate_env.add_argument("--write-gpu-count", action="store_true")
    activate_env.add_argument("--fix-safe-directory", action="store_true")
    activate_env.add_argument("--plan-only", action="store_true")
    activate_env.set_defaults(handler=command_activate_test_env)

    setup_env = subparsers.add_parser("setup-specific-env")
    setup_env.add_argument("--group", required=True)
    setup_env.add_argument("--test-name", required=True)
    setup_env.add_argument("--cuda-version", required=True)
    setup_env.add_argument("--torch-version", required=True)
    setup_env.add_argument("--deps-file", default=str(DEFAULT_DEPS_FILE))
    setup_env.add_argument("--blacklist-file", default=str(Path(__file__).with_name("blacklist.yaml")))
    setup_env.add_argument("--config-file", default=str(DEFAULT_TEST_CONFIG_FILE))
    setup_env.add_argument("--plan-only", action="store_true")
    setup_env.set_defaults(handler=command_setup_specific_env)

    install_package = subparsers.add_parser("install-package")
    install_package.add_argument("--env-name", required=True)
    install_package.add_argument("--install-key", required=True)
    install_package.add_argument("--test-script", required=True)
    install_package.add_argument("--job", required=True)
    install_package.add_argument("--run-id", required=True)
    install_package.add_argument("--runner", required=True)
    install_package.add_argument("--uv-torch-backend", required=True)
    install_package.add_argument("--lock-dir", default="/opt/uv/locks")
    install_package.add_argument("--no-build-isolation", action="store_true")
    install_package.add_argument("--plan-only", action="store_true")
    install_package.set_defaults(handler=command_install_package)

    cleanup_lock = subparsers.add_parser("cleanup-install-lock")
    cleanup_lock.add_argument("--env-name", default="")
    cleanup_lock.add_argument("--install-key", required=True)
    cleanup_lock.add_argument("--lock-dir", default="/opt/uv/locks")
    cleanup_lock.set_defaults(handler=command_cleanup_install_lock)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
