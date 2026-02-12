import os
import subprocess
import sys
import yaml
from pathlib import Path

base_dir = os.path.dirname(os.path.abspath(__file__))

def resolve_test_path(raw_name: str) -> Path:
    return Path("tests") / f"{raw_name}.py"

def normalize_pkg_spec(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    # 已是 git 形式，直接用
    if s.startswith("git+"):
        return s

    # 裸 github URL
    if s.startswith("https://github.com/"):
        s = s.rstrip("/")
        if not s.endswith(".git"):
            s += ".git"
        return "git+" + s

    # 其它（普通 pip 包）
    return s

def collect_pkgs(test_path: Path, deps: dict):
    specific_pkgs = set()

    common_pkgs = set(deps.get("common") or [])

    # 文件级依赖（tests: 下）
    specific_pkgs.update(deps.get("tests", {}).get(test_path.name) or [])

    # 目录级依赖（tests/...）
    test_path_str = test_path.as_posix()
    for key, value in deps.items():
        if not (isinstance(key, str) and key.startswith("tests/")):
            continue
        if not test_path_str.startswith(key + "/"):
            continue

        # 1) 目录直接是列表：tests/models: [jieba, ...]
        if isinstance(value, list):
            specific_pkgs.update(value)

        # 2) 目录是 dict：tests/models: { test_x.py: [..], ... }
        elif isinstance(value, dict):
            specific_pkgs.update(value.get(test_path.name) or [])

        # 3) 其它类型忽略/报错都行，这里选择忽略
        else:
            pass

    return specific_pkgs, common_pkgs



def pip_install(pkgs):
    if not pkgs:
        return

    print("Installing deps:")
    for p in pkgs:
        print("  -", p)

    cmd = [
        sys.executable,
        "-m", "pip", "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
    ]
    pkgs = [normalize_pkg_spec(p) for p in pkgs]
    cmd.extend(pkgs)

    subprocess.check_call(cmd, shell=False)


def uv_install(pkgs):
    if not pkgs:
        return

    print("Installing deps with uv:")
    for p in pkgs:
        print("  -", p)

    cmd = [
        "uv", "pip", "install", "--no-cache",
    ]

    pkgs = [normalize_pkg_spec(p) for p in pkgs]
    cmd.extend(pkgs)

    subprocess.check_call(cmd, shell=False)

if __name__ == "__main__":
    raw_name = sys.argv[1].removeprefix("tests/").removesuffix(".py")
    test_path = resolve_test_path(raw_name)

    with open(os.path.join(base_dir, "deps.yaml")) as f:
        deps = yaml.safe_load(f)

    specific_pkgs, common_pkgs = collect_pkgs(test_path, deps)

    uv_install(sorted(specific_pkgs))

    uv_install(sorted(common_pkgs))
