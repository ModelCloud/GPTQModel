# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from shutil import rmtree

from setuptools import find_namespace_packages, find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

CUTLASS_VERSION = "3.5.0"
CUTLASS_RELEASE_URL = f"https://github.com/NVIDIA/cutlass/archive/refs/tags/v{CUTLASS_VERSION}.tar.gz"


def _ensure_cutlass_source() -> Path:
    deps_dir = Path("build") / "_deps"
    deps_dir.mkdir(parents=True, exist_ok=True)

    cutlass_root = deps_dir / f"cutlass-v{CUTLASS_VERSION}"
    marker = cutlass_root / ".gptqmodel_complete"
    if marker.exists():
        return cutlass_root.resolve()

    archive_path = deps_dir / f"cutlass-v{CUTLASS_VERSION}.tar.gz"
    if not archive_path.exists():
        print(f"Downloading CUTLASS v{CUTLASS_VERSION} ...")
        with urllib.request.urlopen(CUTLASS_RELEASE_URL) as response:
            data = response.read()
        archive_path.write_bytes(data)

    if cutlass_root.exists():
        rmtree(cutlass_root)

    with tarfile.open(archive_path, "r:gz") as tar:
        extract_kwargs = {"path": deps_dir}
        if sys.version_info >= (3, 12):
            extract_kwargs["filter"] = "data"
        tar.extractall(**extract_kwargs)

    extracted_dir = deps_dir / f"cutlass-{CUTLASS_VERSION}"
    if not extracted_dir.exists():
        raise RuntimeError("Failed to extract CUTLASS archive")

    extracted_dir.rename(cutlass_root)
    marker.touch()
    return cutlass_root.resolve()


# ---------------------------
# Helpers (no torch required)
# ---------------------------

def _read_env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default


def _probe_cmd(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def _bool_env(name, default=False):
    v = _read_env(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def _detect_rocm_version():
    v = _read_env("ROCM_VERSION")
    if v:
        return v
    hip = _probe_cmd(["hipcc", "--version"])
    if hip:
        import re
        m = re.search(r"\b([0-9]+\.[0-9]+)\b", hip)
        if m:
            return m.group(1)
    try:
        p = Path("/opt/rocm/.info/version")
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None


def _detect_cuda_arch_list():
    """Return TORCH_CUDA_ARCH_LIST style string for the *installed* GPUs only.
    Priority:
      1) CUDA_ARCH_LIST env override (verbatim)
      2) nvidia-smi compute_cap (actual devices)
    """
    # 1) explicit override
    env_arch = _read_env("CUDA_ARCH_LIST")
    if env_arch:
        return env_arch

    # 2) actual devices present
    smi_out = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi_out:
        caps = []
        for line in smi_out.splitlines():
            cap = line.strip()
            if not cap:
                continue
            # normalize like '8.0'
            try:
                major, minor = cap.split(".", 1)
                caps.append(f"{int(major)}.{int(minor)}")
            except Exception:
                # some drivers return just '8' -> treat as '8.0'
                if cap.isdigit():
                    caps.append(f"{cap}.0")
        caps = sorted(set(caps), key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])))
        if caps:
            # PyTorch prefers ';' separators
            return ";".join(caps)

    # 3) conservative default for modern datacenter GPUs (A100 et al.)
    raise Exception("Could not get compute capability from nvidia-smi. Please check nvidia-utils package is installed.")


def _parse_arch_list(s: str):
    # Accept semicolons, commas, and any whitespace as separators.
    # Keep tokens like "8.0", "8.0+PTX" intact (we’ll strip suffixes later).
    return [tok for tok in re.split(r"[;\s,]+", s) if tok.strip()]


def _has_cuda_v8_from_arch_list(arch_list):
    try:
        vals = []
        for a in arch_list:
            # Handle things like "8.0+PTX"
            base = a.split("+", 1)[0]
            vals.append(float(base))
        return any(v >= 8.0 for v in vals)
    except Exception:
        return False


def _detect_cxx11_abi():
    v = _read_env("CXX11_ABI")
    if v in ("0", "1"):
        return int(v)
    return 1


def _torch_version_for_release():
    # No torch import; allow env override
    v = _read_env("TORCH_VERSION")
    if v:
        parts = v.split(".")
        return ".".join(parts[:2])
    else:
        raise Exception("TORCH_VERSION not passed for wheel generation.")
    return None


def _is_rocm_available():
    return _detect_rocm_version() is not None


# If you already have _probe_cmd elsewhere, you can delete this copy.
def _probe_cmd(args, timeout=6):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception:
        return ""


def _first_token_line(s: str) -> str | None:
    for line in (s or "").splitlines():
        t = line.strip()
        if t:
            return t
    return None


def _detect_torch_version() -> str | None:
    # 1) uv pip show torch
    out = _probe_cmd(["uv", "pip", "show", "torch"])
    if out:
        m = re.search(r"^Version:\s*([^\s]+)\s*$", out, flags=re.MULTILINE)
        if m:
            return m.group(1)

    # 2) pip show torch (both 'pip' and 'python -m pip')
    for cmd in (["pip", "show", "torch"], [sys.executable, "-m", "pip", "show", "torch"]):
        out = _probe_cmd(cmd)
        if out:
            m = re.search(r"^Version:\s*([^\s]+)\s*$", out, flags=re.MULTILINE)
            if m:
                return m.group(1)

    # 3) conda list torch
    out = _probe_cmd(["conda", "list", "torch"])
    if out:
        # Typical line starts with: torch  2.4.1  ...
        for line in out.splitlines():
            if line.strip().startswith("torch"):
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 2 and re.match(r"^\d+\.\d+(\.\d+)?", parts[1]):
                    return parts[1]

    # 4) Fallback: importlib.metadata (does not import torch package module)
    try:
        import importlib.metadata as im  # py3.8+
        version = im.version("torch")
        if not version:
            raise Exception("torch not found")
    except Exception:
        raise Exception("Unable to detect torch version via uv/pip/conda/importlib. Please install torch >= 2.7.1")


def _major_minor(v: str) -> str:
    if v:
        parts = v.split(".")
        return ".".join(parts[:2]) if parts else v
    return v


def _version_geq(version: str | None, major: int, minor: int = 0) -> bool:
    if not version:
        return False
    try:
        parts = re.split(r"[._-]", version)
        ver_major = int(parts[0]) if parts else 0
        ver_minor = int(parts[1]) if len(parts) > 1 else 0
        return (ver_major, ver_minor) >= (major, minor)
    except Exception:
        return False


def _nvcc_release_version() -> str | None:
    # Search for nvcc in common locations before giving up.
    candidates: list[str] = []
    nvcc_env = _read_env("NVCC")
    if nvcc_env:
        candidates.append(nvcc_env)

    cuda_home = _read_env("CUDA_HOME")
    cuda_path = _read_env("CUDA_PATH")

    candidates.extend(
        [
            "nvcc",
            str(Path(cuda_home).joinpath("bin", "nvcc")) if cuda_home else None,
            str(Path(cuda_path).joinpath("bin", "nvcc")) if cuda_path else None,
            "/usr/local/cuda/bin/nvcc",
        ]
    )

    seen = set()
    for cmd in candidates:
        if not cmd or cmd in seen:
            continue
        seen.add(cmd)
        out = _probe_cmd([cmd, "--version"])
        if not out:
            continue
        match = re.search(r"release\s+(\d+)\.(\d+)", out)
        if match:
            return f"{match.group(1)}.{match.group(2)}"

    print(
        "NVCC not found (checked PATH, $CUDA_HOME/bin, $CUDA_PATH/bin, /usr/local/cuda/bin). "
        "For Ubuntu, run `sudo update-alternatives --config cuda` to fix path for already installed Cuda."
    )
    return None


def _detect_cuda_version() -> str | None:
    # Priority: env → nvidia-smi → nvcc
    v = os.environ.get("CUDA_VERSION")
    if v and v.strip():
        return v.strip()

    # nvcc --version (parse 'release X.Y')
    return _nvcc_release_version()


def _detect_nvcc_version() -> str | None:
    return _nvcc_release_version()


def get_version_tag() -> str:
    # TODO FIX ME: cpu wheels don't have torch version tags?
    if BUILD_CUDA_EXT != "1":
        return "cpu"

    # TODO FIX ME: rocm wheels don't have torch version tags?
    if ROCM_VERSION:
        return f"rocm{ROCM_VERSION}"

    if not CUDA_VERSION:
        raise Exception("Trying to compile GPTQModel for CUDA/ROCm, but no cuda or rocm version was detected.")

    torch_suffix = f"torch{_major_minor(TORCH_VERSION)}"

    CUDA_VERSION_COMPACT = "".join(CUDA_VERSION.split("."))
    base = f"cu{CUDA_VERSION_COMPACT[:3]}"
    return f"{base}{torch_suffix}"


# ---------------------------
# Env and versioning
# ---------------------------

TORCH_VERSION = _read_env("TORCH_VERSION")
RELEASE_MODE = _read_env("RELEASE_MODE")
CUDA_VERSION = _read_env("CUDA_VERSION")
ROCM_VERSION = _read_env("ROCM_VERSION")
TORCH_CUDA_ARCH_LIST = _read_env("TORCH_CUDA_ARCH_LIST")
NVCC_VERSION = _read_env("NVCC_VERSION")

# respect user env then detect
if not TORCH_VERSION:
    TORCH_VERSION = _detect_torch_version()
if not CUDA_VERSION:
    CUDA_VERSION = _detect_cuda_version()
if not ROCM_VERSION:
    ROCM_VERSION = _detect_rocm_version()
if not NVCC_VERSION:
    NVCC_VERSION = _detect_nvcc_version()

SKIP_ROCM_VERSION_CHECK = _read_env("SKIP_ROCM_VERSION_CHECK")
FORCE_BUILD = _bool_env("GPTQMODEL_FORCE_BUILD", False)

# BUILD_CUDA_EXT:
# - If user sets explicitly, respect it.
# - Otherwise auto: enable only if CUDA or ROCm detected.
BUILD_CUDA_EXT = _read_env("BUILD_CUDA_EXT")
if BUILD_CUDA_EXT is None:
    BUILD_CUDA_EXT = "1" if (CUDA_VERSION or ROCM_VERSION) else "0"

if ROCM_VERSION and not SKIP_ROCM_VERSION_CHECK:
    try:
        if float(ROCM_VERSION) < 6.2:
            sys.exit(
                "GPTQModel's compatibility with ROCm < 6.2 has not been verified. "
                "Set SKIP_ROCM_VERSION_CHECK=1 to proceed."
            )
    except Exception:
        pass

# Handle CUDA_ARCH_LIST (public) and set TORCH_CUDA_ARCH_LIST for build toolchains
CUDA_ARCH_LIST = _detect_cuda_arch_list() if (BUILD_CUDA_EXT == "1" and not ROCM_VERSION) else None

if not TORCH_CUDA_ARCH_LIST and CUDA_ARCH_LIST:
    archs = _parse_arch_list(CUDA_ARCH_LIST)
    kept = []
    for arch in archs:
        try:
            base = arch.split("+", 1)[0]
            if float(base) >= 6.0:
                kept.append(arch)
            else:
                print(f"we do not support this compute arch: {arch}, skipped.")
        except Exception:
            kept.append(arch)

    # Use semicolons for TORCH_CUDA_ARCH_LIST (PyTorch likes this),
    TORCH_CUDA_ARCH_LIST = ";".join(kept)
    os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST

    print(f"CUDA_ARCH_LIST: {CUDA_ARCH_LIST}")
    print(f"TORCH_CUDA_ARCH_LIST: {TORCH_CUDA_ARCH_LIST}")

version_vars = {}
exec("exec(open('gptqmodel/version.py').read()); version=__version__", {}, version_vars)
gptqmodel_version = version_vars["version"]

# -----------------------------
# Prebuilt wheel download config
# -----------------------------
# Default template (GitHub Releases), can be overridden via env.
DEFAULT_WHEEL_URL_TEMPLATE = "https://github.com/ModelCloud/GPTQModel/releases/download/{tag_name}/{wheel_name}"
WHEEL_URL_TEMPLATE = os.environ.get("GPTQMODEL_WHEEL_URL_TEMPLATE")
WHEEL_BASE_URL = os.environ.get("GPTQMODEL_WHEEL_BASE_URL")
WHEEL_TAG = os.environ.get("GPTQMODEL_WHEEL_TAG")  # Optional override of release tag


def _resolve_wheel_url(tag_name: str, wheel_name: str) -> str:
    """
    Build the final wheel URL based on:
      1) GPTQMODEL_WHEEL_URL_TEMPLATE (highest priority)
      2) GPTQMODEL_WHEEL_BASE_URL (append /{wheel_name})
      3) DEFAULT_WHEEL_URL_TEMPLATE (GitHub Releases)
    """
    # Highest priority: explicit template
    if WHEEL_URL_TEMPLATE:
        tmpl = WHEEL_URL_TEMPLATE
        # If {wheel_name} or {tag_name} not present, treat as base and append name.
        if ("{wheel_name}" in tmpl) or ("{tag_name}" in tmpl):
            return tmpl.format(tag_name=tag_name, wheel_name=wheel_name)
        # Otherwise, join as base
        if tmpl.endswith("/"):
            return tmpl + wheel_name
        return tmpl + "/" + wheel_name

    # Next priority: base URL
    if WHEEL_BASE_URL:
        base = WHEEL_BASE_URL
        if base.endswith("/"):
            return base + wheel_name
        return base + "/" + wheel_name

    # Fallback: default GitHub template
    return DEFAULT_WHEEL_URL_TEMPLATE.format(tag_name=tag_name, wheel_name=wheel_name)


# Decide HAS_CUDA_V8 / HAS_CUDA_V9 without torch
HAS_CUDA_V8 = False
HAS_CUDA_V9 = False
if CUDA_ARCH_LIST:
    arch_list = _parse_arch_list(CUDA_ARCH_LIST)
    try:
        caps = [float(tok.split("+", 1)[0]) for tok in arch_list]
    except Exception:
        caps = []
    if not ROCM_VERSION:
        HAS_CUDA_V8 = any(cap >= 8.0 for cap in caps)
        HAS_CUDA_V9 = any(cap >= 9.0 for cap in caps)
else:
    smi = _probe_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if smi:
        try:
            caps = [float(x.strip()) for x in smi.splitlines() if x.strip()]
            HAS_CUDA_V8 = any(cap >= 8.0 for cap in caps)
            HAS_CUDA_V9 = any(cap >= 9.0 for cap in caps)
        except Exception:
            HAS_CUDA_V8 = False
            HAS_CUDA_V9 = False

if RELEASE_MODE == "1":
    gptqmodel_version = f"{gptqmodel_version}+{get_version_tag()}"

include_dirs = ["gptqmodel_ext"]

extensions = []
additional_setup_kwargs = {}


# ---------------------------
# Build CUDA/ROCm extensions (only when enabled)
# ---------------------------
# -----------------------------
# Per-extension build toggles
# -----------------------------
def _env_enabled(val: str) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() not in ("0", "false", "off", "no")


def _env_enabled_any(names, default="1") -> bool:
    for n in names:
        if n in os.environ:
            return _env_enabled(os.environ.get(n))
    return _env_enabled(default)


BUILD_MARLIN = _env_enabled_any(os.environ.get("GPTQMODEL_BUILD_MARLIN", "1"))
BUILD_MACHETE = _env_enabled(os.environ.get("GPTQMODEL_BUILD_MACHETE", "0"))
BUILD_EXLLAMA_V2 = _env_enabled(os.environ.get("GPTQMODEL_BUILD_EXLLAMA_V2", "1"))
BUILD_QQQ = _env_enabled(os.environ.get("GPTQMODEL_BUILD_QQQ", "1"))
BUILD_AWQ = _env_enabled(os.environ.get("GPTQMODEL_BUILD_AWQ", "1"))

# Optional kernels and not build by default. Enable compile with env flags
BUILD_EORA = _env_enabled(os.environ.get("GPTQMODEL_BUILD_EORA", "0"))
BUILD_EXLLAMA_V1 = _env_enabled(os.environ.get("GPTQMODEL_BUILD_EXLLAMA_V1", "0"))

if BUILD_CUDA_EXT == "1":
    # Import torch's cpp_extension only if we're truly building GPU extensions
    try:

        from torch.utils import cpp_extension as cpp_ext  # type: ignore
    except Exception:
        if FORCE_BUILD:
            sys.exit(
                "FORCE_BUILD is set but PyTorch C++ extension headers are unavailable. "
                "Install torch build deps first (see https://pytorch.org/) or unset GPTQMODEL_FORCE_BUILD."
            )
        # If we can't import cpp_extension, fall back to prebuilt wheel path
        cpp_ext = None

    if cpp_ext is not None:
        # Limit compile parallelism to avoid overwhelming nvcc/cicc invocations.
        # Respect pre-set MAX_JOBS, otherwise fall back to CPU count minus two (min 1).
        cpu_count = os.cpu_count() or 1
        default_max_jobs = max(1, cpu_count - 2)
        max_jobs_raw = os.environ.get("MAX_JOBS")
        if max_jobs_raw is None or max_jobs_raw.strip() == "":
            effective_max_jobs = default_max_jobs
            print(f"MAX_JOBS not set; defaulting to {effective_max_jobs} concurrent CUDA compilations.")
        else:
            try:
                parsed_jobs = int(max_jobs_raw)
            except ValueError:
                effective_max_jobs = default_max_jobs
                print(f"Ignoring invalid MAX_JOBS={max_jobs_raw!r}; using {effective_max_jobs}.")
            else:
                if parsed_jobs <= 0:
                    effective_max_jobs = default_max_jobs
                    print(f"MAX_JOBS={parsed_jobs} is non-positive; using {effective_max_jobs}.")
                else:
                    effective_max_jobs = parsed_jobs

        os.environ["MAX_JOBS"] = str(effective_max_jobs)
        os.environ["NINJA_NUM_JOBS"] = str(effective_max_jobs)
        print(f"Using MAX_JOBS={effective_max_jobs} to cap concurrent CUDA compilations.")

        nvcc_threads = 1
        os.environ["NVCC_THREADS"] = str(nvcc_threads)
        print(f"Using NVCC_THREADS={nvcc_threads} for per-invocation NVCC concurrency.")

        # Optional conda CUDA runtime headers
        # conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
        # if os.path.isdir(conda_cuda_include_dir):
        #     include_dirs.append(conda_cuda_include_dir)
        #     print(f"appending conda cuda include dir {conda_cuda_include_dir}")

        extra_link_args = []
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17", "-DENABLE_BF16"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ],
        }

        cutlass_include_paths = []
        if BUILD_MACHETE:
            cutlass_root = _ensure_cutlass_source()
            cutlass_include_paths = [
                Path("gptqmodel_ext/cutlass_extensions").resolve(),
                cutlass_root / "include",
                cutlass_root / "examples/common/include",
                cutlass_root / "tools/library/include",
            ]
            if "GPTQMODEL_CUTLASS_DIR" not in os.environ:
                os.environ["GPTQMODEL_CUTLASS_DIR"] = str(cutlass_root)
            cutlass_include_flags = [f"-I{path}" for path in cutlass_include_paths]
            extra_compile_args["cxx"] += cutlass_include_flags
            extra_compile_args["nvcc"] += cutlass_include_flags

        # Windows/OpenMP note: adjust flags as needed for MSVC if you add native Windows wheels
        if sys.platform == "win32":
            extra_compile_args["cxx"] = ["/O2", "/std:c++17", "/openmp", "/DNDEBUG", "/DENABLE_BF16"]

        CXX11_ABI = _detect_cxx11_abi()
        extra_compile_args["cxx"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]
        extra_compile_args["nvcc"] += [f"-D_GLIBCXX_USE_CXX11_ABI={CXX11_ABI}"]

        if not ROCM_VERSION:
            # if _version_geq(NVCC_VERSION, 13, 0):
            #     extra_compile_args["nvcc"].append("--device-entity-has-hidden-visibility=false")
            nvcc_extra_flags = [
                "--threads", str(nvcc_threads),  # NVCC parallelism
                "--optimize=3",  # alias for -O3
                # "-rdc=true",  # enable relocatable device code, required for future cuda > 13.x <-- TODO FIX ME broken loading
                # "-dlto",      # compile and link <-- TODO FIX ME
                # Print register/shared-memory usage per kernel (debug aid, no perf effect)
                # Ensure PTXAS uses maximum optimization
                # Cache global loads in both L1 and L2 (better for memory-bound kernels)
                "-Xptxas", "-v,-O3,-dlcm=ca",
                "-lineinfo",  # keep source line info for profiling
                # "--resource-usage",  # show per-kernel register/SMEM usage
                "-Xfatbin", "-compress-all",  # compress fatbin
                # "--expt-relaxed-constexpr",  # relaxed constexpr rules <-- not used
                # "--expt-extended-lambda",  # allow device lambdas <-- not used
                "-diag-suppress=179,39,177",  # silence some template warnings
            ]
            if _version_geq(NVCC_VERSION, 12, 8):
                # Allow instantiations of __global__ templates to live in different TUs; only supported in newer NVCC.
                nvcc_extra_flags.insert(0, "-static-global-template-stub=false")
            extra_compile_args["nvcc"] += nvcc_extra_flags
        else:
            # hipify CUDA-like flags
            def _hipify_compile_flags(flags):
                modified_flags = []
                for flag in flags:
                    if flag.startswith("-") and "CUDA" in flag and not flag.startswith("-I"):
                        parts = flag.split("=", 1)
                        if len(parts) == 2:
                            flag_part, value_part = parts
                            modified_flag_part = flag_part.replace("CUDA", "HIP", 1)
                            modified_flags.append(f"{modified_flag_part}={value_part}")
                        else:
                            modified_flags.append(flag.replace("CUDA", "HIP", 1))
                    else:
                        modified_flags.append(flag)
                return modified_flags


            extra_compile_args["nvcc"] = _hipify_compile_flags(extra_compile_args["nvcc"])

        # Extensions (gate marlin/qqq/eora/exllamav2 on CUDA sm_80+ and non-ROCm)
        if sys.platform != "win32":
            if not ROCM_VERSION and HAS_CUDA_V8:
                if BUILD_MARLIN:
                    marlin_kernel_dir = Path("gptqmodel_ext/marlin")
                    marlin_kernel_files = sorted(marlin_kernel_dir.glob("kernel_*.cu"))

                    if not marlin_kernel_files:
                        generator_script = marlin_kernel_dir / "generate_kernels.py"
                        if generator_script.exists():
                            print("Regenerating marlin template instantiations for parallel compilation...")
                            subprocess.check_call([sys.executable, str(generator_script)])
                            marlin_kernel_files = sorted(marlin_kernel_dir.glob("kernel_*.cu"))

                    if not marlin_kernel_files:
                        raise RuntimeError(
                            "No generated marlin kernel templates detected. Run generate_kernels.py before building."
                        )

                    marlin_template_kernel_srcs = [str(path) for path in marlin_kernel_files]
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_marlin_kernels",
                            [
                                "gptqmodel_ext/marlin/marlin_cuda.cpp",
                                "gptqmodel_ext/marlin/gptq_marlin.cu",
                                "gptqmodel_ext/marlin/gptq_marlin_repack.cu",
                                "gptqmodel_ext/marlin/awq_marlin_repack.cu",
                            ] + marlin_template_kernel_srcs,
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]

                if BUILD_MACHETE and HAS_CUDA_V9 and _version_geq(NVCC_VERSION, 12, 0):
                    try:
                        result = subprocess.run(
                            [sys.executable, "gptqmodel_ext/machete/generate.py"],
                            check=True,
                            text=True,
                            capture_output=True
                        )
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(
                            f"Error generating machete kernel templates:\n"
                            f"Return code: {e.returncode}\n"
                            f"Stderr: {e.stderr}\n"
                            f"Stdout: {e.stdout}"
                        )
                    machete_dir = Path("gptqmodel_ext/machete")
                    machete_generated_dir = machete_dir / "generated"

                    machete_sources = [str(machete_dir / "machete_pytorch.cu")]
                    machete_generated_sources = sorted(machete_generated_dir.glob("*.cu"))

                    if not machete_generated_sources:
                        raise RuntimeError(
                            "No generated machete kernel templates detected. Run gptqmodel_ext/machete/generate.py"
                            " with CUTLASS checkout before building."
                        )

                    machete_sources += [str(path) for path in machete_generated_sources]

                    machete_include_dirs = [str(Path("gptqmodel_ext").resolve())] + [str(path) for path in cutlass_include_paths]

                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_machete_kernels",
                            machete_sources,
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                            include_dirs=machete_include_dirs,
                        )
                    ]

                if BUILD_QQQ:
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_qqq_kernels",
                            [
                                "gptqmodel_ext/qqq/qqq.cpp",
                                "gptqmodel_ext/qqq/qqq_gemm.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]

                if BUILD_EORA:
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_exllama_eora",
                            [
                                "gptqmodel_ext/exllama_eora/eora/q_gemm.cu",
                                "gptqmodel_ext/exllama_eora/eora/pybind.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]
                if BUILD_EXLLAMA_V2:
                    extensions += [
                        cpp_ext.CUDAExtension(
                            "gptqmodel_exllamav2_kernels",
                            [
                                "gptqmodel_ext/exllamav2/ext.cpp",
                                "gptqmodel_ext/exllamav2/cuda/q_matrix.cu",
                                "gptqmodel_ext/exllamav2/cuda/q_gemm.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        )
                    ]

            # both CUDA and ROCm compatible
            if BUILD_EXLLAMA_V1:
                extensions += [
                    cpp_ext.CUDAExtension(
                        "gptqmodel_exllama_kernels",
                        [
                            "gptqmodel_ext/exllama/exllama_ext.cpp",
                            "gptqmodel_ext/exllama/cuda_buffers.cu",
                            "gptqmodel_ext/exllama/cuda_func/column_remap.cu",
                            "gptqmodel_ext/exllama/cuda_func/q4_matmul.cu",
                            "gptqmodel_ext/exllama/cuda_func/q4_matrix.cu",
                        ],
                        extra_link_args=extra_link_args,
                        extra_compile_args=extra_compile_args,
                    )
                ]

            if BUILD_AWQ:
                if ROCM_VERSION:
                    print("Skipping AWQ kernels on ROCm: inline PTX is CUDA-only.")
                else:
                    extensions += [
                        # contain un-hipifiable inline PTX
                        cpp_ext.CUDAExtension(
                            "gptqmodel_awq_kernels",
                            [
                                "gptqmodel_ext/awq/pybind_awq.cpp",
                                "gptqmodel_ext/awq/quantization/gemm_cuda_gen.cu",
                                "gptqmodel_ext/awq/quantization/gemv_cuda.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        ),
                        # TODO only compatible with ampere?
                        # arch_flags = get_compute_capabilities({80, 86, 89, 90})
                        # extra_compile_args_v2 = get_extra_compile_args(arch_flags, generator_flags)
                        cpp_ext.CUDAExtension(
                            "gptqmodel_awq_v2_kernels",
                            [
                                "gptqmodel_ext/awq/pybind_awq_v2.cpp",
                                "gptqmodel_ext/awq/quantization_new/gemv/gemv_cuda.cu",
                                "gptqmodel_ext/awq/quantization_new/gemm/gemm_cuda.cu",
                            ],
                            extra_link_args=extra_link_args,
                            extra_compile_args=extra_compile_args,
                        ),
                    ]

        # Ensure machete kernels are compiled before other extensions
        machete_exts = [ext for ext in extensions if getattr(ext, "name", "") == "gptqmodel_machete_kernels"]
        if machete_exts:
            other_exts = [ext for ext in extensions if getattr(ext, "name", "") != "gptqmodel_machete_kernels"]
            extensions[:] = machete_exts + other_exts

        additional_setup_kwargs = {
            "ext_modules": extensions,
            "cmdclass": {"build_ext": cpp_ext.BuildExtension},
        }

        # additional_setup_kwargs = {
        #     "ext_modules": extensions,
        #     # "include_package_data": True,
        #     # "package_data": {"": ["build/lib/*.so"]},
        #     "cmdclass": {"build_ext": cpp_ext.BuildExtension.with_options(
        #         use_ninja=True,
        #         no_python_abi_suffix=True,
        #         build_temp="build/temp",
        #         # build_lib="build/lib", TODO FIX ME why package_data doesn't work..
        #         clean_first=False  # keep intermediates for reuse
        #     )},
        # }


# ---------------------------
# Cached wheel fetcher
# ---------------------------

class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        # No implicit torch checks; allow explicit override via env
        xpu_avail = _bool_env("XPU_AVAILABLE", False)
        if FORCE_BUILD or xpu_avail:
            return super().run()

        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

        wheel_filename = f"gptqmodel-{gptqmodel_version}+{get_version_tag()}-{python_version}-{python_version}-linux_x86_64.whl"

        # Allow tag override via env; default to "v{gptqmodel_version}"
        tag_name = WHEEL_TAG if WHEEL_TAG else f"v{gptqmodel_version}"
        wheel_url = _resolve_wheel_url(tag_name=tag_name, wheel_name=wheel_filename)

        print(f"Resolved wheel URL: {wheel_url}\nwheel name={wheel_filename}")

        try:
            import urllib.request as req
            req.urlretrieve(wheel_url, wheel_filename)

            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = (f"gptqmodel-{gptqmodel_version}-{impl_tag}-{abi_tag}-{plat_tag}")
            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except BaseException:
            print(f"Precompiled wheel not found at: {wheel_url}. Building from source...")
            super().run()


# ---------------------------
# setup()
# ---------------------------
print(f"CUDA {CUDA_ARCH_LIST}")
print(f"HAS_CUDA_V8 {HAS_CUDA_V8}")
print(f"HAS_CUDA_V9 {HAS_CUDA_V9}")
print(f"SETUP_KWARGS {additional_setup_kwargs}")
print(f"gptqmodel_version={gptqmodel_version}")

_namespace_packages = find_namespace_packages(include=["gptqmodel_ext.*"])
_packages = find_packages()
for _pkg in _namespace_packages:
    if _pkg not in _packages:
        _packages.append(_pkg)

setup(
    version=gptqmodel_version,
    packages=_packages,
    include_package_data=True,
    extras_require={
        "test": ["pytest>=8.2.2", "parameterized"],
        "quality": ["ruff==0.13.0", "isort==6.0.1"],
        "vllm": ["vllm>=0.8.5", "flashinfer-python>=0.2.1"],
        "sglang": ["sglang[srt]>=0.4.6", "flashinfer-python>=0.2.1"],
        "bitblas": ["bitblas==0.0.1-dev13"],
        "hf": ["optimum>=1.21.2"],
        "eval": ["lm_eval>=0.4.7", "evalplus>=0.3.1"],
        "triton": ["triton>=3.4.0"],
        "openai": ["uvicorn", "fastapi", "pydantic"],
        "mlx": ["mlx_lm>=0.28.2"],
    },
    include_dirs=include_dirs,
    cmdclass=(
        {"bdist_wheel": CachedWheelsCommand, "build_ext": additional_setup_kwargs.get("cmdclass", {}).get("build_ext")}
        if (BUILD_CUDA_EXT == "1" and additional_setup_kwargs)
        else {"bdist_wheel": CachedWheelsCommand}
    ),
    ext_modules=additional_setup_kwargs.get("ext_modules", []),
)
