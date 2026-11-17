# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""HIP JIT builds must not hipify into the shipped source tree (#4072, #4092).

torch's ROCm loader writes ``<name>_hip.{hip,cpp}`` next to each source it
processes and rewrites ``cuda`` to ``hip`` in the output path. A CUDA-only
extension therefore overwrites tracked hand-written HIP sources when it is
loaded on ROCm. These tests pin the staging contract that keeps every generated
file under the writable build cache, for every extension rather than the five
that opted in for #4072.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from gptqmodel.utils import cpp
from gptqmodel.utils.cpp import TorchOpsJitExtension


def _extension(tmp_path: Path, *, stage: bool | None = True) -> TorchOpsJitExtension:
    source_root = tmp_path / "gptqmodel_ext" / "demo"
    source_root.mkdir(parents=True)
    (source_root / "demo.cu").write_text("#include \"demo.cuh\"\n", encoding="utf-8")
    (source_root / "demo.cuh").write_text("// header\n", encoding="utf-8")
    kwargs = {} if stage is None else {"stage_hip_sources": stage}
    return TorchOpsJitExtension(
        name="demo_ops",
        namespace="demo",
        required_ops=("run",),
        sources=lambda: [str(source_root / "demo.cu")],
        build_root_env=None,
        default_build_root=tmp_path / "build",
        display_name="Demo",
        extra_include_paths=lambda: [str(source_root)],
        requires_cuda=True,
        **kwargs,
    )


def test_hip_sources_are_staged_under_the_build_root(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp, "IS_ROCM", True)
    extension = _extension(tmp_path, stage=True)
    source_root = tmp_path / "gptqmodel_ext" / "demo"

    sources, includes = extension._resolved_sources_and_include_paths()

    staged_root = tmp_path / "build" / "_src"
    assert sources == [str(staged_root / "demo.cu")]
    assert includes[0] == str(staged_root)
    assert (staged_root / "demo.cu").read_text(encoding="utf-8") == "#include \"demo.cuh\"\n"
    assert (staged_root / "demo.cuh").read_text(encoding="utf-8") == "// header\n"
    # The shipped tree is copied, never generated into.
    assert sorted(path.name for path in source_root.iterdir()) == ["demo.cu", "demo.cuh"]


def test_staging_is_disabled_off_rocm(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp, "IS_ROCM", False)
    extension = _extension(tmp_path, stage=True)
    source_root = tmp_path / "gptqmodel_ext" / "demo"

    sources, includes = extension._resolved_sources_and_include_paths()

    assert sources == [str(source_root / "demo.cu")]
    assert includes[0] == str(source_root)
    assert not (tmp_path / "build" / "_src").exists()


@pytest.mark.parametrize(
    "module_name",
    [
        "gptqmodel.quantization.gptq_cuda",
        "gptqmodel.utils.allspark_hip",
        "gptqmodel.utils.hadamard",
        "gptqmodel.utils.paro",
        "gptqmodel.exllamav3.ext",
    ],
)
def test_every_hip_loader_stages_its_sources(module_name):
    import importlib

    module = importlib.import_module(module_name)
    extensions = [
        value
        for value in vars(module).values()
        if isinstance(value, TorchOpsJitExtension)
    ]

    assert extensions, f"{module_name} has no TorchOpsJitExtension"
    assert all(extension.stage_hip_sources for extension in extensions)


def test_staging_is_the_default_for_every_extension(tmp_path):
    extension = _extension(tmp_path, stage=None)

    assert extension.stage_hip_sources is True


def _fake_hipify_load(monkeypatch, record):
    """Stand-in for torch's HIP loader: it writes hipified files next to inputs."""

    def fake_load(*, sources, build_directory, extra_include_paths=(), **kwargs):
        record["sources"] = [str(source) for source in sources]
        record["includes"] = [str(path) for path in extra_include_paths]
        for source in sources:
            Path(source).with_suffix(".hip").write_text("// hipified\n", encoding="utf-8")
        for root in extra_include_paths:
            for header in Path(root).rglob("*.h"):
                header.with_suffix(".hip").write_text("// hipified\n", encoding="utf-8")
        raise RuntimeError("stop after hipify")

    monkeypatch.setattr(cpp, "load", fake_load)


def test_cuda_only_build_on_rocm_never_writes_into_the_checkout(monkeypatch, tmp_path):
    """A CUDA-only extension must hipify in the cache, not over tracked sources."""

    checkout = tmp_path / "checkout"
    source_root = checkout / "gptqmodel_ext" / "tt_blockfloat_cuda"
    source_root.mkdir(parents=True)
    (source_root / "tt_blockfloat_cuda.cu").write_text("// cuda kernel\n", encoding="utf-8")
    (source_root / "tt_blockfloat_cuda.h").write_text("// header\n", encoding="utf-8")
    before = {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob("*") if path.is_file()}

    monkeypatch.setattr(cpp, "IS_ROCM", True)
    monkeypatch.setattr(cpp, "_source_checkout_root", lambda: checkout)
    record: dict[str, list[str]] = {}
    _fake_hipify_load(monkeypatch, record)
    extension = TorchOpsJitExtension(
        name="tt_blockfloat_cuda_ops",
        namespace="demo",
        required_ops=("run",),
        sources=lambda: [str(source_root / "tt_blockfloat_cuda.cu")],
        build_root_env=None,
        default_build_root=tmp_path / "build",
        display_name="TT blockfloat CUDA",
        extra_include_paths=lambda: [str(source_root)],
        requires_cuda=False,
    )

    # load() reports failure after the stubbed hipify step; the checkout must
    # still be byte-identical and contain no generated file.
    assert extension.load() is False

    after = {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob("*") if path.is_file()}
    assert after == before
    assert record["sources"]
    assert all(str(tmp_path / "build") in source for source in record["sources"])


def test_opted_out_staging_refuses_to_mutate_the_checkout(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    source_root = checkout / "gptqmodel_ext" / "demo"
    source_root.mkdir(parents=True)
    (source_root / "demo.cu").write_text("// kernel\n", encoding="utf-8")

    monkeypatch.setattr(cpp, "IS_ROCM", True)
    monkeypatch.setattr(cpp, "_source_checkout_root", lambda: checkout)
    extension = TorchOpsJitExtension(
        name="demo_ops",
        namespace="demo",
        required_ops=("run",),
        sources=lambda: [str(source_root / "demo.cu")],
        build_root_env=None,
        default_build_root=tmp_path / "build",
        display_name="Demo",
        requires_cuda=False,
        stage_hip_sources=False,
    )

    with pytest.raises(RuntimeError, match="refusing to hipify"):
        extension._resolved_sources_and_include_paths()


def test_include_roots_inside_the_checkout_are_staged(monkeypatch, tmp_path):
    """Machete hipifies headers under its include roots, which are tracked."""

    checkout = tmp_path / "checkout"
    include_root = checkout / "gptqmodel_ext"
    (include_root / "cutlass_extensions").mkdir(parents=True)
    (include_root / "cutlass_extensions" / "epilogue.h").write_text("// header\n", encoding="utf-8")
    repo_source = include_root / "machete" / "machete_pytorch.cu"
    repo_source.parent.mkdir(parents=True)
    repo_source.write_text("// source\n", encoding="utf-8")
    cached_sources = tmp_path / "cache" / "machete"
    cached_sources.mkdir(parents=True)
    (cached_sources / "generated.cu").write_text("// generated\n", encoding="utf-8")
    before = {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob("*") if path.is_file()}

    monkeypatch.setattr(cpp, "IS_ROCM", True)
    monkeypatch.setattr(cpp, "_source_checkout_root", lambda: checkout)
    extension = TorchOpsJitExtension(
        name="machete_ops",
        namespace="demo",
        required_ops=("run",),
        sources=lambda: [str(repo_source), str(cached_sources / "generated.cu")],
        build_root_env=None,
        default_build_root=tmp_path / "build",
        display_name="Machete",
        extra_include_paths=lambda: [str(include_root)],
        requires_cuda=False,
    )

    sources, includes = extension._resolved_sources_and_include_paths()

    # The tracked source is staged; the already-writable cache source is not.
    assert sources[0] == str(tmp_path / "build" / "_src" / "machete_pytorch.cu")
    assert sources[1] == str(cached_sources / "generated.cu")
    assert includes and not includes[0].startswith(str(checkout))
    assert (Path(includes[0]) / "cutlass_extensions" / "epilogue.h").is_file()
    after = {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob("*") if path.is_file()}
    assert after == before


def test_staged_include_root_refreshes_when_a_header_changes(monkeypatch, tmp_path):
    """The fingerprint hashes the staged copy, so a stale stage must not persist.

    Copying an in-checkout include root only once would leave the fingerprint
    hashing the old header, and a later edit would silently reuse a binary
    built against it (issue #4123).
    """

    checkout = tmp_path / "checkout"
    include_root = checkout / "gptqmodel_ext"
    (include_root / "cutlass_extensions").mkdir(parents=True)
    header = include_root / "cutlass_extensions" / "epilogue.h"
    header.write_text("// header v1\n", encoding="utf-8")
    repo_source = include_root / "machete" / "machete_pytorch.cu"
    repo_source.parent.mkdir(parents=True)
    # A real kernel includes the header, which is how the fingerprint reaches
    # it: the include is resolved through the (staged) include root.
    repo_source.write_text(
        '#include "cutlass_extensions/epilogue.h"\n// source\n', encoding="utf-8"
    )

    monkeypatch.setattr(cpp, "IS_ROCM", True)
    monkeypatch.setattr(cpp, "_source_checkout_root", lambda: checkout)
    extension = TorchOpsJitExtension(
        name="machete_ops",
        namespace="demo",
        required_ops=("run",),
        sources=lambda: [str(repo_source)],
        build_root_env=None,
        default_build_root=tmp_path / "build",
        display_name="Machete",
        extra_include_paths=lambda: [str(include_root)],
        requires_cuda=False,
    )

    first_fingerprint = extension._cache_fingerprint()
    _, includes = extension._resolved_sources_and_include_paths()
    staged_header = Path(includes[0]) / "cutlass_extensions" / "epilogue.h"
    assert staged_header.read_text(encoding="utf-8") == "// header v1\n"

    header.write_text("// header v2\n", encoding="utf-8")

    second_fingerprint = extension._cache_fingerprint()
    _, includes = extension._resolved_sources_and_include_paths()
    staged_header = Path(includes[0]) / "cutlass_extensions" / "epilogue.h"

    assert staged_header.read_text(encoding="utf-8") == "// header v2\n"
    assert second_fingerprint != first_fingerprint, "a header edit must change the cache key"


def _optional_header_extension(monkeypatch, tmp_path):
    """Build an extension whose source optionally includes a deletable header."""

    checkout = tmp_path / "checkout"
    source = checkout / "source" / "kernel.cpp"
    source.parent.mkdir(parents=True)
    include = checkout / "include"
    include.mkdir(parents=True)
    source.write_text(
        '#if __has_include("deleted.h")\n'
        '#include "deleted.h"\n'
        "#else\n"
        "#define VERSION 3\n"
        "#endif\n"
        "int kernel_version = VERSION;\n",
        encoding="utf-8",
    )
    header = include / "deleted.h"
    header.write_text("#define VERSION 1\n", encoding="utf-8")

    monkeypatch.setattr(cpp, "IS_ROCM", True)
    monkeypatch.setattr(cpp, "_source_checkout_root", lambda: checkout)
    extension = TorchOpsJitExtension(
        name="native_header_fixture",
        namespace="native_header_fixture",
        required_ops=("fixture",),
        sources=[str(source)],
        build_root_env=None,
        default_build_root=tmp_path / "cache",
        display_name="fixture",
        extra_include_paths=[str(include)],
    )
    return extension, source, include, header


def test_deleted_include_header_changes_the_cache_fingerprint(monkeypatch, tmp_path):
    """A deletion must invalidate the key; hashing the staged union hid it (#4487)."""

    extension, _, _, header = _optional_header_extension(monkeypatch, tmp_path)

    first = extension._cache_fingerprint()
    header.write_text("#define VERSION 2\n", encoding="utf-8")
    edited = extension._cache_fingerprint()
    header.unlink()
    deleted = extension._cache_fingerprint()

    assert first != edited
    assert deleted != edited, "deleting an included header must change the cache key"


def test_deleted_include_header_is_absent_from_the_staged_generation(monkeypatch, tmp_path):
    """The staged snapshot must not retain a header removed from the authoritative tree."""

    extension, source, _, header = _optional_header_extension(monkeypatch, tmp_path)
    extension._resolved_sources_and_include_paths()
    staged_root = extension._staging_root()
    assert (staged_root / "_checkout" / "include" / "deleted.h").is_file()

    header.unlink()
    extension._resolved_sources_and_include_paths()

    staged_source = staged_root / "kernel.cpp"
    staged_include = staged_root / "_checkout" / "include"
    assert not (staged_include / "deleted.h").exists()
    assert staged_source.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")

    compiler = shutil.which("c++")
    if compiler is not None:
        output = subprocess.check_output(
            [compiler, "-E", "-P", "-I", str(staged_include), str(staged_source)],
            text=True,
        ).strip()
        assert output == "int kernel_version = 3;"


def test_fingerprint_and_lock_wait_never_stage_sources(monkeypatch, tmp_path):
    """A waiter that times out must not overwrite the lock holder's staged sources (#4487)."""

    if cpp.fcntl is None:
        pytest.skip("requires fcntl-based cross-process locking")

    monkeypatch.setattr(cpp, "IS_ROCM", True)
    base = tmp_path / "fixture"
    for version, value in (("v1", 1), ("v2", 2)):
        (base / version).mkdir(parents=True)
        (base / version / "kernel.cpp").write_text(f"int kernel_version = {value};\n", encoding="utf-8")

    def loader(version):
        return TorchOpsJitExtension(
            name="native_stage_fixture",
            namespace="native_stage_fixture",
            required_ops=("fixture",),
            sources=[str(base / version / "kernel.cpp")],
            build_root_env=None,
            default_build_root=base / "cache",
            display_name="fixture",
        )

    holder = loader("v1")
    with cpp._cross_process_build_lock(holder._cross_process_lock_path(), timeout_seconds=1) as acquired:
        assert acquired
        holder._resolved_sources_and_include_paths()
        staged = holder._staging_root() / "kernel.cpp"
        assert staged.read_text(encoding="utf-8") == "int kernel_version = 1;\n"

    waiter = loader("v2")
    # A fingerprint is a read: it must not stage v2 over the shared tree.
    waiter._cache_fingerprint()
    assert staged.read_text(encoding="utf-8") == "int kernel_version = 1;\n"

    waiter._build_lock_timeout_seconds = lambda: 0.01
    with cpp._cross_process_build_lock(waiter._cross_process_lock_path(), timeout_seconds=1) as held:
        assert held
        assert waiter.load() is False
        assert "timed out" in waiter.last_error_message()
        assert staged.read_text(encoding="utf-8") == "int kernel_version = 1;\n"
