from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_list_test_files_module():
    script_path = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "list_test_files.py"
    spec = spec_from_file_location("list_test_files", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_files_ignores_nested_paths(tmp_path):
    tests_root = tmp_path / "tests"
    (tests_root / "models").mkdir(parents=True)
    (tests_root / "models" / "test_xverse.py").write_text("", encoding="utf-8")
    (tests_root / "models" / "test_yi.py").write_text("", encoding="utf-8")

    module = _load_list_test_files_module()

    torch_files, mlx_files = module.getFiles(
        ignored_test_files=["models/test_xverse.py"],
        tests_root=tests_root,
    )

    assert torch_files == ["models/test_yi"]
    assert mlx_files == []


def test_get_files_ignores_top_level_stems(tmp_path):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_one.py").write_text("", encoding="utf-8")
    (tests_root / "test_two.py").write_text("", encoding="utf-8")

    module = _load_list_test_files_module()

    torch_files, _ = module.getFiles(
        ignored_test_files=["test_one.py"],
        tests_root=tests_root,
    )

    assert torch_files == ["test_two"]
