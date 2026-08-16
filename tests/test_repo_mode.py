import json
import os
import tempfile

from main import discover_files, write_report


def _touch(path: str, content: str = "x = 1\n") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_discover_files_finds_supported_extensions():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "b.cpp"))
        _touch(os.path.join(root, "readme.md"))  # unsupported, should be skipped
        found = discover_files(root)
        names = sorted(os.path.basename(f) for f in found)
        assert names == ["a.py", "b.cpp"]


def test_discover_files_skips_excluded_dirs():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "src", "a.py"))
        _touch(os.path.join(root, "node_modules", "vendor.js"))
        _touch(os.path.join(root, ".git", "hooks.py"))
        _touch(os.path.join(root, "venv", "lib.py"))
        found = discover_files(root)
        assert len(found) == 1
        assert found[0].endswith("src/a.py") or found[0].endswith("src\\a.py")


def test_discover_files_skips_own_modernized_output():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "legacy.py"))
        _touch(os.path.join(root, "legacy.modernized.py"))
        found = discover_files(root)
        names = [os.path.basename(f) for f in found]
        assert names == ["legacy.py"]


def test_discover_files_recurses_into_subdirectories():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "nested", "deep", "b.js"))
        found = discover_files(root)
        assert len(found) == 2


def test_write_report_produces_valid_json_with_expected_shape():
    with tempfile.TemporaryDirectory() as root:
        report_path = os.path.join(root, "report.json")
        fake_results = [
            {"file_path": "a.py", "language": "python", "chunks_succeeded": 2, "chunks_total": 3},
        ]
        write_report(report_path, "file", fake_results)

        with open(report_path) as f:
            data = json.load(f)

        assert data["mode"] == "file"
        assert "generated_at" in data
        assert data["results"] == fake_results
