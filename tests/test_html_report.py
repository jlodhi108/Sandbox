import os
import tempfile

from main import write_html_report


_SUCCESS_RESULTS = [
    {
        "file_path": "foo.py",
        "language": "python",
        "chunks_total": 2,
        "chunks_succeeded": 1,
        "chunks_already_modern": 0,
        "chunks_gave_up": 1,
        "output_path": None,
        "chunk_details": [
            {
                "kind": "function", "start_byte": 0, "status": "success", "iterations": 1,
                "used_escalation": False, "risk_flag": True, "security_flag": False,
                "mutation_confidence_flag": False, "duration_seconds": 1.2,
                "diff": "--- a\n+++ b\n-old\n+new\n",
            },
            {
                "kind": "function", "start_byte": 50, "status": "gave_up", "iterations": 5,
                "used_escalation": True, "risk_flag": False, "security_flag": False,
                "mutation_confidence_flag": False, "duration_seconds": 4.5,
            },
        ],
    },
]


def test_write_html_report_renders_chunk_and_flag_data():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        write_html_report(path, "repo", _SUCCESS_RESULTS)
        content = open(path).read()

    assert content.startswith("<!doctype html>")
    assert "foo.py" in content
    assert "success" in content
    assert "gave_up" in content
    assert "risk" in content
    assert "escalated" in content
    assert "old" in content and "new" in content  # diff rendered


def test_write_html_report_handles_file_level_errors():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        write_html_report(path, "repo", [{"file_path": "bad.py", "error": "boom"}])
        content = open(path).read()

    assert "bad.py" in content
    assert "boom" in content
    assert "ERROR" in content


def test_write_html_report_escapes_untrusted_content():
    results = [{
        "file_path": "<script>alert(1)</script>.py",
        "language": "python",
        "chunks_total": 0, "chunks_succeeded": 0, "chunks_already_modern": 0,
        "chunks_gave_up": 0, "output_path": None, "chunk_details": [],
    }]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        write_html_report(path, "repo", results)
        content = open(path).read()

    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content
