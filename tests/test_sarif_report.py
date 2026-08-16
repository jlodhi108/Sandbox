import json
import os
import tempfile

from sarif_report import build_sarif, write_sarif_report


def test_build_sarif_has_valid_top_level_shape():
    doc = build_sarif([])
    assert doc["version"] == "2.1.0"
    assert "$schema" in doc
    assert doc["runs"][0]["tool"]["driver"]["name"] == "code-modernizer-semgrep"
    assert doc["runs"][0]["results"] == []


def test_build_sarif_includes_a_finding_with_correct_real_line():
    results = [{
        "file_path": "calc.py",
        "security_flagged": [
            {"start_line": 10, "findings": [{"rule_id": "py-os-system", "line": 2, "message": "danger"}]},
        ],
    }]
    doc = build_sarif(results)
    sarif_results = doc["runs"][0]["results"]
    assert len(sarif_results) == 1
    result = sarif_results[0]
    assert result["ruleId"] == "py-os-system"
    assert result["message"]["text"] == "danger"
    location = result["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] == "calc.py"
    # start_line=10 (chunk begins at real line 10) + finding line=2
    # (1-indexed within the isolated chunk) - 1 = real line 11.
    assert location["region"]["startLine"] == 11


def test_build_sarif_collects_unique_rule_ids_across_files():
    results = [
        {"file_path": "a.py", "security_flagged": [
            {"start_line": 1, "findings": [{"rule_id": "rule-a", "line": 1, "message": "x"}]},
        ]},
        {"file_path": "b.py", "security_flagged": [
            {"start_line": 1, "findings": [
                {"rule_id": "rule-a", "line": 1, "message": "y"},
                {"rule_id": "rule-b", "line": 2, "message": "z"},
            ]},
        ]},
    ]
    doc = build_sarif(results)
    rule_ids = {r["id"] for r in doc["runs"][0]["tool"]["driver"]["rules"]}
    assert rule_ids == {"rule-a", "rule-b"}
    assert len(doc["runs"][0]["results"]) == 3


def test_build_sarif_skips_files_with_errors():
    results = [{"file_path": "bad.py", "error": "boom"}]
    doc = build_sarif(results)
    assert doc["runs"][0]["results"] == []


def test_build_sarif_skips_entries_with_no_file_path():
    results = [{"security_flagged": [{"start_line": 1, "findings": [{"rule_id": "x", "line": 1, "message": "y"}]}]}]
    doc = build_sarif(results)
    assert doc["runs"][0]["results"] == []


def test_build_sarif_handles_files_with_no_security_flags():
    results = [{"file_path": "clean.py", "security_flagged": []}]
    doc = build_sarif(results)
    assert doc["runs"][0]["results"] == []


def test_real_line_never_goes_below_one():
    # Defensive: even a malformed/negative offset must not produce an
    # invalid (< 1) SARIF line number.
    results = [{
        "file_path": "calc.py",
        "security_flagged": [
            {"start_line": 1, "findings": [{"rule_id": "x", "line": -5, "message": "y"}]},
        ],
    }]
    doc = build_sarif(results)
    assert doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]["startLine"] >= 1


def test_write_sarif_report_produces_valid_json_file():
    results = [{
        "file_path": "calc.py",
        "security_flagged": [
            {"start_line": 1, "findings": [{"rule_id": "x", "line": 1, "message": "y"}]},
        ],
    }]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.sarif")
        write_sarif_report(path, results)
        with open(path) as f:
            doc = json.load(f)
    assert doc["version"] == "2.1.0"
    assert len(doc["runs"][0]["results"]) == 1
