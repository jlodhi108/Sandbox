"""SARIF 2.1.0 output for accumulated semgrep security findings (see
agents/nodes.py:scan_security) — the standard interchange format GitHub
Code Scanning / Advanced Security (and GitLab, and most other CI security
dashboards) ingest directly, so findings this project already computes
show up in the same place a team's other SAST tooling already reports
to, instead of only in this project's own JSON/HTML reports.

Builds the document FROM the already-computed findings (main.py's
`security_flagged` stats, accumulated per file across a run) rather than
re-running semgrep with `--sarif` — avoids a second sandbox round-trip
per flagged chunk for output-format reasons alone; the underlying data
is identical either way."""
import json
import os

SARIF_SCHEMA_URI = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
TOOL_NAME = "code-modernizer-semgrep"
TOOL_INFO_URI = "https://github.com/jlodhi108/Sandbox"


def build_sarif(results: list[dict]) -> dict:
    """results: the same list write_report/write_html_report consume —
    one stats dict per file (main.py's run_file output), each with
    "file_path" and "security_flagged": [{"start_line", "findings":
    [{"rule_id", "line", "message"}, ...]}, ...]. Files with an "error"
    key (never actually processed) are skipped. Returns a full SARIF
    2.1.0 document; empty `results` (list of rules) if no chunk was ever
    flagged is still valid SARIF — an empty results array is exactly
    how "scanned, found nothing" is represented."""
    rule_ids: set[str] = set()
    sarif_results = []

    for s in results:
        if "error" in s or not s.get("file_path"):
            continue
        # Relative, forward-slash URI — SARIF's artifactLocation.uri is
        # meant to be portable across machines (e.g. a CI runner's
        # absolute path is meaningless to whoever reads the uploaded
        # report later), and GitHub's SARIF ingestion specifically wants
        # paths relative to the repo root.
        uri = os.path.relpath(s["file_path"]).replace(os.sep, "/")
        for entry in s.get("security_flagged", []):
            start_line = entry.get("start_line", 1)
            for finding in entry.get("findings", []):
                rule_id = finding["rule_id"]
                rule_ids.add(rule_id)
                # finding["line"] is 1-indexed WITHIN the isolated chunk
                # semgrep actually scanned (see scan_security's
                # docstring) — offset by the chunk's real start line in
                # the file, minus 1 since both are 1-indexed.
                real_line = max(1, start_line + finding.get("line", 1) - 1)
                sarif_results.append({
                    "ruleId": rule_id,
                    "message": {"text": finding["message"]},
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": uri},
                            "region": {"startLine": real_line},
                        },
                    }],
                })

    rules = [{"id": rule_id, "shortDescription": {"text": rule_id}} for rule_id in sorted(rule_ids)]
    return {
        "version": "2.1.0",
        "$schema": SARIF_SCHEMA_URI,
        "runs": [{
            "tool": {
                "driver": {
                    "name": TOOL_NAME,
                    "informationUri": TOOL_INFO_URI,
                    "rules": rules,
                },
            },
            "results": sarif_results,
        }],
    }


def write_sarif_report(path: str, results: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(build_sarif(results), f, indent=2)
