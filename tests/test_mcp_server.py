"""Tier 1 (no Docker/LLM needed): the MCP tool surface itself — correct
tools registered, and _update_track_record's pure dict logic. Actually
calling a tool end-to-end (modernize_file/modernize_repo) needs a real
LLM + Docker sandbox, exercised manually rather than in this fast suite
— identical in spirit to why agents.graph.modernize() isn't called for
real here either (see test_graph_integration.py's own docstring)."""
import asyncio
from unittest.mock import patch

import mcp_server


def test_expected_tools_are_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "modernize_file", "modernize_repo", "plan",
        "check_track_record", "list_track_record",
    }


def test_resolve_recipe_instruction_returns_none_for_no_recipe():
    assert mcp_server._resolve_recipe_instruction(None) is None


def test_resolve_recipe_instruction_looks_up_configured_recipe():
    with patch.object(mcp_server.main, "_config", {"recipes": {"good": {"instruction": "Do the thing."}}}):
        assert mcp_server._resolve_recipe_instruction("good") == "Do the thing."


def test_resolve_recipe_instruction_raises_for_unknown_recipe():
    with patch.object(mcp_server.main, "_config", {"recipes": {"good": {"instruction": "Do the thing."}}}):
        try:
            mcp_server._resolve_recipe_instruction("bad")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "bad" in str(e)
            assert "good" in str(e)


def test_plan_reports_scope_and_cost_without_llm_or_docker_calls():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        file_path = os.path.join(root, "calc.py")
        with open(file_path, "w") as f:
            f.write("def add(a, b):\n    return a + b\n")

        result = mcp_server.plan(file_path)
        assert result["summary"]["files"] == 1
        assert result["summary"]["chunks_total"] == 1
        assert result["files"][0]["file_path"] == file_path


def test_update_track_record_accumulates_and_persists():
    with patch.object(mcp_server.main, "_history", {}), \
         patch("mcp_server.save_history") as mock_save:
        mcp_server._update_track_record([
            {"language": "python", "chunks_succeeded": 3, "chunks_gave_up": 1},
        ])
        assert mcp_server.main._history == {"python": {"chunks_succeeded": 3, "chunks_attempted": 4}}
        mock_save.assert_called_once_with(mcp_server.main._history)


def test_update_track_record_skips_entries_with_no_language_or_no_attempts():
    with patch.object(mcp_server.main, "_history", {}), \
         patch("mcp_server.save_history") as mock_save:
        mcp_server._update_track_record([
            {"error": "no language field at all"},
            {"language": "python", "chunks_succeeded": 0, "chunks_gave_up": 0},  # all already-modern
        ])
        assert mcp_server.main._history == {}
        mock_save.assert_not_called()  # nothing changed — must not touch disk


def test_update_track_record_folds_multiple_files_into_one_save():
    with patch.object(mcp_server.main, "_history", {}), \
         patch("mcp_server.save_history") as mock_save:
        mcp_server._update_track_record([
            {"language": "python", "chunks_succeeded": 2, "chunks_gave_up": 0},
            {"language": "python", "chunks_succeeded": 1, "chunks_gave_up": 1},
            {"language": "javascript", "chunks_succeeded": 1, "chunks_gave_up": 0},
        ])
        assert mcp_server.main._history == {
            "python": {"chunks_succeeded": 3, "chunks_attempted": 4},
            "javascript": {"chunks_succeeded": 1, "chunks_attempted": 1},
        }
        mock_save.assert_called_once()  # one save for the whole repo run, not one per file


def test_check_track_record_reflects_current_history():
    with patch.object(mcp_server.main, "_history", {"python": {"chunks_succeeded": 8, "chunks_attempted": 10}}):
        result = mcp_server.check_track_record("python")
        assert result["eligible_for_auto_pr"] is True
        assert result["chunks_succeeded"] == 8
        assert result["chunks_attempted"] == 10


def test_check_track_record_fails_closed_for_unknown_language():
    with patch.object(mcp_server.main, "_history", {}):
        result = mcp_server.check_track_record("cobol")
        assert result["eligible_for_auto_pr"] is False
        assert result["chunks_succeeded"] == 0


def test_list_track_record_reflects_updates_made_since_server_startup():
    # main._history is loaded once at import time; a later
    # _update_track_record call in this same server process must be
    # visible to subsequent reads, not a stale snapshot from startup —
    # this is what makes track record accumulate correctly across
    # multiple tool calls within one long-lived server session.
    with patch.object(mcp_server.main, "_history", {}), \
         patch("mcp_server.save_history"):
        assert mcp_server.list_track_record() == {}
        mcp_server._update_track_record([
            {"language": "php", "chunks_succeeded": 1, "chunks_gave_up": 0},
        ])
        assert mcp_server.list_track_record() == {"php": {"chunks_succeeded": 1, "chunks_attempted": 1}}
