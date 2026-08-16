"""Pre-modernization characterization testing (see --characterize in
main.py): pin the ORIGINAL, unmodified function's observed behavior —
including its quirks and bugs — into a durable, standalone test file,
BEFORE (or independent of whether) any rewrite attempt succeeds.

This is a different guarantee than regression_tests.py's
--generate-regression-tests, which only ever writes a test for a chunk
that already SUCCEEDED, embedding the MODERNIZED code. Characterization
testing exists for the case that matters more for a genuinely legacy,
poorly-tested codebase: a permanent safety net for the ORIGINAL
behavior, written whether or not this project's own modernization
attempt on that chunk succeeds — so a chunk that gives up (or that a
human decides to modernize by hand later) still leaves behind a durable
record of what it used to do, the same "characterization test" concept
Michael Feathers' "Working Effectively with Legacy Code" describes:
pin down what the code does today, not what it should do, so ANY future
change (by this tool, by hand, by a different tool entirely) has
something to diff against.

Reuses regression_tests.py's per-language file generators (same
probe-embedding mechanics, same safety considerations — e.g. a probe
snippet with a trailing `//` comment corrupting a same-line brace) via
their source_field/header parameters rather than duplicating ~200 lines
of per-language template code; only the DATA (original_code instead of
modernized_code) and the file naming/header differ."""
import os

from regression_tests import (
    _generate_python_test, _generate_js_test, _generate_php_test,
    _CHARACTERIZATION_HEADER_PY, _CHARACTERIZATION_HEADER_JS, _CHARACTERIZATION_HEADER_PHP,
)


def characterization_test_filename(language: str, original_file_path: str) -> str:
    """e.g. calc.py -> test_calc_characterization.py. Deliberately a
    DIFFERENT filename than regression_test_filename's output (never
    `_modernized`) — both can be generated for the same file in the same
    run (--characterize and --generate-regression-tests are independent
    flags) without colliding."""
    directory, filename = os.path.split(original_file_path)
    stem, ext = os.path.splitext(filename)
    if language == "python":
        return os.path.join(directory, f"test_{stem}_characterization.py")
    if language in ("javascript", "typescript"):
        return os.path.join(directory, f"{stem}.characterization.test{ext}")
    if language == "php":
        return os.path.join(directory, f"{stem}CharacterizationTest.php")
    raise ValueError(f"characterization tests not supported for language: {language}")


def generate_characterization_test_file(language: str, chunk_characterizations: list[dict]) -> str | None:
    """chunk_characterizations: [{"original_code": str, "probes":
    [{"snippet", "baseline_stdout"}, ...]}, ...] — every chunk this run
    LOOKED AT (attempted and succeeded, attempted and gave up; NOT
    chunks skipped by --punt-check, which never capture probe data at
    all, and NOT chunks skipped as already-modern, which have nothing
    'legacy' to characterize). Returns the generated test file's source
    code, or None if no chunk has any probes at all, or the language
    isn't one of the 4 that generate probes (python/javascript/
    typescript/php — see agents/nodes.py's supports_function_probe)."""
    chunk_characterizations = [c for c in chunk_characterizations if c.get("probes")]
    if not chunk_characterizations:
        return None
    if language == "python":
        return _generate_python_test(
            chunk_characterizations, source_field="original_code", header=_CHARACTERIZATION_HEADER_PY,
            class_name="TestCharacterization",
        )
    if language == "javascript":
        return _generate_js_test(chunk_characterizations, typescript=False, source_field="original_code", header=_CHARACTERIZATION_HEADER_JS)
    if language == "typescript":
        return _generate_js_test(chunk_characterizations, typescript=True, source_field="original_code", header=_CHARACTERIZATION_HEADER_JS)
    if language == "php":
        return _generate_php_test(chunk_characterizations, source_field="original_code", header=_CHARACTERIZATION_HEADER_PHP)
    return None
