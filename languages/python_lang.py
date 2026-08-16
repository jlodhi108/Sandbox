import tree_sitter_python as tspy
from tree_sitter import Language

from languages.base import LanguageHandler

# %-formatting, .format(), or old os.path usage — the exact anti-patterns
# this project's refactor prompt targets. Plain substrings, not regex:
# false negatives (missing a legacy pattern) just mean "send to the LLM
# anyway," the same as today's behavior — safe. False positives (skipping
# something that IS legacy) are the risk to avoid, so keep this simple
# and conservative rather than clever.
_LEGACY_MARKERS = ("%s", ".format(", "os.path.")


class PythonHandler(LanguageHandler):
    name = "python"
    extensions = (".py",)
    sandbox_filename = "main.py"
    supports_function_probe = True
    ts_language = Language(tspy.language())
    # function_definition covers both free functions and methods (methods
    # are function_definition nodes nested inside class_definition) — we
    # deliberately don't capture class_definition, same reasoning as C++:
    # it would overlap with its own methods and corrupt splicing.
    query_src = "(function_definition) @function"

    def run_command(self) -> str:
        # Python has no separate compile step; running the file is both
        # the syntax check and the correctness check.
        return "python3 main.py"

    def import_statement(self, module: str) -> str:
        return f"import {module}\n"

    def has_import(self, source_text: str, module: str) -> bool:
        return f"import {module}" in source_text

    def already_modern(self, code: str) -> bool:
        return not any(marker in code for marker in _LEGACY_MARKERS)

    @property
    def refactor_system_prompt(self) -> str:
        return """You are a Python modernization engine. You will be given
ONE function or method, extracted from a larger file. Rewrite ONLY that
function to modern, idiomatic Python 3: use f-strings instead of %
formatting or .format(), pathlib instead of os.path, type hints on
parameters and return values, comprehensions where they improve clarity,
and avoid changing the function's observable behavior.

CRITICAL: Your output replaces this exact function in the original file.
- Do NOT add import statements inline. If your rewrite needs a module not
  already imported (e.g. pathlib, itertools), request it by putting one
  marker line BEFORE the function:
  # REQUIRES: pathlib
  You may add multiple marker lines, one per module. These are stripped
  automatically and the imports are added to the top of the file for you —
  do not write a real import yourself.
- Do NOT wrap it in a class definition, even if it's a method — output only
  the method itself, exactly as it will be spliced back into the class body.
- Preserve the function's exact name and parameter names — other code in
  the file calls it as-is.
- CRITICAL indentation rule: the snippet you're given has its `def` line
  with no leading whitespace (that whitespace is outside the snippet and
  stays untouched in the file), but every line of the BODY already shows
  its true, absolute indentation depth from the original file (e.g. 8
  spaces for a method one level deep in a class). Your output must match
  that same absolute body indentation exactly — do not re-indent the body
  to start at column 0.

Respond with ONLY the marker lines (if any) followed by the rewritten
function. No markdown fences, no commentary."""

    @property
    def fix_system_prompt(self) -> str:
        return """You are a Python modernization engine. Your previous
attempt at modernizing this function was spliced back into the original
file and the FULL FILE failed to run (syntax error or exception). Fix the
function based on the error below. The error may reference other parts of
the file you cannot see — infer what changed based on the message. If the
error is a missing module (e.g. "NameError: name 'Path' is not defined"),
add a # REQUIRES: pathlib marker line — do not write a real import.

Same rules as before: output ONLY marker lines (if any) plus the corrected
function, no class wrapper, no markdown fences, no commentary."""
