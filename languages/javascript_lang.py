import re
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language

from languages.base import LanguageHandler

# \bvar\b (word boundary) so this doesn't false-match "variable", "avatar",
# etc. — one of the anti-patterns this project's refactor prompt targets.
_VAR_KEYWORD_RE = re.compile(r"\bvar\b")


def _js_already_modern(code: str) -> bool:
    # "No var" alone is too weak: a plain `function foo() {...}` with no
    # var inside its OWN body still isn't "modern" — arrow-function-first
    # is the actual target here, and that check requires knowing the
    # function is written in `const x = (...) => ...` form already, not
    # just that it avoids one anti-pattern. Confirmed by testing against
    # legacy.js directly: an absence-only check flagged BOTH sample
    # functions as "already modern" even though we'd already demonstrated
    # both benefit from arrow-function conversion.
    stripped = code.lstrip()
    already_arrow_style = stripped.startswith(("const ", "let "))
    return already_arrow_style and _VAR_KEYWORD_RE.search(code) is None

_QUERY_SRC = """
(function_declaration) @function
(method_definition) @function
(lexical_declaration
  (variable_declarator
    value: (arrow_function))) @function
"""
# The arrow-function pattern matters both for chunking legacy files (some
# already use `const f = () => ...`) and, critically, for the structural
# output validator: modernizing a function_declaration into an arrow
# function is the single most likely "modernization" a model will make,
# so the validator must recognize that shape as a valid single chunk too
# — otherwise every arrow-function rewrite gets rejected as "0 chunks".

_REFACTOR_PROMPT_TEMPLATE = """You are a {label} modernization engine. You
will be given ONE function or method, extracted from a larger file. Rewrite
ONLY that function to modern, idiomatic {label} (ES2020+): use const/let
instead of var, arrow functions where appropriate, template literals
instead of string concatenation, async/await instead of raw .then() chains,
and avoid changing the function's observable behavior.

CRITICAL: Your output replaces this exact function in the original file.
- Do NOT add import/require statements inline. If your rewrite needs
  something not already imported, request it with a marker line BEFORE
  the function:
  // REQUIRES: module-name
  These are stripped automatically and hoisted to the top of the file —
  do not write a real import yourself.
- Do NOT wrap it in a class definition, even if it's a method — output
  only the method itself, exactly as it will be spliced back into the
  class body.
- Keep the function's exact name and parameters — other code calls it as-is.

Respond with ONLY the marker lines (if any) followed by the rewritten
function. No markdown fences, no commentary."""

_FIX_PROMPT_TEMPLATE = """You are a {label} modernization engine. Your
previous attempt at modernizing this function was spliced back into the
original file and the FULL FILE failed to run. Fix the function based on
the error below. The error may reference other parts of the file you
cannot see — infer what changed based on the message.

Same rules as before: output ONLY marker lines (if any) plus the corrected
function, no class wrapper, no markdown fences, no commentary."""


class JavaScriptHandler(LanguageHandler):
    name = "javascript"
    extensions = (".js",)
    sandbox_filename = "main.js"
    supports_function_probe = True
    ts_language = Language(tsjs.language())
    query_src = _QUERY_SRC

    def run_command(self) -> str:
        return "node main.js"

    def already_modern(self, code: str) -> bool:
        return _js_already_modern(code)

    @property
    def refactor_system_prompt(self) -> str:
        return _REFACTOR_PROMPT_TEMPLATE.format(label="JavaScript")

    @property
    def fix_system_prompt(self) -> str:
        return _FIX_PROMPT_TEMPLATE.format(label="JavaScript")


class TypeScriptHandler(LanguageHandler):
    name = "typescript"
    extensions = (".ts",)
    sandbox_filename = "main.ts"
    supports_function_probe = True
    ts_language = Language(tsts.language_typescript())
    query_src = _QUERY_SRC

    def run_command(self) -> str:
        return "npx tsc main.ts --target ES2020 --module commonjs --outDir out --skipLibCheck && node out/main.js"

    def already_modern(self, code: str) -> bool:
        return _js_already_modern(code)

    @property
    def refactor_system_prompt(self) -> str:
        return _REFACTOR_PROMPT_TEMPLATE.format(label="TypeScript")

    @property
    def fix_system_prompt(self) -> str:
        return _FIX_PROMPT_TEMPLATE.format(label="TypeScript")
