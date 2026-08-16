import re
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language

from languages.base import LanguageHandler

# \bvar\b (word boundary) so this doesn't false-match "variable", "avatar",
# etc. — one of the anti-patterns this project's refactor prompt targets.
_VAR_KEYWORD_RE = re.compile(r"\bvar\b")

# Anything that isn't a valid bare JS identifier character, so a module
# name like "fs/promises" or "node:fs" can be turned into a safe local
# require() binding ("fs_promises", "fs") without colliding with the
# require() PATH string itself, which tolerates those characters fine.
_JS_IDENTIFIER_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_$]")


def _js_require_binding(module: str) -> str:
    name = _JS_IDENTIFIER_SANITIZE_RE.sub("_", module.removeprefix("node:"))
    if not name or name[0].isdigit():
        name = f"_{name}"
    return name


def _js_has_import(source_text: str, module: str) -> bool:
    return f"require('{module}')" in source_text or f'require("{module}")' in source_text


# Node builtin module names that are ALSO ambient TS globals under this
# project's exact tsc invocation (--target ES2020, which implies the
# default DOM lib). Injecting `const NAME = require(...)` for either of
# these collides with the ambient `declare var NAME` TypeScript already
# provides, raising TS2451 "Cannot redeclare block-scoped variable" —
# confirmed by compiling all 54 Node builtins through this exact tsc
# invocation, one at a time: exactly these two collide, none of the
# others do. Both are also genuinely already-global at RUNTIME in Node
# 19+ (this sandbox's Node 20.x) with zero require() needed, so the fix
# isn't a scoping workaround — it's recognizing no import is needed.
_TS_AMBIENT_GLOBAL_MODULES = frozenset({"console", "crypto"})


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

# Covers both function-declaration form and const/let-assigned arrow form
# — a chunk can be either, so name extraction needs to recognize both.
_NAME_QUERY_SRC = """
(function_declaration name: (identifier) @fname)
(variable_declarator name: (identifier) @fname value: (arrow_function))
"""
_CALL_QUERY_SRC = "(call_expression function: (identifier) @fname) @call"
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
- Do NOT add import/require statements inline. This sandbox has no
  network access and installs no packages, so only Node.js BUILT-IN
  modules are usable (fs, path, crypto, util, events, os, url, and
  similar — never a third-party npm package). If your rewrite needs
  one, request it with a marker line BEFORE the function:
  // REQUIRES: fs
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
    name_query_src = _NAME_QUERY_SRC
    call_query_src = _CALL_QUERY_SRC
    ts_language = Language(tsjs.language())
    query_src = _QUERY_SRC

    def run_command(self) -> str:
        return "node main.js"

    def already_modern(self, code: str) -> bool:
        return _js_already_modern(code)

    def import_statement(self, module: str) -> str:
        # Plain Node script, no package.json — CommonJS require() is the
        # only import mechanism that works with zero extra setup here.
        return f"const {_js_require_binding(module)} = require('{module}');\n"

    def has_import(self, source_text: str, module: str) -> bool:
        return _js_has_import(source_text, module)

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
    name_query_src = _NAME_QUERY_SRC
    call_query_src = _CALL_QUERY_SRC
    ts_language = Language(tsts.language_typescript())
    query_src = _QUERY_SRC

    def run_command(self) -> str:
        return "npx tsc main.ts --target ES2020 --module commonjs --outDir out --skipLibCheck && node out/main.js"

    def already_modern(self, code: str) -> bool:
        return _js_already_modern(code)

    def import_statement(self, module: str) -> str:
        # console/crypto: already an ambient TS global AND already a
        # runtime global in Node 20.x — no import needed at all (see
        # _TS_AMBIENT_GLOBAL_MODULES for why injecting one would break
        # compilation instead of fixing anything).
        if module.removeprefix("node:") in _TS_AMBIENT_GLOBAL_MODULES:
            return ""
        # The sandbox has no @types/node (no per-run install step to put
        # it where tsc would find it), so plain `require()` fails to
        # typecheck with "Cannot find name 'require'" even though it's
        # perfectly valid at runtime. An inline ambient declaration fixes
        # that with zero image changes — confirmed safe to repeat once
        # per REQUIRES marker: TypeScript merges identical ambient
        # declarations without conflict.
        binding = _js_require_binding(module)
        return (
            "declare function require(name: string): any;\n"
            f"const {binding} = require('{module}');\n"
        )

    def has_import(self, source_text: str, module: str) -> bool:
        if module.removeprefix("node:") in _TS_AMBIENT_GLOBAL_MODULES:
            return True
        return _js_has_import(source_text, module)

    @property
    def refactor_system_prompt(self) -> str:
        return _REFACTOR_PROMPT_TEMPLATE.format(label="TypeScript")

    @property
    def fix_system_prompt(self) -> str:
        return _FIX_PROMPT_TEMPLATE.format(label="TypeScript")
