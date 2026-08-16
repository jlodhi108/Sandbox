import tree_sitter_php as tsphp
from tree_sitter import Language

from languages.base import LanguageHandler


class PhpHandler(LanguageHandler):
    name = "php"
    extensions = (".php",)
    sandbox_filename = "main.php"
    ts_language = Language(tsphp.language_php())
    query_src = """
    (function_definition) @function
    (method_declaration) @function
    """
    parse_wrapper_prefix = "<?php\n"

    def run_command(self) -> str:
        # -l lints (catches parse errors with a clean message before we
        # even try to execute), then run for real.
        return "php -l main.php && php main.php"

    def import_statement(self, module: str) -> str:
        return f"use {module};\n"

    def has_import(self, source_text: str, module: str) -> bool:
        return f"use {module};" in source_text

    @property
    def refactor_system_prompt(self) -> str:
        return """You are a PHP modernization engine. You will be given ONE
function or method, extracted from a larger file. Rewrite ONLY that
function to modern PHP 8 idioms: add scalar type hints and return types,
use the null-safe operator (?->) instead of manual null checks, use match
instead of verbose switch, and avoid changing the function's observable
behavior.

CRITICAL: Your output replaces this exact function in the original file.
- Do NOT add `use` import statements inline. If your rewrite needs a
  namespaced class not already imported, request it with a marker line
  BEFORE the function:
  // REQUIRES: Some\\Namespace\\ClassName
  These are stripped automatically and hoisted to the top of the file —
  do not write a real `use` statement yourself.
- Do NOT wrap it in a class definition, even if it's a method — output
  only the method itself, exactly as it will be spliced back into the
  class body.
- Do NOT include the `<?php` opening tag.
- Keep the function's exact name and parameters — other code calls it as-is.

Respond with ONLY the marker lines (if any) followed by the rewritten
function. No markdown fences, no commentary."""

    @property
    def fix_system_prompt(self) -> str:
        return """You are a PHP modernization engine. Your previous attempt
at modernizing this function was spliced back into the original file and
the FULL FILE failed to lint or run. Fix the function based on the error
below. The error may reference other parts of the file you cannot see —
infer what changed based on the message.

Same rules as before: output ONLY marker lines (if any) plus the corrected
function, no class wrapper, no <?php tag, no markdown fences, no
commentary."""
