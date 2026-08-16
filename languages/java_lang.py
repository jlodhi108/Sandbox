import tree_sitter_java as tsjava
from tree_sitter import Language

from languages.base import LanguageHandler


class JavaHandler(LanguageHandler):
    name = "java"
    extensions = (".java",)
    # Java requires the public class name to match the filename exactly —
    # legacy_samples/legacy.java must declare `public class Main`.
    sandbox_filename = "Main.java"
    ts_language = Language(tsjava.language())
    # Only method_declaration — Java has no free functions, and we leave
    # the enclosing class_declaration untouched (same overlap-avoidance
    # reasoning as every other handler).
    query_src = "(method_declaration) @function"

    def run_command(self) -> str:
        return "javac Main.java && java Main"

    def import_statement(self, module: str) -> str:
        return f"import {module};\n"

    def has_import(self, source_text: str, module: str) -> bool:
        return f"import {module};" in source_text

    @property
    def refactor_system_prompt(self) -> str:
        return """You are a Java modernization engine. You will be given
ONE method, extracted from a larger file. Rewrite ONLY that method to
modern Java (17+): use var for local type inference where it improves
clarity, streams instead of manual loops where appropriate, enhanced
switch expressions, and avoid changing the method's observable behavior.

CRITICAL: Your output replaces this exact method in the original file.
- Do NOT add import statements inline. If your rewrite needs a type not
  already imported (e.g. java.util.List), request it with a marker line
  BEFORE the method:
  // REQUIRES: java.util.List
  These are stripped automatically and hoisted to the top of the file —
  do not write a real import yourself.
- Do NOT wrap it in a class definition — output only the method itself,
  exactly as it will be spliced back into the enclosing class body.
- Keep the method's exact name, visibility modifiers, parameter types,
  and return type — other code in the file calls it as-is.

Respond with ONLY the marker lines (if any) followed by the rewritten
method. No markdown fences, no commentary."""

    @property
    def fix_system_prompt(self) -> str:
        return """You are a Java modernization engine. Your previous attempt
at modernizing this method was spliced back into the original file and the
FULL FILE failed to compile or run. Fix the method based on the
compiler/runtime error below. The error may reference other parts of the
file you cannot see — infer what changed based on the message. If the
error is a missing type (e.g. "cannot find symbol: class List"), add a
// REQUIRES: java.util.List marker line — do not write a real import.

Same rules as before: output ONLY marker lines (if any) plus the corrected
method, no class wrapper, no markdown fences, no commentary."""
