import tree_sitter_cpp as tscpp
from tree_sitter import Language

from languages.base import LanguageHandler, CodeChunk
from chunker.cpp_chunker import parse_chunks


class CppHandler(LanguageHandler):
    name = "cpp"
    extensions = (".cpp", ".cc", ".cxx", ".hpp")
    sandbox_filename = "main.cpp"
    ts_language = Language(tscpp.language())
    query_src = "(function_definition) @function"

    def chunk(self, source: bytes) -> list[CodeChunk]:
        # Delegate to the original, independently-tested chunker rather
        # than duplicating the query logic here.
        return parse_chunks(source)

    def run_command(self) -> str:
        return "g++ -std=c++20 -Wall main.cpp -o main && ./main"

    def import_statement(self, module: str) -> str:
        return f"#include <{module}>\n"

    def has_import(self, source_text: str, module: str) -> bool:
        return f"#include <{module}>" in source_text

    @property
    def refactor_system_prompt(self) -> str:
        return """You are a C++ modernization engine. You will be given
ONE function or method, extracted from a larger file. Rewrite ONLY that
function to modern C++20 idioms: replace raw new/delete with RAII
(std::unique_ptr, std::vector), prefer range-based for loops, use auto where
it improves clarity, and avoid changing the program's observable behavior.

CRITICAL: Your output replaces this exact function in the original file.
- Do NOT add #include directives inline. If your rewrite needs a standard
  header that might not already be included (e.g. <memory>, <vector>,
  <algorithm>), request it by putting one marker line BEFORE the function:
  // REQUIRES: memory
  You may add multiple marker lines, one per header. These lines are
  stripped automatically and the headers are added to the top of the file
  for you — do not write a real #include yourself.
- Do NOT wrap it in a class definition, even if it's a method — output only
  the method itself, exactly as it will be spliced back into the class body.
- Do NOT add a main() function or any other declarations.
- Keep the function's name, parameter types, and return type
  behavior-compatible with the rest of the file, which calls it as-is.

Respond with ONLY the marker lines (if any) followed by the rewritten
function. No markdown fences, no commentary."""

    @property
    def fix_system_prompt(self) -> str:
        return """You are a C++ modernization engine. Your previous attempt
at modernizing this function was spliced back into the original file and the
FULL FILE failed to compile or run. Fix the function based on the
compiler/runtime error below. The error may reference other parts of the file
that you cannot see — infer what changed based on the message. If the error
is a missing header (e.g. "'unique_ptr' is not a member of 'std'"), add a
// REQUIRES: memory marker line — do not write a real #include.

Same rules as before: output ONLY marker lines (if any) plus the corrected
function, no class wrapper, no main(), no markdown fences, no commentary."""
