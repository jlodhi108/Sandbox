from abc import ABC, abstractmethod
from dataclasses import dataclass
from tree_sitter import Language, Parser, Query, QueryCursor


@dataclass
class CodeChunk:
    kind: str
    code: str
    start_byte: int
    end_byte: int


class LanguageHandler(ABC):
    """One implementation per target language. The agent graph, sandbox
    verifier, and main.py are all written against this interface only —
    adding a language means adding one subclass, not touching the core
    pipeline."""

    name: str                  # e.g. "python"
    extensions: tuple[str, ...]  # e.g. (".py",)
    sandbox_filename: str      # filename written inside the container, e.g. "main.py"
    ts_language: Language      # tree-sitter grammar
    query_src: str             # tree-sitter query selecting chunkable nodes

    # Some grammars refuse to recognize code as the target language without
    # a wrapper (PHP requires a leading `<?php` tag or everything parses as
    # plain HTML/text with zero matches). Prepended only when parsing a
    # standalone snippet in isolation (e.g. the model's raw output during
    # structural validation) — never touches real files, which already
    # have their own wrapper if they need one.
    parse_wrapper_prefix: str = ""

    def chunk(self, source: bytes) -> list[CodeChunk]:
        parser = Parser(self.ts_language)
        tree = parser.parse(source)

        query = Query(self.ts_language, self.query_src)
        captures = QueryCursor(query).captures(tree.root_node)  # {name: [Node, ...]}

        chunks = []
        seen_ranges = set()
        for capture_name, nodes in captures.items():
            for node in nodes:
                key = (node.start_byte, node.end_byte)
                if key in seen_ranges:
                    continue
                seen_ranges.add(key)
                chunks.append(
                    CodeChunk(
                        kind=capture_name,
                        code=source[node.start_byte:node.end_byte].decode("utf-8"),
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                    )
                )

        chunks.sort(key=lambda c: c.start_byte)
        return chunks

    @abstractmethod
    def run_command(self) -> str:
        """Shell command executed inside the sandbox container, cwd
        /workspace, with the file already written as self.sandbox_filename.
        Must both check correctness (compile/lint) AND execute, so a
        chunk that "compiles" but crashes at runtime is still caught."""
        ...

    @property
    @abstractmethod
    def refactor_system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def fix_system_prompt(self) -> str: ...

    def import_statement(self, module: str) -> str:
        """Format one line that imports/includes `module`, used when the
        model requests a new dependency via a `REQUIRES: module` marker."""
        return f"import {module}\n"

    def has_import(self, source_text: str, module: str) -> bool:
        return self.import_statement(module).strip() in source_text

    def build_candidate(
        self,
        full_source: bytes,
        chunk_start: int,
        chunk_end: int,
        modernized_code: str,
        required_imports: list[str],
    ) -> bytes:
        """Splice the candidate chunk into the full file, prepending any
        missing imports the model requested. Default: prepend at byte 0.
        Override if the language needs imports placed elsewhere (rare)."""
        existing_text = full_source.decode("utf-8")
        missing = [m for m in required_imports if not self.has_import(existing_text, m)]
        header = "".join(self.import_statement(m) for m in missing)
        return (
            header.encode("utf-8")
            + full_source[:chunk_start]
            + modernized_code.encode("utf-8")
            + full_source[chunk_end:]
        )
