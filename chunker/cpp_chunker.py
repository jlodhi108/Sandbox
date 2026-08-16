import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Query, QueryCursor
from dataclasses import dataclass

CPP_LANGUAGE = Language(tscpp.language())

QUERY_SRC = """
(function_definition) @function
"""
# Note: we deliberately do NOT capture class_specifier. A class body
# contains its own methods as function_definition nodes, so capturing
# both would produce overlapping byte ranges and corrupt splice_chunks.
# Modernizing at function granularity (free functions AND methods,
# since both are function_definition nodes) avoids the overlap while
# still reaching every executable block in the file.


@dataclass
class CodeChunk:
    kind: str
    code: str
    start_byte: int
    end_byte: int


def parse_chunks(source_code: bytes) -> list[CodeChunk]:
    parser = Parser(CPP_LANGUAGE)
    tree = parser.parse(source_code)

    query = Query(CPP_LANGUAGE, QUERY_SRC)
    captures = QueryCursor(query).captures(tree.root_node)  # {capture_name: [Node, ...]}

    chunks = []
    seen_ranges = set()
    for capture_name, nodes in captures.items():
        for node in nodes:
            # skip nodes we've already captured (e.g. a method matched by
            # both function_definition and being inside a class_specifier)
            key = (node.start_byte, node.end_byte)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)

            chunks.append(
                CodeChunk(
                    kind=capture_name,
                    code=source_code[node.start_byte:node.end_byte].decode("utf-8"),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
            )

    # sort by position, ascending
    chunks.sort(key=lambda c: c.start_byte)
    return chunks


def splice_chunks(source_code: bytes, replacements: list[tuple[CodeChunk, str]]) -> bytes:
    """Replace chunks in the original source. MUST process in reverse byte
    order so earlier offsets stay valid as later ones are spliced in."""
    result = source_code
    for chunk, new_code in sorted(replacements, key=lambda r: r[0].start_byte, reverse=True):
        result = (
            result[:chunk.start_byte]
            + new_code.encode("utf-8")
            + result[chunk.end_byte:]
        )
    return result


if __name__ == "__main__":
    with open("../legacy_samples/legacy.cpp", "rb") as f:
        source = f.read()
    for c in parse_chunks(source):
        print(f"--- {c.kind} [{c.start_byte}:{c.end_byte}] ---")
        print(c.code)
        print()
