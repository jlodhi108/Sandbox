from chunker.cpp_chunker import parse_chunks, splice_chunks


def test_parse_chunks_finds_methods_and_free_functions_without_overlap():
    source = b"""
class Foo {
public:
    void bar() {}
};

int add(int a, int b) {
    return a + b;
}
"""
    chunks = parse_chunks(source)
    kinds = [c.kind for c in chunks]
    assert kinds == ["function", "function"]

    # no two chunks should overlap byte ranges (would corrupt splice_chunks)
    for i, a in enumerate(chunks):
        for b in chunks[i + 1:]:
            assert a.end_byte <= b.start_byte or b.end_byte <= a.start_byte


def test_splice_replaces_single_chunk_correctly():
    source = b"int add(int a, int b) {\n    return a + b;\n}\n"
    chunks = parse_chunks(source)
    assert len(chunks) == 1

    new_code = "int add(int a, int b) { return a + b; } // modernized"
    result = splice_chunks(source, [(chunks[0], new_code)])
    assert result.decode("utf-8") == new_code + "\n"


def test_splice_multiple_chunks_reverse_order_safe():
    source = b"int a() { return 1; }\nint b() { return 2; }\n"
    chunks = parse_chunks(source)
    assert len(chunks) == 2

    replacements = [
        (chunks[0], "int a() { return 100; }"),
        (chunks[1], "int b() { return 200; }"),
    ]
    result = splice_chunks(source, replacements).decode("utf-8")
    assert "return 100;" in result
    assert "return 200;" in result
