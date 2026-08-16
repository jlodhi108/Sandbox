"""Deterministic (non-LLM) fast-path rewrites for chunks a purely
syntactic rule can modernize with certainty, tried BEFORE the LLM call
on a chunk's first attempt (see agents/nodes.py:refactorer_node).

Two things this deliberately is NOT:
1. NOT a bypass of verification — every candidate this module produces
   still goes through the exact same structural/sandbox/probe/
   determinism pipeline (_verify_candidate) as an LLM-generated one.
   This module only ever skips the LLM CALL for chunks it's certain
   about; it never skips the safety checks that decide whether the
   result actually gets written.
2. NOT a general rule engine — each rule here is scoped to a case that
   is PROVABLY behavior-preserving by construction (e.g. `array()` and
   `[]` are the exact same PHP AST node type, just different surface
   syntax — see array_creation_expression in tree-sitter-php's grammar),
   not a heuristic guess. A rule that can't prove its own safety for a
   given chunk returns None and lets the LLM handle it instead, the same
   way this project's `already_modern()` checks fail closed (see
   languages/base.py) rather than guessing.

Each per-language rule function takes one chunk's source code and
returns either the rewritten code (str) or None (rule doesn't apply /
can't safely handle this specific chunk). try_apply() is the single
dispatch point every caller should use.
"""
from tree_sitter import Language, Parser

import tree_sitter_javascript as tsjs
import tree_sitter_php as tsphp

_JS_LANGUAGE = Language(tsjs.language())
_PHP_LANGUAGE = Language(tsphp.language_php())

_REASSIGNMENT_NODE_TYPES = {"assignment_expression", "update_expression", "augmented_assignment_expression"}


def _find_all(node, node_type: str) -> list:
    matches = []
    if node.type == node_type:
        matches.append(node)
    for child in node.children:
        matches.extend(_find_all(child, node_type))
    return matches


def _is_reassigned(root_node, code_bytes: bytes, name: str) -> bool:
    """Whether `name` appears as the TARGET of an assignment or
    increment/decrement anywhere in this chunk — var has function scope
    in JS, and this chunk IS one function/method, so searching the whole
    chunk (not just the declaration's own block) is the correct scope."""
    name_bytes = name.encode("utf-8")
    for node_type in _REASSIGNMENT_NODE_TYPES:
        for node in _find_all(root_node, node_type):
            # assignment_expression / augmented_assignment_expression:
            # target is always children[0]. update_expression (++x or
            # x++, prefix or postfix): the identifier is whichever child
            # isn't the ++/-- operator token, so pick it by TYPE rather
            # than position.
            if node.type == "update_expression":
                target = next((c for c in node.children if c.type == "identifier"), None)
            else:
                target = node.children[0] if node.children else None
            if (
                target is not None and target.type == "identifier"
                and code_bytes[target.start_byte:target.end_byte] == name_bytes
            ):
                return True
    return False


def _js_var_to_let_const(code: str) -> str | None:
    """Rewrite `var` declarations to `let` (reassigned, or no initializer
    — const requires one) or `const` (never reassigned, has an
    initializer). Conservative by design: a declaration with more than
    one declarator (`var a = 1, b = 2;`) or a destructuring pattern
    (`var {a, b} = obj;`) is left ENTIRELY alone — not just that one
    declarator, the whole rule aborts for this chunk — rather than risk
    getting the multi-variable case subtly wrong. Those are left for the
    LLM, unchanged from this project's behavior before this rule existed."""
    code_bytes = code.encode("utf-8")
    parser = Parser(_JS_LANGUAGE)
    tree = parser.parse(code_bytes)

    var_decls = [
        node for node in _find_all(tree.root_node, "variable_declaration")
        if node.children and code_bytes[node.children[0].start_byte:node.children[0].end_byte] == b"var"
    ]
    if not var_decls:
        return None

    replacements = []  # (start_byte, end_byte, new_text)
    for decl_node in var_decls:
        declarators = [c for c in decl_node.children if c.type == "variable_declarator"]
        if len(declarators) != 1:
            return None  # comma list — abort the whole rule for this chunk
        declarator = declarators[0]
        if not declarator.children or declarator.children[0].type != "identifier":
            return None  # destructuring pattern — abort
        name_node = declarator.children[0]
        name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
        has_initializer = any(c.type == "=" for c in declarator.children)

        keyword_node = decl_node.children[0]
        if _is_reassigned(tree.root_node, code_bytes, name) or not has_initializer:
            new_keyword = "let"
        else:
            new_keyword = "const"
        replacements.append((keyword_node.start_byte, keyword_node.end_byte, new_keyword))

    if not replacements:
        return None
    # Apply in REVERSE byte order so earlier offsets stay valid as each
    # splice happens — same pattern main.py's chunk-splicing loop uses.
    result = code_bytes
    for start, end, new_text in sorted(replacements, key=lambda r: r[0], reverse=True):
        result = result[:start] + new_text.encode("utf-8") + result[end:]
    return result.decode("utf-8")


def _php_array_to_bracket_syntax(code: str) -> str | None:
    """Rewrite long-form `array(...)` to short-form `[...]` — the exact
    same AST node (array_creation_expression) either way in PHP's
    grammar, just different surface syntax, so this is unconditionally
    safe for every occurrence, nested or not (each replacement operates
    on its own node's exact byte range from the parse tree, which is
    already nesting-correct by construction)."""
    code_bytes = ("<?php\n" + code).encode("utf-8")
    prefix_len = len(b"<?php\n")
    parser = Parser(_PHP_LANGUAGE)
    tree = parser.parse(code_bytes)

    long_form_nodes = [
        node for node in _find_all(tree.root_node, "array_creation_expression")
        if node.children and node.children[0].type == "array"
    ]
    if not long_form_nodes:
        return None

    replacements = []
    for node in long_form_nodes:
        keyword_node = node.children[0]  # "array"
        open_paren = node.children[1]  # "("
        close_paren = node.children[-1]  # ")"
        if open_paren.type != "(" or close_paren.type != ")":
            continue  # unexpected shape — skip just this node, not the whole rule
        replacements.append((keyword_node.start_byte, open_paren.end_byte, "["))
        replacements.append((close_paren.start_byte, close_paren.end_byte, "]"))

    if not replacements:
        return None
    result = code_bytes
    for start, end, new_text in sorted(replacements, key=lambda r: r[0], reverse=True):
        result = result[:start] + new_text.encode("utf-8") + result[end:]
    return result[prefix_len:].decode("utf-8")


_RULES = {
    "javascript": _js_var_to_let_const,
    "typescript": _js_var_to_let_const,
    "php": _php_array_to_bracket_syntax,
}


def try_apply(language: str, code: str) -> str | None:
    """Single dispatch point: try this language's deterministic rule
    against `code` (one chunk). Returns the rewritten code, or None if
    no rule exists for this language, the rule doesn't apply to this
    specific chunk, or applying it wouldn't actually change anything."""
    rule = _RULES.get(language)
    if rule is None:
        return None
    result = rule(code)
    if result is None or result == code:
        return None
    return result
