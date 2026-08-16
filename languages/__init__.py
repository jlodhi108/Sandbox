import os

from languages.base import LanguageHandler, CodeChunk
from languages.cpp import CppHandler
from languages.python_lang import PythonHandler
from languages.javascript_lang import JavaScriptHandler, TypeScriptHandler
from languages.java_lang import JavaHandler
from languages.php_lang import PhpHandler

_HANDLERS: list[type[LanguageHandler]] = [
    CppHandler,
    PythonHandler,
    JavaScriptHandler,
    TypeScriptHandler,
    JavaHandler,
    PhpHandler,
]

_EXTENSION_MAP: dict[str, type[LanguageHandler]] = {
    ext: handler_cls for handler_cls in _HANDLERS for ext in handler_cls.extensions
}
_NAME_MAP: dict[str, type[LanguageHandler]] = {
    handler_cls.name: handler_cls for handler_cls in _HANDLERS
}


def get_handler(file_path: str) -> LanguageHandler:
    ext = os.path.splitext(file_path)[1]
    handler_cls = _EXTENSION_MAP.get(ext)
    if handler_cls is None:
        supported = ", ".join(sorted(_EXTENSION_MAP))
        raise ValueError(f"No modernization handler for '{ext}' files. Supported: {supported}")
    return handler_cls()


def get_handler_by_name(name: str) -> LanguageHandler:
    handler_cls = _NAME_MAP.get(name)
    if handler_cls is None:
        supported = ", ".join(sorted(_NAME_MAP))
        raise ValueError(f"Unknown language '{name}'. Supported: {supported}")
    return handler_cls()


__all__ = ["LanguageHandler", "CodeChunk", "get_handler", "get_handler_by_name"]
