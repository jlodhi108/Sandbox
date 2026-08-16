import os
import tomllib

DEFAULT_CONFIG_PATH = ".modernizer.toml"


def load_config(path: str | None = None) -> dict:
    """Load .modernizer.toml (or an explicit path). Returns {} if the
    file doesn't exist — the config file is entirely optional, every
    setting it can provide already has a hardcoded default or CLI flag.

    Deliberately does NOT support secrets (GITHUB_TOKEN, LANGSMITH_API_KEY):
    this file is meant to be committed to the repo it configures, and a
    committed TOML file is not a safe place for a token. GITHUB_REPO and
    the LangSmith project/endpoint (not secret) are fine here; the
    tokens/API keys stay .env/shell-env-only."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_recipes(config: dict) -> dict[str, str]:
    """Named modernization recipes from .modernizer.toml's [recipes.<name>]
    tables — {name: instruction}. Each recipe's `instruction` string is
    appended to the language handler's base system prompt (see
    agents/nodes.py:_with_recipe) to SCOPE what a run does, e.g. "only
    convert callback-style functions to async/await, leave everything
    else untouched" — without that, every chunk gets the same generic
    'modernize this' prompt regardless of what the user actually wants
    changed. A malformed entry (missing `instruction`) is skipped rather
    than raising, so one bad recipe in the file doesn't break every run
    that doesn't even use --recipe."""
    recipes = config.get("recipes", {})
    result = {}
    for name, table in recipes.items():
        if isinstance(table, dict) and isinstance(table.get("instruction"), str):
            result[name] = table["instruction"]
    return result


def apply_config_to_environment(config: dict) -> None:
    """Push config values that other modules read from os.environ at
    IMPORT time (ESCALATION_MODEL, ESCALATION_THRESHOLD, REVIEWER_MODEL,
    GITHUB_REPO, EMBEDDING_MODEL) into the environment — using setdefault,
    so an already-set env var (from the shell, or from .env via
    load_dotenv()) always wins over the config file. Precedence end to
    end: CLI flag > env var > .env file > config file > hardcoded default.

    Must be called BEFORE `from agents.graph import modernize` for
    ESCALATION_MODEL/ESCALATION_THRESHOLD/REVIEWER_MODEL/SANDBOX_RUNTIME/
    EMBEDDING_MODEL specifically — those are read from os.environ once at
    that import's module-load time (agents.graph imports embeddings,
    which reads EMBEDDING_MODEL exactly like agents.nodes reads
    ESCALATION_MODEL), so anything set only here would silently never
    take effect if this ran after it. LANGSMITH_TRACING/PROJECT/ENDPOINT
    don't share that constraint (confirmed by reading langchain_core's
    callback manager: tracing config is resolved fresh on every
    .invoke() call, not baked in when ChatOllama is constructed) but are
    applied here too, for one consistent place to reason about
    configuration."""
    escalation = config.get("escalation", {})
    if "model" in escalation:
        os.environ.setdefault("ESCALATION_MODEL", str(escalation["model"]))
    if "threshold" in escalation:
        os.environ.setdefault("ESCALATION_THRESHOLD", str(escalation["threshold"]))
    if "reviewer_model" in escalation:
        os.environ.setdefault("REVIEWER_MODEL", str(escalation["reviewer_model"]))

    github = config.get("github", {})
    if "repo" in github:
        os.environ.setdefault("GITHUB_REPO", str(github["repo"]))

    sandbox = config.get("sandbox", {})
    if "runtime" in sandbox:
        os.environ.setdefault("SANDBOX_RUNTIME", str(sandbox["runtime"]))

    context = config.get("context", {})
    if "embedding_model" in context:
        os.environ.setdefault("EMBEDDING_MODEL", str(context["embedding_model"]))

    observability = config.get("observability", {})
    if observability.get("tracing"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    if "project" in observability:
        os.environ.setdefault("LANGSMITH_PROJECT", str(observability["project"]))
    if "endpoint" in observability:
        os.environ.setdefault("LANGSMITH_ENDPOINT", str(observability["endpoint"]))
