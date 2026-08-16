import os
import tomllib

DEFAULT_CONFIG_PATH = ".modernizer.toml"


def load_config(path: str | None = None) -> dict:
    """Load .modernizer.toml (or an explicit path). Returns {} if the
    file doesn't exist — the config file is entirely optional, every
    setting it can provide already has a hardcoded default or CLI flag.

    Deliberately does NOT support secrets (GITHUB_TOKEN): this file is
    meant to be committed to the repo it configures, and a committed
    TOML file is not a safe place for a token. GITHUB_REPO (not secret)
    is fine here; GITHUB_TOKEN stays .env/shell-env-only."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def apply_config_to_environment(config: dict) -> None:
    """Push config values that other modules read from os.environ at
    IMPORT time (ESCALATION_MODEL, ESCALATION_THRESHOLD, GITHUB_REPO)
    into the environment — using setdefault, so an already-set env var
    (from the shell, or from .env via load_dotenv()) always wins over
    the config file. Precedence end to end: CLI flag > env var >
    .env file > config file > hardcoded default.

    Must be called BEFORE `from agents.graph import modernize` — that
    import chain reads ESCALATION_MODEL/ESCALATION_THRESHOLD from
    os.environ at module load time, so anything set only here would
    silently never take effect if this ran after that import."""
    escalation = config.get("escalation", {})
    if "model" in escalation:
        os.environ.setdefault("ESCALATION_MODEL", str(escalation["model"]))
    if "threshold" in escalation:
        os.environ.setdefault("ESCALATION_THRESHOLD", str(escalation["threshold"]))

    github = config.get("github", {})
    if "repo" in github:
        os.environ.setdefault("GITHUB_REPO", str(github["repo"]))
