import os
import tempfile

from config import load_config, apply_config_to_environment


def test_load_config_returns_empty_dict_when_file_missing():
    config = load_config("/nonexistent/path/.modernizer.toml")
    assert config == {}


def test_load_config_parses_toml():
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write("[settings]\nmax_iterations = 7\nworkers = 3\n")
        path = f.name
    try:
        config = load_config(path)
        assert config["settings"]["max_iterations"] == 7
        assert config["settings"]["workers"] == 3
    finally:
        os.unlink(path)


def test_apply_config_to_environment_sets_escalation_vars():
    config = {"escalation": {"model": "qwen2.5-coder:32b", "threshold": 4}}
    env_backup = dict(os.environ)
    try:
        os.environ.pop("ESCALATION_MODEL", None)
        os.environ.pop("ESCALATION_THRESHOLD", None)
        apply_config_to_environment(config)
        assert os.environ["ESCALATION_MODEL"] == "qwen2.5-coder:32b"
        assert os.environ["ESCALATION_THRESHOLD"] == "4"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_apply_config_to_environment_never_overrides_existing_env_var():
    # An explicit shell/`.env` value must always win over the config
    # file — this is the whole precedence contract the feature promises.
    config = {"escalation": {"model": "config-file-model"}}
    env_backup = dict(os.environ)
    try:
        os.environ["ESCALATION_MODEL"] = "shell-set-model"
        apply_config_to_environment(config)
        assert os.environ["ESCALATION_MODEL"] == "shell-set-model"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_apply_config_to_environment_sets_github_repo():
    config = {"github": {"repo": "owner/repo"}}
    env_backup = dict(os.environ)
    try:
        os.environ.pop("GITHUB_REPO", None)
        apply_config_to_environment(config)
        assert os.environ["GITHUB_REPO"] == "owner/repo"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_apply_config_to_environment_handles_empty_config():
    # Must not raise on a missing/empty config — every setting is optional.
    apply_config_to_environment({})


def test_apply_config_to_environment_sets_sandbox_runtime():
    config = {"sandbox": {"runtime": "runsc"}}
    env_backup = dict(os.environ)
    try:
        os.environ.pop("SANDBOX_RUNTIME", None)
        apply_config_to_environment(config)
        assert os.environ["SANDBOX_RUNTIME"] == "runsc"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_apply_config_to_environment_sets_observability_vars():
    config = {"observability": {"tracing": True, "project": "code-modernizer", "endpoint": "https://x.test"}}
    env_backup = dict(os.environ)
    try:
        for key in ("LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
            os.environ.pop(key, None)
        apply_config_to_environment(config)
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "code-modernizer"
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://x.test"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_apply_config_to_environment_tracing_false_does_not_set_var():
    # Explicit `tracing = false` (or the key simply absent) must not set
    # LANGSMITH_TRACING=false — leaving it entirely unset is what
    # actually disables tracing; a literal "false" string is untested
    # territory we shouldn't rely on LangChain interpreting correctly.
    config = {"observability": {"tracing": False}}
    env_backup = dict(os.environ)
    try:
        os.environ.pop("LANGSMITH_TRACING", None)
        apply_config_to_environment(config)
        assert "LANGSMITH_TRACING" not in os.environ
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_apply_config_to_environment_never_overrides_existing_langsmith_key():
    config = {"observability": {"project": "config-project"}}
    env_backup = dict(os.environ)
    try:
        os.environ["LANGSMITH_PROJECT"] = "shell-set-project"
        apply_config_to_environment(config)
        assert os.environ["LANGSMITH_PROJECT"] == "shell-set-project"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
