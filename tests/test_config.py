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
