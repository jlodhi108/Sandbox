import os
import tempfile

from config import load_config, apply_config_to_environment, load_recipes, load_profiles, BUILTIN_PROFILES


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


def test_load_recipes_extracts_instruction_strings():
    config = {
        "recipes": {
            "callbacks-to-async": {"instruction": "Convert callbacks to async/await."},
            "py2-to-py3": {"instruction": "Fix Python 2/3 incompatibilities."},
        }
    }
    recipes = load_recipes(config)
    assert recipes == {
        "callbacks-to-async": "Convert callbacks to async/await.",
        "py2-to-py3": "Fix Python 2/3 incompatibilities.",
    }


def test_load_recipes_returns_empty_dict_when_no_recipes_table():
    assert load_recipes({}) == {}


def test_load_recipes_skips_malformed_entries():
    config = {
        "recipes": {
            "good": {"instruction": "Do the thing."},
            "missing_instruction": {"other_key": "x"},
            "not_a_table": "oops",
        }
    }
    assert load_recipes(config) == {"good": "Do the thing."}


def test_load_profiles_returns_builtins_when_no_config():
    profiles = load_profiles({})
    assert profiles["safe"] == BUILTIN_PROFILES["safe"]
    assert profiles["fast"] == BUILTIN_PROFILES["fast"]


def test_load_profiles_user_table_overrides_individual_builtin_fields():
    config = {"profiles": {"safe": {"max_iterations": 3}}}
    profiles = load_profiles(config)
    # only max_iterations overridden — every other builtin "safe" field survives
    assert profiles["safe"]["max_iterations"] == 3
    assert profiles["safe"]["punt_check"] == BUILTIN_PROFILES["safe"]["punt_check"]


def test_load_profiles_adds_a_new_custom_profile():
    config = {"profiles": {"thorough": {"max_iterations": 10, "characterize": True}}}
    profiles = load_profiles(config)
    assert profiles["thorough"] == {"max_iterations": 10, "characterize": True}
    assert "safe" in profiles and "fast" in profiles  # builtins still present


def test_load_profiles_ignores_malformed_entries():
    config = {"profiles": {"bad": "not a table"}}
    profiles = load_profiles(config)
    assert "bad" not in profiles


def test_apply_config_to_environment_sets_base_model():
    config = {"model": {"base_model": "qwen2.5-coder:7b"}}
    env_backup = dict(os.environ)
    try:
        os.environ.pop("BASE_MODEL", None)
        apply_config_to_environment(config)
        assert os.environ["BASE_MODEL"] == "qwen2.5-coder:7b"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_apply_config_to_environment_never_overrides_existing_base_model():
    config = {"model": {"base_model": "config-file-model"}}
    env_backup = dict(os.environ)
    try:
        os.environ["BASE_MODEL"] = "shell-set-model"
        apply_config_to_environment(config)
        assert os.environ["BASE_MODEL"] == "shell-set-model"
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


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


def test_apply_config_to_environment_sets_reviewer_model():
    config = {"escalation": {"reviewer_model": "qwen2.5-coder:1.5b"}}
    env_backup = dict(os.environ)
    try:
        os.environ.pop("REVIEWER_MODEL", None)
        apply_config_to_environment(config)
        assert os.environ["REVIEWER_MODEL"] == "qwen2.5-coder:1.5b"
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


def test_apply_config_to_environment_sets_embedding_model():
    config = {"context": {"embedding_model": "nomic-embed-text"}}
    env_backup = dict(os.environ)
    try:
        os.environ.pop("EMBEDDING_MODEL", None)
        apply_config_to_environment(config)
        assert os.environ["EMBEDDING_MODEL"] == "nomic-embed-text"
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
