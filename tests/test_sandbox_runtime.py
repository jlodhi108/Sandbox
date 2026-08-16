"""Unit tests for the optional SANDBOX_RUNTIME (e.g. gVisor's "runsc")
support in sandbox/verifier.py. Mocks the Docker client entirely — these
prove the WIRING is correct (runtime kwarg passed only when configured,
omitted otherwise), not that gVisor itself works, which is a host-level
Linux setup step outside what a unit test (or this Mac) can verify.
"""
from unittest.mock import MagicMock, patch

import docker
import requests.exceptions

import sandbox.verifier as verifier_module


def _make_mock_client(exit_code=0, logs=b"ok\n"):
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": exit_code}
    mock_container.logs.return_value = logs
    mock_client.containers.run.return_value = mock_container
    return mock_client, mock_container


def test_verify_omits_runtime_kwarg_when_not_configured():
    mock_client, _ = _make_mock_client()
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", None):
        verifier_module.verify("print(1)", "main.py", "python3 main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert "runtime" not in call_kwargs


def test_verify_passes_runtime_kwarg_when_configured():
    mock_client, _ = _make_mock_client()
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", "runsc"):
        verifier_module.verify("print(1)", "main.py", "python3 main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert call_kwargs["runtime"] == "runsc"


def test_verify_reports_real_timeout_as_execution_timed_out():
    mock_client, mock_container = _make_mock_client()
    mock_container.wait.side_effect = requests.exceptions.ReadTimeout("timed out")
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", None):
        result = verifier_module.verify("while True: pass", "main.py", "python3 main.py")

    assert result["status"] == "failed"
    assert result["stderr"] == "Execution timed out"
    mock_container.kill.assert_called_once()


def test_verify_reports_docker_api_error_distinctly_from_timeout():
    # A genuine daemon-side failure (container OOM-killed, API
    # disconnect) must NOT be mislabeled as "Execution timed out" —
    # that would misleadingly send the fix-retry loop back at a compile
    # that never actually ran.
    mock_client, mock_container = _make_mock_client()
    mock_container.wait.side_effect = docker.errors.APIError("container died")
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", None):
        result = verifier_module.verify("print(1)", "main.py", "python3 main.py")

    assert result["status"] == "failed"
    assert "Docker error" in result["stderr"]
    assert result["stderr"] != "Execution timed out"


def test_verify_retries_transient_docker_api_error_on_container_start():
    mock_client, mock_container = _make_mock_client()
    mock_client.containers.run.side_effect = [
        docker.errors.APIError("transient"),
        mock_container,
    ]
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", None), \
         patch.object(verifier_module, "_DOCKER_RETRY_DELAY_SECONDS", 0):
        result = verifier_module.verify("print(1)", "main.py", "python3 main.py")

    assert result["status"] == "success"
    assert mock_client.containers.run.call_count == 2


def test_verify_gives_up_after_exhausting_docker_retries():
    mock_client = MagicMock()
    mock_client.containers.run.side_effect = docker.errors.APIError("daemon down")
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", None), \
         patch.object(verifier_module, "_DOCKER_RETRY_DELAY_SECONDS", 0):
        try:
            verifier_module.verify("print(1)", "main.py", "python3 main.py")
            assert False, "expected ConnectionError"
        except ConnectionError as e:
            assert "Docker daemon" in str(e)
    assert mock_client.containers.run.call_count == verifier_module._DOCKER_RETRY_ATTEMPTS


def test_run_semgrep_omits_runtime_kwarg_when_not_configured():
    mock_client, _ = _make_mock_client(logs=b'{"results": [], "errors": []}')
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", None):
        verifier_module.run_semgrep("print(1)", "main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert "runtime" not in call_kwargs


def test_run_semgrep_passes_runtime_kwarg_when_configured():
    mock_client, _ = _make_mock_client(logs=b'{"results": [], "errors": []}')
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "SANDBOX_RUNTIME", "runsc"):
        verifier_module.run_semgrep("print(1)", "main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert call_kwargs["runtime"] == "runsc"
