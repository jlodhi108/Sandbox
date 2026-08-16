"""Unit tests for the optional SANDBOX_RUNTIME (e.g. gVisor's "runsc")
support in sandbox/verifier.py. Mocks the Docker client entirely — these
prove the WIRING is correct (runtime kwarg passed only when configured,
omitted otherwise), not that gVisor itself works, which is a host-level
Linux setup step outside what a unit test (or this Mac) can verify.
"""
from unittest.mock import MagicMock, patch

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
