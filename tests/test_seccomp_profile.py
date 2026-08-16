"""Unit tests for the seccomp hardening in sandbox/verifier.py. Mocks the
Docker client entirely — these prove the WIRING is correct (security_opt
passed only when the profile is loaded, omitted when disabled), not that
the profile itself actually blocks anything at the kernel level, which
needs a real container (see sandbox/verifier.py's __main__ self-test,
which verifies ptrace is genuinely blocked against the real sandbox
image).
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


def test_verify_passes_security_opt_when_profile_loaded():
    mock_client, _ = _make_mock_client()
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "_seccomp_profile_json", '{"defaultAction": "SCMP_ACT_ERRNO"}'):
        verifier_module.verify("print(1)", "main.py", "python3 main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert call_kwargs["security_opt"] == ['seccomp={"defaultAction": "SCMP_ACT_ERRNO"}']


def test_verify_omits_security_opt_when_profile_disabled():
    mock_client, _ = _make_mock_client()
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "_seccomp_profile_json", None):
        verifier_module.verify("print(1)", "main.py", "python3 main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert "security_opt" not in call_kwargs


def test_run_semgrep_passes_security_opt_when_profile_loaded():
    mock_client, _ = _make_mock_client(logs=b'{"results": [], "errors": []}')
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "_seccomp_profile_json", '{"defaultAction": "SCMP_ACT_ERRNO"}'):
        verifier_module.run_semgrep("print(1)", "main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert call_kwargs["security_opt"] == ['seccomp={"defaultAction": "SCMP_ACT_ERRNO"}']


def test_run_semgrep_omits_security_opt_when_profile_disabled():
    mock_client, _ = _make_mock_client(logs=b'{"results": [], "errors": []}')
    with patch.object(verifier_module, "_get_client", return_value=mock_client), \
         patch.object(verifier_module, "_seccomp_profile_json", None):
        verifier_module.run_semgrep("print(1)", "main.py")

    call_kwargs = mock_client.containers.run.call_args[1]
    assert "security_opt" not in call_kwargs


def test_seccomp_profile_file_is_valid_json_and_omits_ptrace():
    # The actual shipped profile, not a mock — confirms the file itself
    # stays valid and the intentional hardening (no unconditional
    # ptrace/process_vm_readv/process_vm_writev allow) isn't accidentally
    # reverted by a future edit.
    import json
    with open(verifier_module._SECCOMP_PROFILE_PATH) as f:
        profile = json.load(f)
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    allowed_names = {name for group in profile["syscalls"] for name in group["names"]}
    assert "ptrace" not in allowed_names
    assert "process_vm_readv" not in allowed_names
    assert "process_vm_writev" not in allowed_names
    # Sanity: this must still be a substantial, permissive base profile —
    # not an accidentally-emptied file that would break every toolchain.
    assert len(allowed_names) > 300
