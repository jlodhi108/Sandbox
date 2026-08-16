# Security Policy

## Supported Versions

This project has not yet cut any tagged releases — there is one supported
line: the `main` branch. Security fixes are applied there.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately rather than opening a
public issue: email **jlodhi108@gmail.com** with a description and, if
possible, steps to reproduce. Expect an initial response within 5 business
days. If the report is confirmed, a fix will be prioritized on `main`;
if it's declined (e.g. out of scope, not reproducible), you'll get an
explanation why.

## What's in scope

- Sandbox/isolation escapes — code executed during chunk verification
  breaking out of the sandbox (see `sandbox/verifier.py`, which supports
  running under gVisor (`runsc`) and applies a custom seccomp profile
  blocking syscalls like `ptrace`).
- Path traversal or arbitrary file write/overwrite via the `path` CLI
  argument or MCP tool arguments.
- Secret leakage — `GITHUB_TOKEN`/`LANGSMITH_API_KEY` (from `.env`) ending
  up in logs, LLM prompts, generated reports, or committed config.
- Prompt injection in target source code that causes the pipeline to
  exfiltrate secrets, escape the sandbox, or take actions outside the
  intended modernization scope.

## Known limitations (not yet hardened)

- User-supplied file/directory paths (CLI `path` argument, MCP tool
  arguments) are not currently validated against path traversal — treat
  this tool as operating with the permissions of whoever runs it, and
  don't point it at paths from untrusted input.
- This project runs arbitrary target-repo code (and LLM-rewritten code)
  inside a sandbox for verification, but that sandbox is only as strong as
  Docker + the configured isolation layer (gVisor/seccomp) — don't rely on
  it as a hard security boundary against a fully malicious target repo.
