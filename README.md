# code-modernizer

[![test](https://github.com/jlodhi108/Sandbox/actions/workflows/test.yml/badge.svg)](https://github.com/jlodhi108/Sandbox/actions/workflows/test.yml)

An automated code modernization engine. Point it at a legacy source file or
an entire repository and it rewrites outdated code chunk-by-chunk using a
local LLM (via [Ollama](https://ollama.com/)), verifying every rewrite in an
isolated sandbox before accepting it — structural checks, behavioral probes,
determinism checks, and optional semgrep security scanning — so nothing
gets written back unless it's proven equivalent to the original.

Supports Python, JavaScript, Java, C++, and PHP out of the box.

## How it works

1. **Chunk** — the target file is parsed (tree-sitter) and split into
   independently modernizable units (functions/methods).
2. **Modernize** — each chunk is sent to a local model, which rewrites it.
   Chunks that are already modern are skipped with zero LLM calls.
3. **Verify** — the rewrite is checked structurally, run in a sandboxed
   subprocess (optionally under gVisor/seccomp), probed with fuzzed
   call-site inputs, and checked for determinism against the original.
   Failures feed back into the next rewrite attempt (up to
   `--max-iterations`); a chunk that never passes is left untouched rather
   than risking a bad write.
4. **(Optional) Escalate** — if a cheap model keeps failing the same chunk,
   retry with a stronger model (`ESCALATION_MODEL`).
5. **(Optional) Ship** — open a GitHub PR per modernized file (`--pr`), or
   run the target repo's own test suite against the result
   (`--run-target-tests`).

See [`agents/`](agents/) for the LangGraph pipeline, [`sandbox/`](sandbox/)
for the verification/isolation layer, and [`languages/`](languages/) for
per-language chunking and syntax handling.

## Requirements

- Python 3.10+
- [Docker](https://www.docker.com/) (running) — used for sandboxed
  verification of every rewrite
- [Ollama](https://ollama.com/) — runs the local model that does the
  rewriting

## Setup

```bash
git clone https://github.com/jlodhi108/Sandbox.git
cd Sandbox
./setup.sh
```

`setup.sh` creates a `.venv`, installs dependencies, copies `.env.example`
→ `.env` and `.modernizer.toml.example` → `.modernizer.toml`, and checks
that Docker/Ollama are available. Run it once (and again whenever
`requirements.txt` changes).

Then pull the model:

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

## Usage

```bash
./run.sh <path_to_file_or_directory> [flags]
```

Examples:

```bash
# Modernize a single file
./run.sh path/to/legacy_module.py

# Modernize a whole repo, 4 files concurrently, open a PR per file
./run.sh ../my-legacy-project --workers 4 --pr

# Cap spend and write a structured run report
./run.sh ../my-legacy-project --max-llm-calls 200 --report run-report.json

# Pause for human approval on any risky/low-confidence chunk
./run.sh path/to/legacy_module.py --interactive
```

Key flags (see `python main.py --help` for the full list):

| Flag | Description |
|---|---|
| `--pr` | Open a GitHub PR for each modernized file |
| `--max-iterations N` | Max fix-attempt loops per chunk (default 5) |
| `--max-llm-calls N` | Hard ceiling on LLM calls for the run |
| `--workers N` | Modernize N files concurrently in repo mode |
| `--run-target-tests` | Run the target repo's own test suite as an extra gate |
| `--generate-regression-tests` | Generate regression tests for modernized chunks |
| `--interactive` | Pause and prompt for approval on flagged chunks |
| `--report PATH` | Write a structured JSON run report |
| `--report-html PATH` | Write a self-contained HTML run report (per-chunk status, flags, diffs) — open it in a browser |

## Configuration

- **`.env`** — secrets only: `GITHUB_TOKEN`, `GITHUB_REPO`, optional
  `LANGSMITH_API_KEY`. Never committed (gitignored).
- **`.modernizer.toml`** — everything else: model selection, escalation,
  sandbox/isolation settings, autonomy thresholds, observability. See
  `.modernizer.toml.example` for every option and its rationale.

## Running as an MCP server

`mcp_server.py` exposes the same pipeline as MCP tools, so an MCP client
(Claude Code, Cursor, Claude Desktop, etc.) can drive a modernization run
directly:

```bash
python mcp_server.py
```

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Security

See [SECURITY.md](SECURITY.md) for the sandboxing/isolation model and how
to report a vulnerability.

## License

[MIT](LICENSE)
