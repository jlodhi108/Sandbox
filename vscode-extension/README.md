# Code Modernizer for VS Code

A thin VS Code extension over code-modernizer's own CLI (`main.py`) — it
shells out to your local checkout, streams progress into an output
panel, and shows results inline (a diff view for single-file runs, the
self-contained HTML run report for everything else). It doesn't
reimplement any pipeline logic; the CLI stays the single source of
truth.

## Requirements

- A local [code-modernizer](../README.md) checkout with its own
  dependencies installed (`./setup.sh`), Docker running, and Ollama
  serving the configured model — exactly what running the CLI directly
  needs, since that's what this extension does under the hood.

## Setup

1. Open VS Code Settings and set:
   - `codeModernizer.repoPath` — absolute path to your code-modernizer
     checkout (the directory containing `main.py`).
   - `codeModernizer.pythonPath` — the Python interpreter to run it
     with, typically `<repoPath>/.venv/bin/python`.
2. Optionally set `codeModernizer.extraArgs` for flags you always want
   applied, e.g. `["--recipe", "py2-to-py3"]`.

## Commands

Open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`) and run:

- **Code Modernizer: Plan Current File** — scope + estimated LLM-call
  cost, zero LLM/Docker calls, output streamed to the "Code Modernizer"
  output panel.
- **Code Modernizer: Modernize Current File** — runs the full pipeline
  on the active file. On success, opens VS Code's native diff view
  (original vs. `*.modernized.*`) plus the HTML run report in a side
  panel.
- **Code Modernizer: Modernize Workspace** — same, but repo mode on the
  first workspace folder. Opens the HTML report only (a diff view per
  file doesn't make sense for a multi-file run — the report already
  covers that).

Both file-scoped commands are also available from the editor's
right-click context menu.

## Running it locally (development)

This extension has no build step — plain CommonJS, no bundler. To try
it in a VS Code Extension Development Host:

```bash
cd vscode-extension
code .
# Press F5 in VS Code — opens a new "Extension Development Host" window
# with this extension loaded.
```

There is no packaged `.vsix` published yet; install from source via the
steps above, or package it yourself with `vsce package` if you have
[`@vscode/vsce`](https://github.com/microsoft/vscode-vsce) installed.
