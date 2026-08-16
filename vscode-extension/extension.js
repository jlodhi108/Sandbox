// Minimal VS Code extension for code-modernizer. Deliberately thin: it
// shells out to the EXISTING Python CLI (main.py) rather than
// reimplementing any pipeline logic here or talking to mcp_server.py —
// the CLI is already the fully-featured, tested entry point (--plan,
// --report-html, --recipe, etc. all Just Work by forwarding to it), so
// this extension's only job is "run it and show the result inside the
// editor" rather than duplicating behavior in JS.
//
// No bundler, no npm dependencies beyond the vscode API itself — plain
// CommonJS, matching the rest of this project's "no dependency beyond
// what a feature strictly needs" philosophy (see requirements.txt's
// per-line comments).
const vscode = require("vscode");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const os = require("os");

let outputChannel;

function activate(context) {
  outputChannel = vscode.window.createOutputChannel("Code Modernizer");
  context.subscriptions.push(outputChannel);

  context.subscriptions.push(
    vscode.commands.registerCommand("codeModernizer.planFile", () => runPlan(activeFilePath())),
    vscode.commands.registerCommand("codeModernizer.modernizeFile", () => runModernize(activeFilePath())),
    vscode.commands.registerCommand("codeModernizer.modernizeWorkspace", () => runModernize(workspacePath())),
  );
}

function deactivate() {}

function activeFilePath() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showErrorMessage("Code Modernizer: no active file to run on.");
    return null;
  }
  return editor.document.uri.fsPath;
}

function workspacePath() {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showErrorMessage("Code Modernizer: no workspace folder open.");
    return null;
  }
  return folders[0].uri.fsPath;
}

function getConfig() {
  const config = vscode.workspace.getConfiguration("codeModernizer");
  const repoPath = config.get("repoPath", "");
  if (!repoPath) {
    vscode.window.showErrorMessage(
      "Code Modernizer: set codeModernizer.repoPath in Settings to your local code-modernizer checkout first.",
    );
    return null;
  }
  return {
    repoPath,
    pythonPath: config.get("pythonPath", "python3"),
    extraArgs: config.get("extraArgs", []),
  };
}

// Runs main.py with `args`, streaming stdout/stderr into the output
// channel as it arrives (not just at the end) — a modernization run can
// take minutes across many chunks, and a silent extension with no
// progress signal for that long reads as hung, not busy.
function runCli(args) {
  const config = getConfig();
  if (!config) return Promise.resolve(null);
  const mainPy = path.join(config.repoPath, "main.py");
  if (!fs.existsSync(mainPy)) {
    vscode.window.showErrorMessage(
      `Code Modernizer: main.py not found at ${mainPy} — check codeModernizer.repoPath.`,
    );
    return Promise.resolve(null);
  }

  outputChannel.clear();
  outputChannel.show(true);
  outputChannel.appendLine(`$ ${config.pythonPath} ${mainPy} ${args.join(" ")}\n`);

  return new Promise((resolve) => {
    const child = spawn(config.pythonPath, [mainPy, ...args, ...config.extraArgs], { cwd: config.repoPath });
    child.stdout.on("data", (chunk) => outputChannel.append(chunk.toString()));
    child.stderr.on("data", (chunk) => outputChannel.append(chunk.toString()));
    child.on("error", (err) => {
      outputChannel.appendLine(`\nFailed to start: ${err.message}`);
      resolve({ code: -1 });
    });
    child.on("close", (code) => resolve({ code }));
  });
}

async function runPlan(targetPath) {
  if (!targetPath) return;
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Code Modernizer: planning..." },
    () => runCli([targetPath, "--plan"]),
  );
}

async function runModernize(targetPath) {
  if (!targetPath) return;
  const reportPath = path.join(os.tmpdir(), `code-modernizer-report-${Date.now()}.html`);

  const result = await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Code Modernizer: modernizing..." },
    () => runCli([targetPath, "--report-html", reportPath]),
  );
  if (!result) return;

  if (result.code !== 0) {
    vscode.window.showErrorMessage(`Code Modernizer exited with code ${result.code} — see the output panel.`);
    return;
  }

  if (fs.existsSync(reportPath)) {
    showReport(reportPath);
  } else {
    vscode.window.showInformationMessage("Code Modernizer finished, but no report was written — see output panel.");
  }

  // Single-file mode: also offer VS Code's native diff view, which is a
  // more familiar/actionable surface than reading the HTML report for
  // "should I accept this" on one file specifically. Skipped for
  // workspace runs (many files — the HTML report's per-file diffs
  // already cover that case).
  if (fs.statSync(targetPath).isFile()) {
    const ext = path.extname(targetPath);
    const modernizedPath = targetPath.slice(0, -ext.length) + ".modernized" + ext;
    if (fs.existsSync(modernizedPath)) {
      const original = vscode.Uri.file(targetPath);
      const modernized = vscode.Uri.file(modernizedPath);
      vscode.commands.executeCommand(
        "vscode.diff", original, modernized, `${path.basename(targetPath)} ↔ modernized`,
      );
    }
  }
}

function showReport(reportPath) {
  const panel = vscode.window.createWebviewPanel(
    "codeModernizerReport", "Code Modernizer Report", vscode.ViewColumn.Beside, { enableScripts: false },
  );
  // The report is fully self-contained (inline CSS, no external assets,
  // no <script> tags — see main.py:write_html_report) so loading its
  // raw content directly into the webview is safe: no CSP exceptions or
  // localResourceRoots needed for anything it references.
  panel.webview.html = fs.readFileSync(reportPath, "utf8");
}

module.exports = { activate, deactivate };
