import * as vscode from "vscode";
import { ManglishClient } from "./manglish-client";
import { ManglishHoverProvider } from "./providers/hover-provider";
import { ManglishDiagnosticsProvider } from "./providers/diagnostics-provider";
import { ManglishCodeActionProvider } from "./providers/code-action-provider";
import { ManglishStatusBar } from "./providers/status-bar";

let client: ManglishClient;
let outputChannel: vscode.OutputChannel;
let diagnosticsProvider: ManglishDiagnosticsProvider;
let statusBar: ManglishStatusBar;
let resultPanel: vscode.WebviewPanel | undefined;

// --- Activation ---

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel("Manglish NLP");
  client = new ManglishClient();
  diagnosticsProvider = new ManglishDiagnosticsProvider();
  statusBar = new ManglishStatusBar(client);

  outputChannel.appendLine("Manglish NLP extension activated");

  // --- Commands ---
  context.subscriptions.push(
    vscode.commands.registerCommand("manglish.analyzeSelection", cmdAnalyzeSelection),
    vscode.commands.registerCommand("manglish.normalize", cmdNormalize),
    vscode.commands.registerCommand("manglish.translate", cmdTranslate),
    vscode.commands.registerCommand("manglish.sentiment", cmdSentiment),
    vscode.commands.registerCommand("manglish.ner", cmdNer),
    vscode.commands.registerCommand("manglish.showStatusInfo", () => statusBar.showDetails()),
    vscode.commands.registerCommand("manglish.checkApi", cmdCheckApi)
  );

  // --- Providers ---
  const hoverProvider = new ManglishHoverProvider(client);

  context.subscriptions.push(
    vscode.languages.registerHoverProvider({ scheme: "file" }, hoverProvider),
    vscode.languages.registerCodeActionsProvider(
      { scheme: "file" },
      new ManglishCodeActionProvider(client),
      { providedCodeActionKinds: ManglishCodeActionProvider.providedCodeActionKinds }
    )
  );

  // --- Diagnostics on document events ---
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((doc) => {
      diagnosticsProvider.scheduleUpdate(doc);
    }),
    vscode.workspace.onDidChangeTextDocument((e) => {
      diagnosticsProvider.scheduleUpdate(e.document);
    }),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      diagnosticsProvider.clearDocument(doc.uri);
    })
  );

  // --- Status bar on editor change ---
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (editor) {
        statusBar.scheduleUpdate(editor);
      }
    }),
    vscode.window.onDidChangeTextEditorSelection((e) => {
      statusBar.scheduleUpdate(e.textEditor);
    })
  );

  // --- Auto-normalize on save ---
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((doc) => {
      const config = vscode.workspace.getConfiguration("manglish");
      if (config.get<boolean>("autoNormalizeOnSave")) {
        normalizeDocumentOnSave(doc);
      }
    })
  );

  // --- Initial run ---
  if (vscode.window.activeTextEditor) {
    diagnosticsProvider.scheduleUpdate(vscode.window.activeTextEditor.document);
    statusBar.scheduleUpdate(vscode.window.activeTextEditor);
  }

  // --- Disposables ---
  context.subscriptions.push(
    outputChannel,
    diagnosticsProvider,
    statusBar
  );
}

export function deactivate() {
  resultPanel?.dispose();
}

// --- Commands ---

async function cmdAnalyzeSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  if (!text.trim()) {
    vscode.window.showWarningMessage("No text to analyze");
    return;
  }

  try {
    const result = await client.analyze(text);

    // Also run NER in parallel
    let nerResult;
    try {
      nerResult = await client.extractEntities(text);
    } catch {
      // NER might not be available
    }

    showAnalysisPanel(result, nerResult?.entities, text);
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Unknown error";
    vscode.window.showErrorMessage(`Manglish Analysis failed: ${msg}`);
    outputChannel.appendLine(`Analyze error: ${error}`);
  }
}

async function cmdNormalize() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  if (!text.trim()) {
    vscode.window.showWarningMessage("No text to normalize");
    return;
  }

  try {
    const result = await client.normalize(text);

    if (!result.normalized) {
      vscode.window.showInformationMessage("No changes needed");
      return;
    }

    // Ask: replace or show in panel
    const action = await vscode.window.showQuickPick(
      [
        { label: "Replace selection", description: "Replace text in editor", value: "replace" },
        { label: "Show in panel", description: "View normalized text without replacing", value: "panel" },
      ],
      { placeHolder: "How to apply normalization?" }
    );

    if (!action) {
      return;
    }

    if (action.value === "replace") {
      const range = selection.isEmpty
        ? new vscode.Range(
            editor.document.positionAt(0),
            editor.document.positionAt(editor.document.getText().length)
          )
        : selection;

      await editor.edit((editBuilder) => {
        editBuilder.replace(range, result.normalized);
      });

      const count = result.shortforms?.length || 0;
      vscode.window.showInformationMessage(
        `Normalized ${count} shortform${count !== 1 ? "s" : ""}`
      );
    } else {
      showNormalizedPanel(text, result.normalized, result.shortforms);
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Unknown error";
    vscode.window.showErrorMessage(`Manglish Normalize failed: ${msg}`);
    outputChannel.appendLine(`Normalize error: ${error}`);
  }
}

async function cmdTranslate() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  if (!text.trim()) {
    vscode.window.showWarningMessage("No text to translate");
    return;
  }

  // Pick target language
  const target = await vscode.window.showQuickPick(
    [
      { label: "Malay → English", description: "Translate BM to EN", value: "en" },
      { label: "English → Malay", description: "Translate EN to BM", value: "ms" },
    ],
    { placeHolder: "Select translation direction" }
  );

  if (!target) {
    return;
  }

  try {
    const result = await client.translate(text, target.value);

    const action = await vscode.window.showQuickPick(
      [
        { label: "Replace selection", value: "replace" },
        { label: "Show in panel", value: "panel" },
      ],
      { placeHolder: "How to show translation?" }
    );

    if (!action) {
      return;
    }

    if (action.value === "replace") {
      const range = selection.isEmpty
        ? new vscode.Range(
            editor.document.positionAt(0),
            editor.document.positionAt(editor.document.getText().length)
          )
        : selection;

      await editor.edit((editBuilder) => {
        editBuilder.replace(range, result.translated);
      });
    } else {
      showTranslationPanel(text, result.translated, result.source_lang, result.target_lang);
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Unknown error";
    vscode.window.showErrorMessage(`Manglish Translate failed: ${msg}`);
    outputChannel.appendLine(`Translate error: ${error}`);
  }
}

async function cmdSentiment() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  if (!text.trim()) {
    vscode.window.showWarningMessage("No text to analyze");
    return;
  }

  try {
    const result = await client.analyzeSentiment(text);
    const pct = (result.confidence * 100).toFixed(1);
    vscode.window.showInformationMessage(
      `Sentiment: ${result.sentiment} (${pct}% confidence)`
    );
    outputChannel.appendLine(`Sentiment: ${result.sentiment} (${pct}%) on: ${text.substring(0, 80)}...`);
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Unknown error";
    vscode.window.showErrorMessage(`Manglish Sentiment failed: ${msg}`);
  }
}

async function cmdNer() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  if (!text.trim()) {
    vscode.window.showWarningMessage("No text to analyze");
    return;
  }

  try {
    const result = await client.extractEntities(text);
    showNerPanel(result.entities, text);
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Unknown error";
    vscode.window.showErrorMessage(`Manglish NER failed: ${msg}`);
  }
}

async function cmdCheckApi() {
  const healthy = await client.healthCheck();
  if (healthy) {
    vscode.window.showInformationMessage("Manglish NLP API is running ✓");
  } else {
    const config = vscode.workspace.getConfiguration("manglish");
    const url = config.get<string>("serverUrl") || "http://localhost:8000";
    vscode.window.showErrorMessage(
      `Manglish NLP API unreachable at ${url}. Start the server: python -m malaysian_manglish_nlp.api`
    );
  }
}

// --- Auto-normalize on save ---

async function normalizeDocumentOnSave(document: vscode.TextDocument) {
  const text = document.getText();
  try {
    const result = await client.normalize(text);
    if (result.normalized && result.normalized !== text) {
      const edit = new vscode.WorkspaceEdit();
      const fullRange = new vscode.Range(
        document.positionAt(0),
        document.positionAt(text.length)
      );
      edit.replace(document.uri, fullRange, result.normalized);
      await vscode.workspace.applyEdit(edit);
    }
  } catch (error) {
    outputChannel.appendLine(`Auto-normalize error: ${error}`);
  }
}

// --- Webview Panels ---

function getOrCreatePanel(): vscode.WebviewPanel {
  if (resultPanel) {
    resultPanel.reveal(vscode.ViewColumn.Beside);
    return resultPanel;
  }

  resultPanel = vscode.window.createWebviewPanel(
    "manglishResults",
    "Manglish NLP Results",
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );

  resultPanel.onDidDispose(() => {
    resultPanel = undefined;
  });

  return resultPanel;
}

function showAnalysisPanel(
  result: import("./manglish-client").AnalyzeResult,
  entities: import("./manglish-client").NerEntity[] | undefined,
  originalText: string
) {
  const panel = getOrCreatePanel();
  panel.title = "Manglish Analysis";

  const sentimentColor = getSentimentColor(result.sentiment);
  const shortformRows = (result.shortforms || [])
    .map((sf) => `<tr><td class="sf-orig">${esc(sf.original)}</td><td>→</td><td class="sf-norm">${esc(sf.normalized)}</td></tr>`)
    .join("");

  const entityRows = (entities || [])
    .map((e) => `<tr><td>${esc(e.text)}</td><td><span class="badge badge-${e.label.toLowerCase()}">${esc(e.label)}</span></td><td>${((e.confidence || 0) * 100).toFixed(0)}%</td></tr>`)
    .join("");

  panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); background: var(--vscode-editor-background); padding: 16px; line-height: 1.6; }
  h1 { font-size: 1.4em; margin-bottom: 8px; color: var(--vscode-textLink-foreground); }
  h2 { font-size: 1.1em; margin-top: 20px; color: var(--vscode-textLink-foreground); border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: 4px; }
  .card { background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 6px; padding: 12px; margin: 8px 0; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .metric { text-align: center; }
  .metric .value { font-size: 1.5em; font-weight: bold; }
  .metric .label { font-size: 0.85em; opacity: 0.7; }
  .sentiment-bar { height: 8px; border-radius: 4px; background: ${sentimentColor}; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  td, th { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--vscode-panel-border); }
  th { font-size: 0.85em; opacity: 0.7; text-transform: uppercase; }
  .sf-orig { color: var(--vscode-editorWarning-foreground); font-weight: bold; }
  .sf-norm { color: var(--vscode-testing-iconPassed); }
  .badge { padding: 2px 8px; border-radius: 3px; font-size: 0.85em; }
  .badge-person { background: #2196F3; color: white; }
  .badge-location, .badge-loc { background: #4CAF50; color: white; }
  .badge-organization, .badge-org { background: #FF9800; color: white; }
  .badge-date { background: #9C27B0; color: white; }
  .badge-event { background: #E91E63; color: white; }
  .text-preview { font-style: italic; opacity: 0.7; max-height: 60px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
</head>
<body>
  <h1>Manglish NLP Analysis</h1>
  <div class="text-preview" title="${esc(originalText)}">${esc(originalText.substring(0, 200))}${originalText.length > 200 ? "..." : ""}</div>

  <h2>Overview</h2>
  <div class="grid">
    <div class="metric card">
      <div class="value" style="color:${sentimentColor}">${esc(result.sentiment || "N/A")}</div>
      <div class="label">Sentiment</div>
      <div class="sentiment-bar"></div>
    </div>
    <div class="metric card">
      <div class="value">${esc(result.emotion || "N/A")}</div>
      <div class="label">Emotion</div>
    </div>
    <div class="metric card">
      <div class="value">${esc(result.language || "N/A")}</div>
      <div class="label">Language</div>
    </div>
    <div class="metric card">
      <div class="value">${esc(result.dialect || "N/A")}</div>
      <div class="label">Dialect</div>
    </div>
    <div class="metric card">
      <div class="value">${esc(result.intent || "N/A")}</div>
      <div class="label">Intent</div>
    </div>
    <div class="metric card">
      <div class="value">${esc(result.topic || "N/A")}</div>
      <div class="label">Topic</div>
    </div>
  </div>

  ${result.shortforms && result.shortforms.length > 0 ? `
  <h2>Shortforms (${result.shortforms.length})</h2>
  <table>
    <tr><th>Original</th><th></th><th>Normalized</th></tr>
    ${shortformRows}
  </table>` : ""}

  ${entities && entities.length > 0 ? `
  <h2>Named Entities (${entities.length})</h2>
  <table>
    <tr><th>Entity</th><th>Type</th><th>Confidence</th></tr>
    ${entityRows}
  </table>` : ""}

  ${result.normalized ? `
  <h2>Normalized Text</h2>
  <div class="card">${esc(result.normalized)}</div>` : ""}
</body>
</html>`;
}

function showNormalizedPanel(
  original: string,
  normalized: string,
  shortforms?: Array<{ original: string; normalized: string }>
) {
  const panel = getOrCreatePanel();
  panel.title = "Manglish Normalization";

  const sfRows = (shortforms || [])
    .map((sf) => `<tr><td class="sf-orig">${esc(sf.original)}</td><td>→</td><td class="sf-norm">${esc(sf.normalized)}</td></tr>`)
    .join("");

  panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); background: var(--vscode-editor-background); padding: 16px; line-height: 1.6; }
  h1 { font-size: 1.3em; color: var(--vscode-textLink-foreground); }
  h2 { font-size: 1.05em; margin-top: 20px; color: var(--vscode-textLink-foreground); }
  .text-block { background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 6px; padding: 12px; margin: 8px 0; white-space: pre-wrap; word-break: break-word; }
  .diff { border-left: 3px solid var(--vscode-testing-iconPassed); padding-left: 12px; }
  table { width: 100%; border-collapse: collapse; }
  td { padding: 4px 8px; border-bottom: 1px solid var(--vscode-panel-border); }
  .sf-orig { color: var(--vscode-editorWarning-foreground); font-weight: bold; }
  .sf-norm { color: var(--vscode-testing-iconPassed); }
</style>
</head>
<body>
  <h1>Normalization Result</h1>
  <h2>Original</h2>
  <div class="text-block">${esc(original)}</div>
  <h2>Normalized</h2>
  <div class="text-block diff">${esc(normalized)}</div>
  ${shortforms && shortforms.length > 0 ? `
  <h2>Changes (${shortforms.length})</h2>
  <table><tr><th>Original</th><th></th><th>Normalized</th></tr>${sfRows}</table>` : ""}
</body>
</html>`;
}

function showTranslationPanel(
  original: string,
  translated: string,
  sourceLang: string,
  targetLang: string
) {
  const panel = getOrCreatePanel();
  panel.title = "Manglish Translation";

  panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); background: var(--vscode-editor-background); padding: 16px; line-height: 1.6; }
  h1 { font-size: 1.3em; color: var(--vscode-textLink-foreground); }
  .lang-label { font-size: 0.85em; opacity: 0.6; text-transform: uppercase; margin-bottom: 4px; }
  .text-block { background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 6px; padding: 12px; margin: 4px 0 16px; white-space: pre-wrap; word-break: break-word; }
  .arrow { text-align: center; font-size: 1.5em; opacity: 0.4; margin: 8px 0; }
</style>
</head>
<body>
  <h1>Translation</h1>
  <div class="lang-label">${esc(sourceLang)}</div>
  <div class="text-block">${esc(original)}</div>
  <div class="arrow">↓</div>
  <div class="lang-label">${esc(targetLang)}</div>
  <div class="text-block">${esc(translated)}</div>
</body>
</html>`;
}

function showNerPanel(entities: import("./manglish-client").NerEntity[], originalText: string) {
  const panel = getOrCreatePanel();
  panel.title = "Manglish NER";

  const rows = entities
    .map((e) => `<tr><td>${esc(e.text)}</td><td><span class="badge badge-${e.label.toLowerCase()}">${esc(e.label)}</span></td><td>${((e.confidence || 0) * 100).toFixed(0)}%</td></tr>`)
    .join("");

  panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); background: var(--vscode-editor-background); padding: 16px; line-height: 1.6; }
  h1 { font-size: 1.3em; color: var(--vscode-textLink-foreground); }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  td, th { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--vscode-panel-border); }
  th { font-size: 0.85em; opacity: 0.7; }
  .badge { padding: 2px 8px; border-radius: 3px; font-size: 0.85em; color: white; }
  .badge-person { background: #2196F3; }
  .badge-location, .badge-loc { background: #4CAF50; }
  .badge-organization, .badge-org { background: #FF9800; }
  .badge-date { background: #9C27B0; }
  .badge-event { background: #E91E63; }
  .empty { opacity: 0.5; font-style: italic; }
</style>
</head>
<body>
  <h1>Named Entity Recognition</h1>
  <p>${entities.length} entities found in text (${originalText.length} chars)</p>
  ${entities.length > 0 ? `
  <table>
    <tr><th>Entity</th><th>Type</th><th>Confidence</th></tr>
    ${rows}
  </table>` : `<p class="empty">No entities detected.</p>`}
</body>
</html>`;
}

// --- Helpers ---

function getSentimentColor(sentiment?: string): string {
  switch ((sentiment || "").toLowerCase()) {
    case "positive":
      return "#4CAF50";
    case "negative":
      return "#F44336";
    case "neutral":
      return "#FF9800";
    default:
      return "#888";
  }
}

function esc(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
