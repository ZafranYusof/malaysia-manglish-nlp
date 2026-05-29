import * as vscode from "vscode";

// Manglish shortforms mapping
const SHORTFORMS: Record<string, string> = {
  x: "tidak",
  tk: "tidak",
  xde: "tiada",
  nk: "nak",
  mcm: "macam",
  sbb: "sebab",
  dgn: "dengan",
  utk: "untuk",
  yg: "yang",
  ni: "ini",
  tu: "itu",
  je: "sahaja",
  dah: "sudah",
  blh: "boleh",
  ngan: "dengan",
  kat: "dekat",
  kt: "dekat",
  lg: "lagi",
  skrg: "sekarang",
  cmne: "macam mana",
  ape: "apa",
  sape: "siapa",
  bile: "bila",
  mne: "mana",
  psl: "pasal",
  smpai: "sampai",
  mmg: "memang",
  btl: "betul",
  sngt: "sangat",
  byk: "banyak",
  skit: "sikit",
  org: "orang",
  nape: "kenapa",
  camtu: "macam itu",
  camni: "macam ini",
};

let diagnosticCollection: vscode.DiagnosticCollection;
let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel("Manglish NLP");
  diagnosticCollection =
    vscode.languages.createDiagnosticCollection("manglish");

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand("manglish.normalize", normalizeCommand),
    vscode.commands.registerCommand("manglish.analyze", analyzeCommand),
    vscode.commands.registerCommand("manglish.sentiment", sentimentCommand)
  );

  // Register code action provider (quick fixes)
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      { scheme: "file" },
      new ManglishCodeActionProvider(),
      { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }
    )
  );

  // Highlight shortforms on document open/change
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(highlightShortforms),
    vscode.workspace.onDidChangeTextDocument((e) =>
      highlightShortforms(e.document)
    )
  );

  // Auto-normalize on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((doc) => {
      const config = vscode.workspace.getConfiguration("manglish");
      if (config.get<boolean>("autoNormalizeOnSave")) {
        normalizeDocument(doc);
      }
    })
  );

  // Highlight active document on activation
  if (vscode.window.activeTextEditor) {
    highlightShortforms(vscode.window.activeTextEditor.document);
  }

  outputChannel.appendLine("Manglish NLP extension activated");
}

export function deactivate() {
  diagnosticCollection.dispose();
  outputChannel.dispose();
}

// --- API Communication ---

function getServerUrl(): string {
  const config = vscode.workspace.getConfiguration("manglish");
  return config.get<string>("serverUrl") || "http://localhost:8000";
}

async function callApi(
  endpoint: string,
  body: Record<string, unknown>
): Promise<unknown> {
  const url = `${getServerUrl()}${endpoint}`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// --- Shortform Highlighting ---

function highlightShortforms(document: vscode.TextDocument) {
  const diagnostics: vscode.Diagnostic[] = [];
  const text = document.getText();
  const words = Object.keys(SHORTFORMS);

  // Build regex pattern for all shortforms (word boundaries)
  const pattern = new RegExp(`\\b(${words.join("|")})\\b`, "gi");
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    const startPos = document.positionAt(match.index);
    const endPos = document.positionAt(match.index + match[0].length);
    const range = new vscode.Range(startPos, endPos);

    const shortform = match[0].toLowerCase();
    const normalized = SHORTFORMS[shortform];

    if (normalized) {
      const diagnostic = new vscode.Diagnostic(
        range,
        `Manglish shortform: "${match[0]}" → "${normalized}"`,
        vscode.DiagnosticSeverity.Information
      );
      diagnostic.source = "manglish-nlp";
      diagnostic.code = shortform;
      diagnostics.push(diagnostic);
    }
  }

  diagnosticCollection.set(document.uri, diagnostics);
}

// --- Code Action Provider (Quick Fix) ---

class ManglishCodeActionProvider implements vscode.CodeActionProvider {
  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range,
    context: vscode.CodeActionContext
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];

    for (const diagnostic of context.diagnostics) {
      if (diagnostic.source !== "manglish-nlp") continue;

      const shortform = diagnostic.code as string;
      const normalized = SHORTFORMS[shortform];

      if (normalized) {
        const action = new vscode.CodeAction(
          `Normalize: "${shortform}" → "${normalized}"`,
          vscode.CodeActionKind.QuickFix
        );

        action.edit = new vscode.WorkspaceEdit();
        action.edit.replace(document.uri, diagnostic.range, normalized);
        action.diagnostics = [diagnostic];
        action.isPreferred = true;
        actions.push(action);
      }
    }

    // Add "Normalize all" action if multiple diagnostics
    if (context.diagnostics.length > 1) {
      const normalizeAll = new vscode.CodeAction(
        "Normalize all Manglish shortforms in selection",
        vscode.CodeActionKind.QuickFix
      );
      normalizeAll.edit = new vscode.WorkspaceEdit();

      for (const diagnostic of context.diagnostics) {
        if (diagnostic.source !== "manglish-nlp") continue;
        const shortform = diagnostic.code as string;
        const normalized = SHORTFORMS[shortform];
        if (normalized) {
          normalizeAll.edit.replace(
            document.uri,
            diagnostic.range,
            normalized
          );
        }
      }

      actions.push(normalizeAll);
    }

    return actions;
  }
}

// --- Commands ---

async function normalizeCommand() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  try {
    const result = (await callApi("/normalize", { text })) as {
      normalized: string;
    };

    if (result.normalized) {
      const range = selection.isEmpty
        ? new vscode.Range(
            editor.document.positionAt(0),
            editor.document.positionAt(editor.document.getText().length)
          )
        : selection;

      await editor.edit((editBuilder) => {
        editBuilder.replace(range, result.normalized);
      });

      vscode.window.showInformationMessage("Text normalized!");
    }
  } catch (error) {
    vscode.window.showErrorMessage(
      `Manglish NLP: ${error instanceof Error ? error.message : "API error"}`
    );
    outputChannel.appendLine(`Normalize error: ${error}`);
  }
}

async function analyzeCommand() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  try {
    const result = (await callApi("/analyze", { text })) as {
      sentiment?: string;
      emotion?: string;
      intent?: string;
      topic?: string;
      dialect?: string;
      language?: string;
      normalized?: string;
      shortforms?: Array<{ original: string; normalized: string }>;
    };

    outputChannel.clear();
    outputChannel.appendLine("=== Manglish NLP Analysis ===");
    outputChannel.appendLine(`Text: ${text.substring(0, 100)}...`);
    outputChannel.appendLine("");
    outputChannel.appendLine(`Sentiment: ${result.sentiment || "N/A"}`);
    outputChannel.appendLine(`Emotion:   ${result.emotion || "N/A"}`);
    outputChannel.appendLine(`Intent:    ${result.intent || "N/A"}`);
    outputChannel.appendLine(`Topic:     ${result.topic || "N/A"}`);
    outputChannel.appendLine(`Dialect:   ${result.dialect || "N/A"}`);
    outputChannel.appendLine(`Language:  ${result.language || "N/A"}`);
    outputChannel.appendLine("");

    if (result.normalized) {
      outputChannel.appendLine(`Normalized: ${result.normalized}`);
    }

    if (result.shortforms && result.shortforms.length > 0) {
      outputChannel.appendLine("");
      outputChannel.appendLine("Shortforms found:");
      for (const sf of result.shortforms) {
        outputChannel.appendLine(`  ${sf.original} → ${sf.normalized}`);
      }
    }

    outputChannel.show();
  } catch (error) {
    vscode.window.showErrorMessage(
      `Manglish NLP: ${error instanceof Error ? error.message : "API error"}`
    );
    outputChannel.appendLine(`Analyze error: ${error}`);
  }
}

async function sentimentCommand() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active editor");
    return;
  }

  const selection = editor.selection;
  const text = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  try {
    const result = (await callApi("/sentiment", { text })) as {
      sentiment: string;
      confidence?: number;
    };

    const confidence = result.confidence
      ? ` (${(result.confidence * 100).toFixed(1)}%)`
      : "";

    vscode.window.showInformationMessage(
      `Sentiment: ${result.sentiment}${confidence}`
    );
  } catch (error) {
    vscode.window.showErrorMessage(
      `Manglish NLP: ${error instanceof Error ? error.message : "API error"}`
    );
    outputChannel.appendLine(`Sentiment error: ${error}`);
  }
}

// --- Helper ---

async function normalizeDocument(document: vscode.TextDocument) {
  const text = document.getText();

  try {
    const result = (await callApi("/normalize", { text })) as {
      normalized: string;
    };

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
