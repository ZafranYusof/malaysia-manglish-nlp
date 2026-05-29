import * as vscode from "vscode";
import { ManglishClient, DetectResult } from "../manglish-client";

// Local shortforms for fast offline detection
const MANGISH_MARKERS = new Set([
  "x", "tk", "xde", "nk", "mcm", "sbb", "dgn", "utk", "yg", "ni", "tu",
  "je", "jer", "dah", "dh", "blh", "ngan", "kat", "kt", "lg", "skrg",
  "cmne", "ape", "sape", "bile", "mne", "psl", "mmg", "btl", "sngt",
  "byk", "skit", "org", "nape", "camtu", "camni", "gak", "depa",
  "lu", "gua", "meh", "korang", "lah", "kan", "weh",
]);

export class ManglishStatusBar {
  private statusBarItem: vscode.StatusBarItem;
  private client: ManglishClient;
  private debounceTimer: NodeJS.Timeout | undefined;
  private lastDetect: DetectResult | undefined;
  private readonly DEBOUNCE_MS = 500;

  constructor(client: ManglishClient) {
    this.client = client;
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.statusBarItem.command = "manglish.showStatusInfo";
    this.statusBarItem.tooltip = "Manglish NLP - Click for details";
    this.statusBarItem.text = "$(symbol-misc) Manglish";
    this.statusBarItem.show();
  }

  get item(): vscode.StatusBarItem {
    return this.statusBarItem;
  }

  scheduleUpdate(editor: vscode.TextEditor): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.update(editor);
    }, this.DEBOUNCE_MS);
  }

  private async update(editor: vscode.TextEditor): Promise<void> {
    const line = editor.document.lineAt(editor.selection.active.line).text;

    if (!line.trim()) {
      this.statusBarItem.text = "$(symbol-misc) --";
      this.statusBarItem.backgroundColor = undefined;
      return;
    }

    // Fast local detection first
    const words = line.toLowerCase().split(/[\s,.!?;:'"()]+/);
    const manglishWords = words.filter((w) => MANGISH_MARKERS.has(w));

    if (manglishWords.length > 0) {
      // Manglish detected locally
      this.statusBarItem.text = `$(symbol-misc) Manglish (${manglishWords.length})`;
      this.statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
      this.statusBarItem.tooltip = `Manglish detected: ${manglishWords.join(", ")}\nClick for full analysis`;
      return;
    }

    // No local manglish markers. Try API for language detection
    try {
      const result = await this.client.detectLanguage(line);
      this.lastDetect = result;

      if (result.is_manglish) {
        this.statusBarItem.text = `$(symbol-misc) ${result.dialect || "Manglish"}`;
        this.statusBarItem.backgroundColor = new vscode.ThemeColor(
          "statusBarItem.warningBackground"
        );
      } else {
        const lang = result.language || "unknown";
        this.statusBarItem.text = `$(symbol-misc) ${lang}`;
        this.statusBarItem.backgroundColor = undefined;
      }
      this.statusBarItem.tooltip = `Language: ${result.language}\nDialect: ${result.dialect || "N/A"}\nConfidence: ${(result.confidence * 100).toFixed(0)}%\nClick for details`;
    } catch {
      // API unavailable - show offline mode
      this.statusBarItem.text = "$(symbol-misc) Manglish (offline)";
      this.statusBarItem.backgroundColor = undefined;
      this.statusBarItem.tooltip = "Manglish NLP API unavailable. Using offline detection only.";
    }
  }

  showDetails(): void {
    if (this.lastDetect) {
      const r = this.lastDetect;
      vscode.window.showInformationMessage(
        `Language: ${r.language} | Dialect: ${r.dialect || "N/A"} | Confidence: ${(r.confidence * 100).toFixed(0)}% | Manglish: ${r.is_manglish ? "Yes" : "No"}`
      );
    } else {
      vscode.window.showInformationMessage("Manglish NLP: No detection data available. Move cursor to a line with text.");
    }
  }

  dispose(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.statusBarItem.dispose();
  }
}
