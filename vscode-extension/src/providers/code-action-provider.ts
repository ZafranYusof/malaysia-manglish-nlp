import * as vscode from "vscode";
import { ManglishClient } from "../manglish-client";

const SHORTFORMS: Record<string, string> = {
  x: "tidak",
  tk: "tidak",
  xde: "tiada",
  xdak: "tiada",
  nk: "nak",
  mcm: "macam",
  sbb: "sebab",
  dgn: "dengan",
  utk: "untuk",
  yg: "yang",
  ni: "ini",
  tu: "itu",
  je: "sahaja",
  jer: "sahaja",
  dah: "sudah",
  dh: "sudah",
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
  gak: "juga",
  depa: "mereka",
  lu: "kamu",
  gua: "saya",
  meh: "mari",
  korang: "kamu semua",
};

export class ManglishCodeActionProvider implements vscode.CodeActionProvider {
  static readonly providedCodeActionKinds = [vscode.CodeActionKind.QuickFix];

  constructor(private client: ManglishClient) {}

  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range,
    context: vscode.CodeActionContext,
    _token: vscode.CancellationToken
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];

    // Individual quick fixes for each diagnostic
    for (const diagnostic of context.diagnostics) {
      if (diagnostic.source !== "manglish-nlp") {
        continue;
      }

      const shortform = diagnostic.code as string;
      const normalized = SHORTFORMS[shortform];

      if (normalized) {
        const fix = new vscode.CodeAction(
          `Normalize: "${shortform}" → "${normalized}"`,
          vscode.CodeActionKind.QuickFix
        );
        fix.edit = new vscode.WorkspaceEdit();
        fix.edit.replace(document.uri, diagnostic.range, normalized);
        fix.diagnostics = [diagnostic];
        fix.isPreferred = true;
        actions.push(fix);
      }
    }

    // "Normalize all in selection" if multiple diagnostics
    const manglishDiags = context.diagnostics.filter(
      (d) => d.source === "manglish-nlp"
    );

    if (manglishDiags.length > 1) {
      const fixAll = new vscode.CodeAction(
        `Normalize all Manglish shortforms (${manglishDiags.length} found)`,
        vscode.CodeActionKind.QuickFix
      );
      fixAll.edit = new vscode.WorkspaceEdit();

      for (const d of manglishDiags) {
        const sf = d.code as string;
        const norm = SHORTFORMS[sf];
        if (norm) {
          fixAll.edit.replace(document.uri, d.range, norm);
        }
      }

      actions.push(fixAll);
    }

    // "Normalize to formal Malay" via API for selection
    if (!range.isEmpty) {
      const apiNormalize = new vscode.CodeAction(
        "Normalize to formal Malay (API)",
        vscode.CodeActionKind.RefactorRewrite
      );
      apiNormalize.command = {
        command: "manglish.normalize",
        title: "Normalize to formal Malay",
      };
      actions.push(apiNormalize);
    }

    return actions;
  }
}
