import * as vscode from "vscode";
import { ManglishClient, SentimentResult } from "../manglish-client";

// Manglish shortforms for local detection
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
  jer: "sahaja",
  gak: "juga",
  dh: "dah",
  xdak: "tiada",
  depa: "mereka",
  hang: "kamu",
  aku: "saya",
  lu: "kamu",
  gua: "saya",
  bro: "saudara",
  weh: "hei",
  lah: "(partikel)",
  kan: "bukan",
  meh: "mari",
  korang: "kamu semua",
};

export class ManglishHoverProvider implements vscode.HoverProvider {
  private sentimentCache = new Map<string, { result: SentimentResult; ts: number }>();
  private readonly CACHE_TTL = 30000; // 30s

  constructor(private client: ManglishClient) {}

  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    token: vscode.CancellationToken
  ): Promise<vscode.Hover | undefined> {
    // Get word under cursor
    const wordRange = document.getWordRangeAtPosition(position, /[\w]+/);
    if (!wordRange) {
      return undefined;
    }

    const word = document.getText(wordRange);
    const lower = word.toLowerCase();

    // Check if it's a known shortform
    const normalized = SHORTFORMS[lower];
    if (normalized) {
      // Build sentence context for sentiment
      const line = document.lineAt(position.line).text;

      // Try to get sentiment for the line (cached)
      let sentimentInfo = "";
      const cached = this.sentimentCache.get(line);
      if (cached && Date.now() - cached.ts < this.CACHE_TTL) {
        const s = cached.result;
        const pct = (s.confidence * 100).toFixed(1);
        sentimentInfo = `\n\n**Sentiment:** ${s.sentiment} (${pct}%)`;
      }

      const md = new vscode.MarkdownString();
      md.isTrusted = true;
      md.appendMarkdown(`**Manglish shortform**\n\n`);
      md.appendMarkdown(`**${word}** → **${normalized}**\n`);
      md.appendMarkdown(`\n_Quick fix available (Ctrl+.)_${sentimentInfo}`);

      return new vscode.Hover(md, wordRange);
    }

    // For non-shortform words, check if line contains manglish and show sentiment
    const line = document.lineAt(position.line).text;
    const hasManglish = this.lineContainsManglish(line);

    if (hasManglish) {
      try {
        let result = this.sentimentCache.get(line)?.result;

        if (!result || Date.now() - (this.sentimentCache.get(line)?.ts || 0) > this.CACHE_TTL) {
          if (token.isCancellationRequested) {
            return undefined;
          }
          result = await this.client.analyzeSentiment(line);
          this.sentimentCache.set(line, { result, ts: Date.now() });
        }

        const pct = (result.confidence * 100).toFixed(1);
        const emoji = this.sentimentEmoji(result.sentiment);

        const md = new vscode.MarkdownString();
        md.appendMarkdown(`${emoji} **Line Sentiment:** ${result.sentiment} (${pct}%)\n`);

        if (result.scores) {
          md.appendMarkdown(`\n| Label | Score |\n|-------|-------|\n`);
          for (const [label, score] of Object.entries(result.scores)) {
            md.appendMarkdown(`| ${label} | ${(score * 100).toFixed(1)}% |\n`);
          }
        }

        return new vscode.Hover(md, wordRange);
      } catch {
        // API unavailable, skip hover
        return undefined;
      }
    }

    return undefined;
  }

  private lineContainsManglish(line: string): boolean {
    const words = line.toLowerCase().split(/\s+/);
    return words.some((w) => SHORTFORMS[w] !== undefined);
  }

  private sentimentEmoji(sentiment: string): string {
    switch (sentiment.toLowerCase()) {
      case "positive":
        return "😊";
      case "negative":
        return "😞";
      case "neutral":
        return "😐";
      default:
        return "📊";
    }
  }

  clearCache(): void {
    this.sentimentCache.clear();
  }
}
