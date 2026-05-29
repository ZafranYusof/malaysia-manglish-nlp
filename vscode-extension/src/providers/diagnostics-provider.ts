import * as vscode from "vscode";

export interface ShortformMatch {
  range: vscode.Range;
  original: string;
  normalized: string;
}

// Extended shortforms dictionary
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
  hang: "kamu",
  lu: "kamu",
  gua: "saya",
  meh: "mari",
  korang: "kamu semua",
  aritu: "hari itu",
  ari: "hari",
  mlm: "malam",
  pg: "pagi",
  tgh: "tengah",
  ptg: "petang",
  thn: "tahun",
  bln: "bulan",
  mggu: "minggu",
  hr: "hari",
  kjp: "kejap",
  jap: "kejap",
  wat: "buat",
  keje: "kerja",
  crite: "cerita",
  cite: "cerita",
  mkn: "makan",
  mnum: "minum",
  tdo: "tidur",
  bgn: "bangun",
  plg: "pulang",
  dtg: "datang",
  prgi: "pergi",
  bli: "beli",
  jual: "jual",
  byr: "bayar",
  hutg: "hutang",
  duit: "duit",
  umh: "rumah",
  kete: "kereta",
  moto: "motor",
  fon: "telefon",
  msg: "mesej",
  call: "panggil",
  ws: "WhatsApp",
  ig: "Instagram",
  fb: "Facebook",
  twt: "Twitter",
  yt: "YouTube",
};

export class ManglishDiagnosticsProvider {
  private diagnosticCollection: vscode.DiagnosticCollection;
  private debounceTimers = new Map<string, NodeJS.Timeout>();
  private readonly DEBOUNCE_MS = 300;

  constructor() {
    this.diagnosticCollection = vscode.languages.createDiagnosticCollection("manglish");
  }

  get collection(): vscode.DiagnosticCollection {
    return this.diagnosticCollection;
  }

  get shortforms(): Record<string, string> {
    return SHORTFORMS;
  }

  scheduleUpdate(document: vscode.TextDocument): void {
    const key = document.uri.toString();
    const existing = this.debounceTimers.get(key);
    if (existing) {
      clearTimeout(existing);
    }

    this.debounceTimers.set(
      key,
      setTimeout(() => {
        this.updateDiagnostics(document);
        this.debounceTimers.delete(key);
      }, this.DEBOUNCE_MS)
    );
  }

  updateDiagnostics(document: vscode.TextDocument): ShortformMatch[] {
    const diagnostics: vscode.Diagnostic[] = [];
    const matches: ShortformMatch[] = [];
    const text = document.getText();
    const words = Object.keys(SHORTFORMS);

    // Sort by length descending to match longer forms first
    words.sort((a, b) => b.length - a.length);

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
          `Manglish: "${match[0]}" → "${normalized}"`,
          vscode.DiagnosticSeverity.Hint
        );
        diagnostic.source = "manglish-nlp";
        diagnostic.code = shortform;
        diagnostic.tags = [];
        diagnostics.push(diagnostic);

        matches.push({ range, original: match[0], normalized });
      }
    }

    this.diagnosticCollection.set(document.uri, diagnostics);
    return matches;
  }

  clearDocument(uri: vscode.Uri): void {
    this.diagnosticCollection.delete(uri);
  }

  clearAll(): void {
    this.diagnosticCollection.clear();
  }

  dispose(): void {
    for (const timer of this.debounceTimers.values()) {
      clearTimeout(timer);
    }
    this.debounceTimers.clear();
    this.diagnosticCollection.dispose();
  }
}
