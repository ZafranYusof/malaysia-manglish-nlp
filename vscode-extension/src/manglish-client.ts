import * as vscode from "vscode";

// --- Types ---

export interface SentimentResult {
  sentiment: string;
  confidence: number;
  scores?: Record<string, number>;
}

export interface NormalizeResult {
  normalized: string;
  shortforms: Array<{ original: string; normalized: string; position?: number }>;
}

export interface NerEntity {
  text: string;
  label: string;
  start: number;
  end: number;
  confidence?: number;
}

export interface NerResult {
  entities: NerEntity[];
  text: string;
}

export interface TranslateResult {
  translated: string;
  source_lang: string;
  target_lang: string;
}

export interface AnalyzeResult {
  sentiment?: string;
  sentiment_score?: number;
  emotion?: string;
  intent?: string;
  topic?: string;
  dialect?: string;
  language?: string;
  normalized?: string;
  shortforms?: Array<{ original: string; normalized: string }>;
  entities?: NerEntity[];
}

export interface DetectResult {
  language: string;
  dialect?: string;
  confidence: number;
  is_manglish: boolean;
}

// --- Client ---

export class ManglishClient {
  private getBaseUrl(): string {
    const config = vscode.workspace.getConfiguration("manglish");
    return config.get<string>("serverUrl") || "http://localhost:8000";
  }

  private get timeout(): number {
    const config = vscode.workspace.getConfiguration("manglish");
    return config.get<number>("requestTimeout") || 10000;
  }

  private async request<T>(endpoint: string, body: Record<string, unknown>): Promise<T> {
    const url = `${this.getBaseUrl()}${endpoint}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(`API ${response.status}: ${response.statusText}${text ? ` - ${text}` : ""}`);
      }

      return (await response.json()) as T;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new Error(`Request timeout (${this.timeout}ms) to ${endpoint}`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  // --- API Methods ---

  async analyzeSentiment(text: string): Promise<SentimentResult> {
    return this.request<SentimentResult>("/sentiment", { text });
  }

  async normalize(text: string): Promise<NormalizeResult> {
    return this.request<NormalizeResult>("/normalize", { text });
  }

  async extractEntities(text: string): Promise<NerResult> {
    return this.request<NerResult>("/ner", { text });
  }

  async translate(text: string, targetLang: string): Promise<TranslateResult> {
    return this.request<TranslateResult>("/translate", {
      text,
      target_lang: targetLang,
    });
  }

  async analyze(text: string): Promise<AnalyzeResult> {
    return this.request<AnalyzeResult>("/analyze", { text });
  }

  async detectLanguage(text: string): Promise<DetectResult> {
    return this.request<DetectResult>("/detect", { text });
  }

  async healthCheck(): Promise<boolean> {
    try {
      const url = `${this.getBaseUrl()}/health`;
      const response = await fetch(url, {
        method: "GET",
        signal: AbortSignal.timeout(3000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}
