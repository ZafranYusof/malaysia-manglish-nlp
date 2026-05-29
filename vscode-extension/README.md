# Manglish NLP — VS Code Extension

Highlight, normalize, and analyze Manglish (Malaysian English) text directly in VS Code.

## Features

- **Shortform Detection** — Highlights Manglish shortforms (x, tk, mcm, nk, etc.) with squiggly underlines
- **Quick Fix Normalization** — Hover over a shortform to see its expansion; click to normalize
- **Sentiment Analysis** — Run sentiment analysis on selected text or entire document
- **Full Analysis** — Get emotion, intent, topic, and dialect detection
- **Auto-normalize on Save** — Optionally normalize shortforms when saving files

## Installation

1. Install the extension from the VS Code Marketplace (or load from `.vsix`)
2. Ensure the Manglish NLP API server is running:
   ```bash
   cd manglish-nlp
   python -m manglish_nlp.api
   # Server starts at http://localhost:8000
   ```

## Usage

### Commands

Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and type:

| Command | Description |
|---------|-------------|
| `Manglish: Normalize Text` | Normalize shortforms in selection or document |
| `Manglish: Analyze Text` | Full analysis (sentiment, emotion, intent, topic) |
| `Manglish: Sentiment Analysis` | Quick sentiment check on selected text |

### Inline Features

- **Squiggly underlines** appear under detected Manglish shortforms
- **Hover** over underlined text to see the normalized form
- **Quick Fix** (lightbulb icon) to replace shortform with standard form

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `manglish.serverUrl` | `http://localhost:8000` | URL of the Manglish NLP API server |
| `manglish.autoNormalizeOnSave` | `false` | Auto-normalize shortforms on file save |

Configure in VS Code Settings (`Ctrl+,`):

```json
{
  "manglish.serverUrl": "http://localhost:8000",
  "manglish.autoNormalizeOnSave": true
}
```

## Requirements

- **Manglish NLP API** must be running locally or on a reachable server
- Install the manglish-nlp package: `pip install manglish-nlp`
- Start the API: `python -m manglish_nlp.api`

## Development

```bash
cd vscode-extension
npm install
npm run compile
# Press F5 in VS Code to launch Extension Development Host
```

## Known Issues

- Large files (>10K lines) may experience delay in shortform highlighting
- Auto-normalize only works with text/markdown files by default

## License

MIT
