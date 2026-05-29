# Manglish NLP for VS Code

Analyze, normalize, and translate Manglish (Malaysian colloquial Malay) text directly in your editor.

![Version](https://img.shields.io/badge/version-0.2.0-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Features

### Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| **Manglish: Analyze Selection** | `Ctrl+Shift+M` | Full analysis: sentiment, emotion, intent, topic, dialect, NER in a rich webview panel |
| **Manglish: Normalize** | `Ctrl+Shift+N` | Normalize Manglish shortforms to formal Malay. Replace in-place or view in panel |
| **Manglish: Translate** | `Ctrl+Shift+T` | Translate between BM and EN. Replace or view side-by-side |
| **Manglish: Sentiment Analysis** | - | Quick sentiment check on selected text |
| **Manglish: Named Entity Recognition** | - | Extract people, places, organizations from text |
| **Manglish: Check API Status** | - | Verify the API server is reachable |

### Inline Features

- **Hover Provider** - Hover over Manglish shortforms to see normalized form + line sentiment score
- **Diagnostics** - Squiggly underlines on detected Manglish shortforms with hint messages
- **Quick Fix (Code Action)** - Press `Ctrl+.` on underlined text to normalize individual shortforms or all at once
- **Status Bar** - Real-time language/dialect detection displayed in the status bar

### Right-Click Context Menu

All commands available via editor context menu under **Manglish NLP** submenu.

## Requirements

The extension requires the **Manglish NLP API server** running locally or on a reachable host.

```bash
# Install the Python package
pip install malaysian-manglish-nlp

# Start the API server
python -m malaysian_manglish_nlp.api
# Server starts at http://localhost:8000
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `manglish.serverUrl` | `http://localhost:8000` | API server URL |
| `manglish.autoNormalizeOnSave` | `false` | Auto-normalize shortforms on file save |
| `manglish.requestTimeout` | `10000` | API request timeout (ms) |
| `manglish.enableDiagnostics` | `true` | Underline shortforms with hints |
| `manglish.enableHover` | `true` | Show sentiment on hover |
| `manglish.enableStatusBar` | `true` | Show language in status bar |

Configure in VS Code Settings (`Ctrl+,`):

```json
{
  "manglish.serverUrl": "http://localhost:8000",
  "manglish.autoNormalizeOnSave": false,
  "manglish.requestTimeout": 10000
}
```

## Supported Shortforms

The extension recognizes 80+ Manglish shortforms including:

| Shortform | Formal Malay |
|-----------|-------------|
| x / tk | tidak |
| xde / xdak | tiada |
| nk | nak |
| mcm | macam |
| sbb | sebab |
| dgn / ngan | dengan |
| utk | untuk |
| yg | yang |
| je / jer | sahaja |
| dah / dh | sudah |
| blh | boleh |
| kat / kt | dekat |
| skrg | sekarang |
| cmne | macam mana |
| org | orang |
| byk | banyak |

Plus many more. Full dictionary loaded from the API server.

## Development

```bash
cd vscode-extension
npm install
npm run compile

# Press F5 in VS Code to launch Extension Development Host
# Or package as VSIX:
npm run package
```

## Architecture

```
src/
  extension.ts              # Main entry - command registration, webview panels
  manglish-client.ts        # REST API client with types
  providers/
    hover-provider.ts       # Hover for sentiment + shortform expansion
    diagnostics-provider.ts # Shortform underline detection
    code-action-provider.ts # Quick fix to normalize
    status-bar.ts           # Language detection in status bar
```

## License

MIT
