# Manglish NLP Chrome Extension

Chrome extension for analyzing Malaysian Manglish text on any webpage. Connects to the manglish-nlp REST API.

## Features

- **Popup Analyzer** - Type/paste text, pick analysis type (Sentiment, Normalize, NER)
- **Right-click Menu** - Select text on any page, right-click "Analyze with Manglish NLP"
- **In-page Tooltips** - Results appear near your selection
- **Entity Highlighting** - NER results highlight entities in the page
- **Dark/Light Theme** - Toggle in popup
- **Configurable API** - Point to any manglish-nlp server

## Installation

1. Make sure the manglish-nlp API server is running (default: `http://localhost:8000`)
2. Open Chrome, go to `chrome://extensions/`
3. Enable **Developer mode** (toggle top-right)
4. Click **Load unpacked**
5. Select the `chrome-extension/` folder
6. Extension icon appears in toolbar

## Usage

### Popup
- Click extension icon in toolbar
- Type or paste Manglish text
- Select tab: Sentiment / Normalize / NER
- Click Analyze (or Ctrl+Enter)

### Right-click
- Select any text on a webpage
- Right-click > "Analyze with Manglish NLP" > pick analysis type
- Result tooltip appears near selection

### Settings
- Click ⚙️ Settings in popup footer
- Change API endpoint URL
- Click Save

## API Endpoints Used

| Feature   | Endpoint         | Method |
|-----------|-----------------|--------|
| Sentiment | POST /sentiment  | JSON   |
| Normalize | POST /normalize  | JSON   |
| NER       | POST /ner        | JSON   |
| Translate | POST /translate  | JSON   |
| Full      | POST /analyze    | JSON   |

Request body: `{ "text": "your manglish text here" }`

## Building Icons

The included icons are simple teal circles. To regenerate:

```bash
python -c "
import struct, zlib
# ... (see build script) or replace icons/icons/*.png with your own
"
```

Or drop your own `icon16.png`, `icon48.png`, `icon128.png` into `icons/`.

## Development

No build step needed. Edit files, reload extension in `chrome://extensions/`.

## License

Same as manglish-nlp project.
