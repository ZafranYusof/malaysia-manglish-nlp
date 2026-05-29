# manglish-nlp Benchmark Dashboard

A self-contained, single-page HTML dashboard that displays benchmark results for the manglish-nlp project.

## How to Open

Just open `index.html` in any modern browser:

```bash
# Option 1: Direct open
open dashboard/index.html        # macOS
start dashboard/index.html       # Windows
xdg-open dashboard/index.html   # Linux

# Option 2: Python HTTP server
python serve.py
# Then visit http://localhost:8080
```

## What It Shows

- **Summary cards** - Total modules (43), tests passing (425), benchmark pass rate (100%), average accuracy
- **Per-task breakdown** - Tabs for Sentiment, POS, NER, Stemming, Normalization, Language Detection, Tokenization
- **Comparison charts** - CSS bar charts comparing manglish-nlp vs Malaya baseline
- **Metrics per task** - Accuracy %, speed (ms/sample), F1 scores, coverage stats
- **Version history** - Track benchmark improvements over time
- **Search/filter** - Filter tasks by name
- **Export** - Download all results as JSON

## How to Update Data

The benchmark data is hardcoded in `index.html` inside the `<script>` tag. To update:

1. Open `index.html` in a text editor
2. Find the `benchmarkData` array near the bottom
3. Update accuracy, speed, and status values
4. Update the summary cards in the HTML if totals change
5. Add new entries to the version history list

```javascript
const benchmarkData = [
  { task: "Sentiment", accuracy: 97.5, speed: 2.1, malayaAcc: 94.2, malayaSpeed: 8.5, status: "pass" },
  // ... add or modify entries here
];
```

## Tech Stack

- Vanilla HTML + CSS + JavaScript
- No external dependencies or build tools
- Dark theme with responsive design
- Works offline (no CDN or network requests)

## Color Coding

- 🟢 Green = pass (accuracy meets threshold)
- 🔴 Red = fail
- 🟡 Yellow = partial pass
