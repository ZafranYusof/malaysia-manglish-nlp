"""
Manglish NLP Demo — Gradio app for HuggingFace Spaces.
"""
import sys
import json
import traceback
from pathlib import Path

# Add project root to path for dev mode (when not installed via pip)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gradio as gr
import manglish_nlp

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
CUSTOM_CSS = """
.main-header {
    text-align: center;
    margin-bottom: 1rem;
}
.main-header h1 {
    background: linear-gradient(90deg, #0d9488, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    margin-bottom: 0.25rem;
}
.main-header p {
    color: #64748b;
    font-size: 0.95rem;
}
.result-box {
    border-radius: 8px;
    padding: 1rem;
    min-height: 80px;
}
.footer-links {
    text-align: center;
    margin-top: 1rem;
    color: #94a3b8;
    font-size: 0.85rem;
}
.footer-links a {
    color: #0d9488;
    text-decoration: none;
    margin: 0 0.5rem;
}
.footer-links a:hover {
    text-decoration: underline;
}
.entity-pos { background: #dbeafe; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; font-size: 0.85em; }
.entity-neg { background: #fee2e2; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; font-size: 0.85em; }
.entity-per { background: #fef3c7; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; }
.entity-loc { background: #d1fae5; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; }
.entity-org { background: #e0e7ff; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; }
.entity-date { background: #fce7f3; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; }
.switch-bm { background: #dbeafe; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; font-weight: bold; }
.switch-en { background: #fef3c7; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; font-weight: bold; }
.switch-manglish { background: #d1fae5; padding: 2px 6px; border-radius: 4px; margin: 1px; display: inline-block; font-weight: bold; }
"""

# ──────────────────────────────────────────────
# Helper: error wrapper
# ──────────────────────────────────────────────
def safe_run(fn):
    """Wrap tab functions to catch errors gracefully."""
    def wrapper(*args):
        try:
            return fn(*args)
        except Exception as e:
            traceback.print_exc()
            return f"⚠️ Error: {e}"
    return wrapper


# ──────────────────────────────────────────────
# Tab 1: Sentiment Analysis
# ──────────────────────────────────────────────
SENTIMENT_EXAMPLES = [
    "gila best makanan dia, confirm repeat lagi",
    "bodoh punya servis, lambat gila babi",
    "biasa je, takde special mana pun",
    "weh sedih doh tak dapat tiket konsert tu",
    "fuh power gila phone ni, harga pun ok",
]

def analyze_sentiment_tab(text):
    if not text.strip():
        return "⚠️ Enter some text first.", "", ""

    # Overall sentiment
    result = manglish_nlp.sentiment(text)
    label = result.get("label", result.get("sentiment", "unknown"))
    score = result.get("score", result.get("confidence", 0))
    if isinstance(score, float):
        score_str = f"{score:.2%}"
    else:
        score_str = str(score)

    summary_md = f"### 🎭 Sentiment\n**Label:** `{label}`  \n**Score:** `{score_str}`"

    # Aspect sentiment
    aspects = manglish_nlp.aspect_sentiment(text)
    aspect_md = "### 🔍 Aspect Breakdown\n"
    if isinstance(aspects, dict) and aspects:
        for aspect, data in aspects.items():
            if isinstance(data, dict):
                a_label = data.get("label", data.get("sentiment", ""))
                a_score = data.get("score", data.get("confidence", ""))
                if isinstance(a_score, float):
                    a_score = f"{a_score:.2f}"
                aspect_md += f"- **{aspect}**: `{a_label}` ({a_score})\n"
            else:
                aspect_md += f"- **{aspect}**: `{data}`\n"
    else:
        aspect_md += "_No aspect breakdown available._"

    # Full JSON
    try:
        raw = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(result)

    return summary_md, aspect_md, raw


# ──────────────────────────────────────────────
# Tab 2: Text Normalization
# ──────────────────────────────────────────────
NORMALIZE_EXAMPLES = [
    "nk tnya brp sem utk grad kat ump ni",
    "aku dh smpi, jom la mkn",
    "xde duit la, nk pinjam bole?",
    "dia ckp esok ada meeting kat opis",
    "tolong htr brg tu sblm pkl 5 ptg",
]

def normalize_tab(text):
    if not text.strip():
        return "⚠️ Enter some text first.", ""

    normalized = manglish_nlp.normalize(text)
    formal = manglish_nlp.formalize(text)

    # Build side-by-side table
    original_words = text.split()
    norm_words = normalized.split() if isinstance(normalized, str) else [str(normalized)]
    formal_words = formal.split() if isinstance(formal, str) else [str(formal)]

    rows = []
    max_len = max(len(original_words), len(norm_words), len(formal_words))
    for i in range(max_len):
        o = original_words[i] if i < len(original_words) else ""
        n = norm_words[i] if i < len(norm_words) else ""
        f = formal_words[i] if i < len(formal_words) else ""
        changed_n = f"**{n}**" if o.lower() != n.lower() else n
        changed_f = f"**{f}**" if o.lower() != f.lower() else f
        rows.append(f"| `{o}` | {changed_n} | {changed_f} |")

    table = "| Original | Normalized | Formal |\n|----------|-----------|--------|\n" + "\n".join(rows)

    # Full text output
    full_text = f"""### 📝 Normalized
`{normalized}`

### 🎩 Formal BM
`{formal}`"""

    return table, full_text


# ──────────────────────────────────────────────
# Tab 3: Named Entity Recognition
# ──────────────────────────────────────────────
NER_EXAMPLES = [
    "Ahmad pergi ke Kuala Lumpur semalam untuk jumpa CEO Petronas",
    "Dr Siti dari UMP bentang paper kat conference di Penang",
    "PM Anwar jumpa Elon Musk di Putrajaya hari Isnin",
    "Apple buka store baru kat Mid Valley tahun depan",
    "Zafran study kat UMP Pahang guna scholarship MARA",
]

ENTITY_COLORS = {
    "PER": "#f59e0b", "PERSON": "#f59e0b",
    "LOC": "#10b981", "LOCATION": "#10b981", "GPE": "#10b981",
    "ORG": "#6366f1", "ORGANIZATION": "#6366f1",
    "DATE": "#ec4899", "TIME": "#ec4899",
    "EVENT": "#8b5cf6",
    "MISC": "#64748b",
}

def ner_tab(text):
    if not text.strip():
        return "⚠️ Enter some text first.", "", ""

    entities = manglish_nlp.ner_tag(text)

    # Build highlighted text
    highlighted = ""
    entity_table_rows = []

    if isinstance(entities, list) and entities:
        # Build text with colored spans
        for item in entities:
            if isinstance(item, dict):
                word = item.get("word", item.get("text", str(item)))
                tag = item.get("entity", item.get("label", item.get("tag", "O")))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                word, tag = str(item[0]), str(item[1])
            else:
                word, tag = str(item), "O"

            color = ENTITY_COLORS.get(tag.upper(), "#64748b")
            if tag.upper() != "O" and tag.upper() != "MISC":
                highlighted += f'<span style="background:{color}; color:white; padding:2px 6px; border-radius:4px; margin:2px; display:inline-block; font-weight:bold;">{word}<sub>{tag}</sub></span> '
                entity_table_rows.append(f"| {word} | {tag} |")
            else:
                highlighted += f"{word} "
    else:
        highlighted = f"<pre>{str(entities)}</pre>"

    # Table
    table = "| Entity | Type |\n|--------|------|\n"
    table += "\n".join(entity_table_rows) if entity_table_rows else "| _No entities found_ | |"

    # Raw JSON
    try:
        raw = json.dumps(entities, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(entities)

    return highlighted, table, raw


# ──────────────────────────────────────────────
# Tab 4: Translation
# ──────────────────────────────────────────────
TRANSLATION_EXAMPLES = [
    "saya nak pergi kedai beli barang dapur",
    "malaysia has beautiful beaches and food",
    "aku tak faham apa yang dia cakap tadi",
    "the weather is very hot today",
    "jom lepak mamak malam ni",
]

def translate_tab(text, direction):
    if not text.strip():
        return "⚠️ Enter some text first.", "", ""

    if direction == "BM → EN":
        translated = manglish_nlp.to_english(text)
    elif direction == "EN → BM":
        translated = manglish_nlp.to_malay(text)
    else:
        # Auto detect
        translated = manglish_nlp.detect_and_translate(text)

    # Word-level alignment
    src_words = text.split()
    tgt_words = translated.split() if isinstance(translated, str) else [str(translated)]

    alignment_md = "### 🔗 Word Alignment\n"
    # Use word_translate for alignment
    alignment_rows = []
    for w in src_words:
        try:
            wt = manglish_nlp.word_translate(w)
            if isinstance(wt, dict):
                tgt_word = wt.get("translation", wt.get("target", str(wt)))
            elif isinstance(wt, str):
                tgt_word = wt
            else:
                tgt_word = str(wt)
            alignment_rows.append(f"| `{w}` | `{tgt_word}` |")
        except Exception:
            alignment_rows.append(f"| `{w}` | _(no translation)_ |")

    if alignment_rows:
        alignment_md += "| Source | Target |\n|--------|--------|\n" + "\n".join(alignment_rows)
    else:
        alignment_md += "_Word-level alignment unavailable._"

    # Output box
    output_md = f"""### 🔄 Translation ({direction})
`{translated}`

### 📝 Original
`{text}`"""

    # Raw
    try:
        raw = json.dumps({"translated": translated, "direction": direction}, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(translated)

    return output_md, alignment_md, raw


# ──────────────────────────────────────────────
# Tab 5: Language Detection
# ──────────────────────────────────────────────
LANG_DETECT_EXAMPLES = [
    "weh korang nak lepak mana malam ni",
    "I think we should go to the mall tomorrow",
    "aku rasa macam nak makan nasi lemak je",
    "the meeting has been rescheduled to Friday",
    "jom pergi pasar malam beli satay and roti canai",
]

def detect_lang_tab(text):
    if not text.strip():
        return "⚠️ Enter some text first.", ""

    result = manglish_nlp.detect_language(text)

    if isinstance(result, dict):
        lang = result.get("language", result.get("label", "unknown"))
        conf = result.get("confidence", result.get("score", 0))
        if isinstance(conf, float):
            conf_str = f"{conf:.2%}"
        else:
            conf_str = str(conf)
    elif isinstance(result, str):
        lang = result
        conf_str = "N/A"
    else:
        lang = str(result)
        conf_str = "N/A"

    # Dialect detection
    dialect_info = ""
    try:
        dialect = manglish_nlp.detect_dialect(text)
        if isinstance(dialect, dict) and dialect:
            dialect_info = "### 🗣️ Dialect\n"
            for k, v in dialect.items():
                dialect_info += f"- **{k}**: `{v}`\n"
        elif isinstance(dialect, str) and dialect:
            dialect_info = f"### 🗣️ Dialect: `{dialect}`"
    except Exception:
        dialect_info = "### 🗣️ Dialect\n_Dialect detection unavailable._"

    summary = f"""### 🌐 Language Detection
| Property | Value |
|----------|-------|
| **Language** | `{lang}` |
| **Confidence** | `{conf_str}` |

{dialect_info}"""

    try:
        raw = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(result)

    return summary, raw


# ──────────────────────────────────────────────
# Tab 6: Code-Switching
# ──────────────────────────────────────────────
CODE_SWITCH_EXAMPLES = [
    "I want to pergi kedai but it's already tutup",
    "dia sangat happy bila dapat result SPM yesterday",
    "we need to hantar report before the deadline esok",
    "aku rasa this movie memang best gila seriously",
    "the lecturer kata submission is due next week kan",
]

def code_switch_tab(text):
    if not text.strip():
        return "⚠️ Enter some text first.", "", ""

    result = manglish_nlp.segment_text(text)

    # Build highlighted output
    highlighted = ""
    segments_table = ""

    if isinstance(result, list) and result:
        segments_rows = ["| Segment | Language |\n|---------|----------|"]
        for item in result:
            if isinstance(item, dict):
                seg_text = item.get("text", item.get("segment", str(item)))
                seg_lang = item.get("language", item.get("lang", "?"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                seg_text, seg_lang = str(item[0]), str(item[1])
            else:
                seg_text, seg_lang = str(item), "?"

            lang_upper = seg_lang.upper()
            if lang_upper in ("BM", "MALAY", "MS"):
                css_class = "switch-bm"
                display_lang = "BM"
            elif lang_upper in ("EN", "ENGLISH"):
                css_class = "switch-en"
                display_lang = "EN"
            else:
                css_class = "switch-manglish"
                display_lang = seg_lang

            highlighted += f'<span class="{css_class}">{seg_text}</span> '
            segments_rows.append(f"| `{seg_text}` | {display_lang} |")

        segments_table = "\n".join(segments_rows)
    elif isinstance(result, str):
        highlighted = f"<pre>{result}</pre>"
        segments_table = result
    else:
        highlighted = f"<pre>{json.dumps(result, indent=2, ensure_ascii=False, default=str)}</pre>"
        segments_table = str(result)

    # Segment with segment module
    try:
        segment_detail = manglish_nlp.segment(text)
        if isinstance(segment_detail, list):
            detail_rows = []
            for item in segment_detail:
                if isinstance(item, dict):
                    w = item.get("word", item.get("token", ""))
                    l = item.get("language", item.get("lang", ""))
                    detail_rows.append(f"| `{w}` | {l} |")
                elif isinstance(item, (list, tuple)):
                    detail_rows.append(f"| `{item[0]}` | {item[1]} |")
            if detail_rows:
                segments_table += "\n\n### Token-level\n| Token | Language |\n|-------|----------|\n"
                segments_table += "\n".join(detail_rows)
    except Exception:
        pass

    # Raw
    try:
        raw = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(result)

    return highlighted, segments_table, raw


# ──────────────────────────────────────────────
# Tab 7: Full Pipeline
# ──────────────────────────────────────────────
PIPELINE_EXAMPLES = [
    "gila best nasi lemak kat gerai Makcik Kiah tadi, confirm datang lagi",
    "PM Anwar jumpa Elon Musk kat Putrajaya untuk bincang pasal AI investment",
    "aku rasa macam sedih je bila ingat zaman universiti dulu",
    "Dr Siti from UMP present paper pasal climate change at conference in Penang",
    "weh korang dah try new cafe dekat Mid Valley tu? review dia power gila",
]

def full_pipeline_tab(text):
    if not text.strip():
        return "⚠️ Enter some text first."

    results = []

    # 1. Language detection
    try:
        lang = manglish_nlp.detect_language(text)
        if isinstance(lang, dict):
            lang_str = f"**{lang.get('language', '?')}** (confidence: {lang.get('confidence', '?')})"
        else:
            lang_str = str(lang)
        results.append(f"### 🌐 Language Detection\n{lang_str}")
    except Exception as e:
        results.append(f"### 🌐 Language Detection\n⚠️ {e}")

    # 2. Sentiment
    try:
        sent = manglish_nlp.sentiment(text)
        if isinstance(sent, dict):
            sent_str = f"**{sent.get('label', sent.get('sentiment', '?'))}** (score: {sent.get('score', sent.get('confidence', '?'))})"
        else:
            sent_str = str(sent)
        results.append(f"### 🎭 Sentiment\n{sent_str}")
    except Exception as e:
        results.append(f"### 🎭 Sentiment\n⚠️ {e}")

    # 3. Emotion
    try:
        emo = manglish_nlp.detect_emotion(text)
        if isinstance(emo, dict):
            emo_str = f"**{emo.get('emotion', emo.get('label', '?'))}** (confidence: {emo.get('confidence', emo.get('score', '?'))})"
        else:
            emo_str = str(emo)
        results.append(f"### 😊 Emotion\n{emo_str}")
    except Exception as e:
        results.append(f"### 😊 Emotion\n⚠️ {e}")

    # 4. Normalization
    try:
        norm = manglish_nlp.normalize(text)
        results.append(f"### 📝 Normalized\n`{norm}`")
    except Exception as e:
        results.append(f"### 📝 Normalized\n⚠️ {e}")

    # 5. Formalize
    try:
        form = manglish_nlp.formalize(text)
        results.append(f"### 🎩 Formal\n`{form}`")
    except Exception as e:
        results.append(f"### 🎩 Formal\n⚠️ {e}")

    # 6. NER
    try:
        ner = manglish_nlp.ner_tag(text)
        if isinstance(ner, list) and ner:
            entities = []
            for item in ner:
                if isinstance(item, dict):
                    w = item.get("word", item.get("text", ""))
                    t = item.get("entity", item.get("label", item.get("tag", "O")))
                elif isinstance(item, (list, tuple)):
                    w, t = str(item[0]), str(item[1])
                else:
                    w, t = str(item), "?"
                if t.upper() not in ("O", "MISC"):
                    entities.append(f"`{w}` [{t}]")
            if entities:
                results.append(f"### 🏷️ Entities\n{', '.join(entities)}")
            else:
                results.append("### 🏷️ Entities\n_No entities found._")
        else:
            results.append("### 🏷️ Entities\n_No entities found._")
    except Exception as e:
        results.append(f"### 🏷️ Entities\n⚠️ {e}")

    # 7. Keywords
    try:
        kw = manglish_nlp.extract_keywords(text)
        if isinstance(kw, list):
            kw_str = ", ".join([f"`{k}`" for k in kw[:10]])
        else:
            kw_str = str(kw)
        results.append(f"### 🔑 Keywords\n{kw_str}")
    except Exception as e:
        results.append(f"### 🔑 Keywords\n⚠️ {e}")

    # 8. Translation
    try:
        trans = manglish_nlp.detect_and_translate(text)
        results.append(f"### 🔄 Translation\n`{trans}`")
    except Exception as e:
        results.append(f"### 🔄 Translation\n⚠️ {e}")

    # 9. Sarcasm
    try:
        sarc = manglish_nlp.detect_sarcasm(text)
        if isinstance(sarc, dict):
            sarc_str = f"**{sarc.get('label', sarc.get('sarcastic', '?'))}** (score: {sarc.get('score', sarc.get('confidence', '?'))})"
        elif isinstance(sarc, bool):
            sarc_str = "Yes" if sarc else "No"
        else:
            sarc_str = str(sarc)
        results.append(f"### 🙃 Sarcasm\n{sarc_str}")
    except Exception as e:
        results.append(f"### 🙃 Sarcasm\n⚠️ {e}")

    # 10. Topic
    try:
        topic = manglish_nlp.classify_topic(text)
        if isinstance(topic, dict):
            topic_str = f"**{topic.get('topic', topic.get('label', '?'))}** (confidence: {topic.get('confidence', topic.get('score', '?'))})"
        else:
            topic_str = str(topic)
        results.append(f"### 📂 Topic\n{topic_str}")
    except Exception as e:
        results.append(f"### 📂 Topic\n⚠️ {e}")

    return "\n\n".join(results)


# ──────────────────────────────────────────────
# Build Gradio UI
# ──────────────────────────────────────────────
with gr.Blocks(
    title="Manglish NLP Demo",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"),
) as demo:

    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🇲🇾 Manglish NLP Demo</h1>
        <p>Natural Language Processing for Malaysian Manglish — zero ML dependencies, pure rule-based</p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Sentiment ──
        with gr.Tab("🎭 Sentiment"):
            with gr.Row():
                with gr.Column(scale=3):
                    sent_input = gr.Textbox(
                        label="Manglish Text",
                        placeholder="Type Manglish text here...",
                        lines=3,
                    )
                    sent_example = gr.Dropdown(
                        choices=SENTIMENT_EXAMPLES,
                        label="Example Inputs",
                        value=None,
                    )
                    sent_btn = gr.Button("Analyze Sentiment", variant="primary")
                with gr.Column(scale=4):
                    sent_output = gr.Markdown(label="Sentiment Result", elem_classes=["result-box"])
                    sent_aspect = gr.Markdown(label="Aspect Breakdown", elem_classes=["result-box"])
            with gr.Accordion("Raw JSON Output", open=False):
                sent_raw = gr.Code(label="Raw", language="json")
            sent_btn.click(
                safe_run(analyze_sentiment_tab),
                inputs=[sent_input],
                outputs=[sent_output, sent_aspect, sent_raw],
            )
            sent_example.change(lambda x: x, inputs=[sent_example], outputs=[sent_input])

        # ── Tab 2: Normalization ──
        with gr.Tab("📝 Normalization"):
            with gr.Row():
                with gr.Column(scale=3):
                    norm_input = gr.Textbox(
                        label="Manglish Text",
                        placeholder="Type text with shortforms...",
                        lines=3,
                    )
                    norm_example = gr.Dropdown(
                        choices=NORMALIZE_EXAMPLES,
                        label="Example Inputs",
                        value=None,
                    )
                    norm_btn = gr.Button("Normalize", variant="primary")
                with gr.Column(scale=4):
                    norm_table = gr.Markdown(label="Word Comparison", elem_classes=["result-box"])
                    norm_text = gr.Markdown(label="Full Text", elem_classes=["result-box"])
            norm_btn.click(
                safe_run(normalize_tab),
                inputs=[norm_input],
                outputs=[norm_table, norm_text],
            )
            norm_example.change(lambda x: x, inputs=[norm_example], outputs=[norm_input])

        # ── Tab 3: NER ──
        with gr.Tab("🏷️ NER"):
            with gr.Row():
                with gr.Column(scale=3):
                    ner_input = gr.Textbox(
                        label="Text",
                        placeholder="Enter text with names, places, orgs...",
                        lines=3,
                    )
                    ner_example = gr.Dropdown(
                        choices=NER_EXAMPLES,
                        label="Example Inputs",
                        value=None,
                    )
                    ner_btn = gr.Button("Detect Entities", variant="primary")
                with gr.Column(scale=4):
                    ner_highlighted = gr.HTML(label="Highlighted Entities", elem_classes=["result-box"])
                    ner_table = gr.Markdown(label="Entity Table", elem_classes=["result-box"])
            with gr.Accordion("Raw JSON Output", open=False):
                ner_raw = gr.Code(label="Raw", language="json")
            ner_btn.click(
                safe_run(ner_tab),
                inputs=[ner_input],
                outputs=[ner_highlighted, ner_table, ner_raw],
            )
            ner_example.change(lambda x: x, inputs=[ner_example], outputs=[ner_input])

        # ── Tab 4: Translation ──
        with gr.Tab("🔄 Translation"):
            with gr.Row():
                with gr.Column(scale=3):
                    trans_input = gr.Textbox(
                        label="Text to Translate",
                        placeholder="Enter BM or EN text...",
                        lines=3,
                    )
                    trans_direction = gr.Radio(
                        choices=["BM → EN", "EN → BM", "Auto Detect"],
                        value="BM → EN",
                        label="Direction",
                    )
                    trans_example = gr.Dropdown(
                        choices=TRANSLATION_EXAMPLES,
                        label="Example Inputs",
                        value=None,
                    )
                    trans_btn = gr.Button("Translate", variant="primary")
                with gr.Column(scale=4):
                    trans_output = gr.Markdown(label="Translation", elem_classes=["result-box"])
                    trans_align = gr.Markdown(label="Word Alignment", elem_classes=["result-box"])
            with gr.Accordion("Raw JSON Output", open=False):
                trans_raw = gr.Code(label="Raw", language="json")
            trans_btn.click(
                safe_run(translate_tab),
                inputs=[trans_input, trans_direction],
                outputs=[trans_output, trans_align, trans_raw],
            )
            trans_example.change(lambda x: x, inputs=[trans_example], outputs=[trans_input])

        # ── Tab 5: Language Detection ──
        with gr.Tab("🌐 Language Detection"):
            with gr.Row():
                with gr.Column(scale=3):
                    lang_input = gr.Textbox(
                        label="Text",
                        placeholder="Enter any text...",
                        lines=3,
                    )
                    lang_example = gr.Dropdown(
                        choices=LANG_DETECT_EXAMPLES,
                        label="Example Inputs",
                        value=None,
                    )
                    lang_btn = gr.Button("Detect Language", variant="primary")
                with gr.Column(scale=4):
                    lang_output = gr.Markdown(label="Detection Result", elem_classes=["result-box"])
            with gr.Accordion("Raw JSON Output", open=False):
                lang_raw = gr.Code(label="Raw", language="json")
            lang_btn.click(
                safe_run(detect_lang_tab),
                inputs=[lang_input],
                outputs=[lang_output, lang_raw],
            )
            lang_example.change(lambda x: x, inputs=[lang_example], outputs=[lang_input])

        # ── Tab 6: Code-Switching ──
        with gr.Tab("🔀 Code-Switching"):
            with gr.Row():
                with gr.Column(scale=3):
                    cs_input = gr.Textbox(
                        label="Mixed Language Text",
                        placeholder="Enter code-switched text...",
                        lines=3,
                    )
                    cs_example = gr.Dropdown(
                        choices=CODE_SWITCH_EXAMPLES,
                        label="Example Inputs",
                        value=None,
                    )
                    cs_btn = gr.Button("Detect Switches", variant="primary")
                with gr.Column(scale=4):
                    cs_highlighted = gr.HTML(label="Highlighted Switches", elem_classes=["result-box"])
                    cs_table = gr.Markdown(label="Segment Table", elem_classes=["result-box"])
            with gr.Accordion("Raw JSON Output", open=False):
                cs_raw = gr.Code(label="Raw", language="json")
            cs_btn.click(
                safe_run(code_switch_tab),
                inputs=[cs_input],
                outputs=[cs_highlighted, cs_table, cs_raw],
            )
            cs_example.change(lambda x: x, inputs=[cs_example], outputs=[cs_input])

        # ── Tab 7: Full Pipeline ──
        with gr.Tab("⚡ Full Pipeline"):
            with gr.Row():
                with gr.Column(scale=3):
                    pipe_input = gr.Textbox(
                        label="Text",
                        placeholder="Enter any text for full analysis...",
                        lines=3,
                    )
                    pipe_example = gr.Dropdown(
                        choices=PIPELINE_EXAMPLES,
                        label="Example Inputs",
                        value=None,
                    )
                    pipe_btn = gr.Button("Run Full Pipeline", variant="primary")
                with gr.Column(scale=4):
                    pipe_output = gr.Markdown(label="Full Analysis Dashboard", elem_classes=["result-box"])
            pipe_btn.click(
                safe_run(full_pipeline_tab),
                inputs=[pipe_input],
                outputs=[pipe_output],
            )
            pipe_example.change(lambda x: x, inputs=[pipe_example], outputs=[pipe_input])

    # Footer
    gr.HTML("""
    <div class="footer-links">
        <p>
            <a href="https://github.com/zafra/manglish-nlp" target="_blank">GitHub</a> |
            <a href="https://manglish-nlp.readthedocs.io" target="_blank">Docs</a> |
            <a href="https://pypi.org/project/manglish-nlp/" target="_blank">PyPI</a>
        </p>
        <p>Powered by <strong>manglish-nlp</strong> v""" + manglish_nlp.__version__ + """ — Built with ❤️ for Malaysia</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
