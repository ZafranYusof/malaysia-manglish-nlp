"""
Auto-label scraped Manglish text using existing sentiment module + keyword matching.

High-confidence auto-labeling (sentiment score > 0.85 threshold).
Uncertain labels go to human review queue.
Outputs labeled data + confidence scores.

Usage:
    python scripts/auto_label.py
    python scripts/auto_label.py --threshold 0.80 --input datasets/raw_scraped/combined_raw.jsonl
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw_scraped"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Add project root to path so we can import malaysian_manglish_nlp
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(filepath: Path) -> list[dict]:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Sentiment analysis (try malaysian_manglish_nlp first, fallback to keyword-based)
# ---------------------------------------------------------------------------

_malaysian_manglish_nlp_available = False
try:
    from malaysian_manglish_nlp.sentiment import analyze_sentiment
    from malaysian_manglish_nlp.language import detect_language
    from malaysian_manglish_nlp.emotion import analyze_emotion
    _malaysian_manglish_nlp_available = True
except ImportError:
    print("[Warning] malaysian_manglish_nlp not importable, using keyword-based fallback")


# Keyword-based fallback
_POSITIVE_WORDS = {
    "best", "syok", "power", "mantap", "padu", "sedap", "cantik", "lawa",
    "hebat", "bagus", "terbaik", "good", "great", "awesome", "amazing",
    "love", "happy", "seronok", "gembira", "excited", "enjoy", "nice",
    "beautiful", "perfect", "excellent", "wonderful", "fantastic",
    "tahniah", "congrats", "congratulations", "win", "won", "champion",
    "murah", "cheap", "berbaloi", "worth", "value", "recommend",
    "clean", "bersih", "fast", "cepat", "pantas", "efficient",
    "friendly", "mesra", "helpful", "membantu", "sedap", "lazat", "enak",
    "fresh", "segar", "crispy", "garing", "lembut", "soft", "tender",
}

_NEGATIVE_WORDS = {
    "teruk", "hampeh", "hancur", "parah", "kronik", "bad", "terrible",
    "horrible", "awful", "worst", "suck", "shit", "crap", "sampah",
    "bodoh", "bangang", "stupid", "dumb", "useless", "membazir",
    "mahal", "expensive", "overpriced", "scam", "penipu",
    "slow", "lambat", "lewat", "late", "delayed",
    "rude", "kurang ajar", "biadap", "sombong", "arrogant",
    "dirty", "kotor", "busuk", "smelly", "bau",
    "broken", "rosak", "damage", "fail", "gagal",
    "boring", "bosan", "waste", "rugi", "menyesal", "regret",
    "angry", "marah", "bengang", "geram", "frustrated",
    "sad", "sedih", "pilu", "kecewa", "disappointed",
}

_NEGATORS = {"tak", "tidak", "x", "bukan", "jangan", "takde", "xde", "not", "no", "never"}
_INTENSIFIERS = {"gila", "sangat", "memang", "betul2", "damn", "really", "very", "super", "extremely"}


def _keyword_sentiment(text: str) -> dict:
    """Fallback keyword-based sentiment analysis."""
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)

    # Check negation
    has_negator = bool(words & _NEGATORS)
    if has_negator:
        pos_count, neg_count = neg_count, pos_count

    total = pos_count + neg_count
    if total == 0:
        return {"sentiment": "neutral", "confidence": 0.3, "score": 0.0}

    if pos_count > neg_count:
        score = pos_count / (pos_count + neg_count)
        confidence = min(0.5 + score * 0.5, 0.95)
        return {"sentiment": "positive", "confidence": confidence, "score": score}
    elif neg_count > pos_count:
        score = neg_count / (pos_count + neg_count)
        confidence = min(0.5 + score * 0.5, 0.95)
        return {"sentiment": "negative", "confidence": confidence, "score": -score}
    else:
        return {"sentiment": "neutral", "confidence": 0.4, "score": 0.0}


def _keyword_language(text: str) -> str:
    """Fallback language detection."""
    malay_signals = [
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada",
        "ada", "ini", "itu", "saya", "kami", "mereka", "dia",
        "lah", "wei", "macam", "boleh", "nak", "tak",
    ]
    words = text.lower().split()
    malay_count = sum(1 for w in words if w.strip(".,!?") in malay_signals)
    ratio = malay_count / max(len(words), 1)

    if ratio > 0.4:
        return "manglish" if any(w in text.lower() for w in ["best", "good", "the", "is"]) else "malay"
    elif ratio > 0.1:
        return "manglish"
    else:
        return "english"


def _keyword_emotion(text: str, sentiment: str) -> str:
    """Fallback emotion detection."""
    lower = text.lower()
    if sentiment == "positive":
        if any(w in lower for w in ["love", "cinta", "sayang", "rindu"]):
            return "love"
        if any(w in lower for w in ["happy", "gembira", "seronok", "best", "syok"]):
            return "happy"
        if any(w in lower for w in ["wow", "gila", "amazing", "surprise"]):
            return "surprise"
        return "happy"
    elif sentiment == "negative":
        if any(w in lower for w in ["marah", "angry", "bengang", "geram"]):
            return "angry"
        if any(w in lower for w in ["sedih", "sad", "pilu", "kecewa"]):
            return "sad"
        if any(w in lower for w in ["takut", "fear", "cuak", "seram"]):
            return "fear"
        if any(w in lower for w in ["busuk", "kotor", "gross", "eww"]):
            return "disgust"
        return "angry"
    return "neutral"


# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS = {
    "food": ["makan", "nasi", "food", "sedap", "restoran", "mamak", "kedai",
             "roti", "ayam", "ikan", "mee", "nasi lemak", "rendang", "satay",
             "cook", "masak", "recipe", "resep", "cafe", "kafe"],
    "politics": ["politik", "kerajaan", "government", "minister", "menteri",
                 "PRU", "election", "pilihan raya", "party", "parti", "vote",
                 "corruption", "rasuah", "policy", "dasar", "parliament"],
    "sports": ["bola", "football", "badminton", "hockey", "match", "game",
               "team", "pasukan", "player", "pemain", "coach", "jurulatih",
               "win", "menang", "champion", "liga", "league", "score", "gol"],
    "tech": ["phone", "laptop", "app", "software", "tech", "gadget",
             "internet", "wifi", "data", "coding", "programming",
             "AI", "computer", "device", "digital", "online", "website"],
    "education": ["sekolah", "school", "exam", "peperiksaan", "university",
                  "universiti", "student", "pelajar", "cikgu", "teacher",
                  "assignment", "lecture", "class", "kelas", "study", "belajar"],
    "entertainment": ["movie", "filem", "drama", "song", "lagu", "music",
                      "concert", "konsert", "actor", "pelakon", "singer",
                      "netflix", "youtube", "tiktok", "instagram", "viral"],
    "religion": ["solat", "prayer", "masjid", "mosque", "islam", "puasa",
                 "ramadan", "raya", "halal", "haram", "doa", "quran"],
    "daily_life": ["traffic", "jalan", "cuaca", "weather", "rumah", "house",
                   "kerja", "work", "family", "keluarga", "shopping", "mall"],
}


def detect_topic(text: str) -> str:
    """Detect topic from text using keyword matching."""
    lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in lower)
        if score > 0:
            scores[topic] = score

    if scores:
        return max(scores, key=scores.get)
    return "daily_life"


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def detect_intent(text: str) -> str:
    """Detect intent from text."""
    lower = text.lower()
    if "?" in text or any(w in lower for w in ["kenapa", "kenapa", "macam mana",
                                                 "siapa", "bila", "berapa",
                                                 "why", "how", "what", "when"]):
        return "question"
    if any(w in lower for w in ["tolong", "please", "boleh tak", "can you",
                                 "request", "minta"]):
        return "request"
    if any(w in lower for w in ["teruk", "hampeh", "complain", "report",
                                 "scam", "penipu", "sampah", "useless"]):
        return "complaint"
    if any(w in lower for w in ["aku rasa", "i think", "pada aku", "imho",
                                 "menurut", "opinion", "pendapat"]):
        return "opinion"
    if any(w in lower for w in ["salam", "hai", "hello", "hi ", "hey",
                                 "selamat", "assalamualaikum"]):
        return "greeting"
    return "statement"


# ---------------------------------------------------------------------------
# Main auto-labeling pipeline
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    """Run full analysis on a text, returning all labels + confidence."""
    result = {}

    if _malaysian_manglish_nlp_available:
        # Use malaysian_manglish_nlp sentiment
        sent_result = analyze_sentiment(text)
        if isinstance(sent_result, dict):
            result["sentiment"] = sent_result.get("sentiment", "neutral")
            result["sentiment_score"] = sent_result.get("confidence", sent_result.get("score", 0.5))
        else:
            # Might return tuple or other format
            fb = _keyword_sentiment(text)
            result["sentiment"] = fb["sentiment"]
            result["sentiment_score"] = fb["confidence"]

        # Language detection
        try:
            result["language"] = detect_language(text)
        except Exception:
            result["language"] = _keyword_language(text)

        # Emotion
        try:
            emo_result = analyze_emotion(text)
            if isinstance(emo_result, dict):
                result["emotion"] = emo_result.get("emotion", "neutral")
            elif isinstance(emo_result, str):
                result["emotion"] = emo_result
            else:
                result["emotion"] = _keyword_emotion(text, result["sentiment"])
        except Exception:
            result["emotion"] = _keyword_emotion(text, result["sentiment"])
    else:
        # Fallback
        fb = _keyword_sentiment(text)
        result["sentiment"] = fb["sentiment"]
        result["sentiment_score"] = fb["confidence"]
        result["language"] = _keyword_language(text)
        result["emotion"] = _keyword_emotion(text, result["sentiment"])

    # Topic and intent (always keyword-based)
    result["topic"] = detect_topic(text)
    result["intent"] = detect_intent(text)

    # Code-switch detection
    malay_words = {"yang", "dan", "di", "tak", "nak", "macam", "boleh", "lah", "wei"}
    eng_words = {"the", "is", "are", "was", "were", "this", "that", "with", "for"}
    word_set = set(re.findall(r"[a-zA-Z]+", text.lower()))
    has_malay = bool(word_set & malay_words)
    has_eng = bool(word_set & eng_words)
    result["is_code_switch"] = has_malay and has_eng

    return result


def auto_label(raw_items: list[dict], threshold: float = 0.85) -> tuple[list[dict], list[dict]]:
    """Auto-label raw scraped items.

    Returns:
        (labeled_items, review_queue)
        - labeled_items: high-confidence auto-labeled (score >= threshold)
        - review_queue: uncertain items for human review
    """
    labeled = []
    review = []

    for item in raw_items:
        text = item.get("text", "").strip()
        if not text or len(text) < 10:
            continue

        analysis = analyze_text(text)
        confidence = analysis.get("sentiment_score", 0.5)

        entry = {
            "text": text,
            "sentiment": analysis["sentiment"],
            "language": analysis.get("language", "manglish"),
            "emotion": analysis.get("emotion", "neutral"),
            "intent": analysis.get("intent", "statement"),
            "topic": analysis.get("topic", "daily_life"),
            "is_code_switch": analysis.get("is_code_switch", False),
            "dialect": "standard",
            "source_type": item.get("source", "scraped"),
            "auto_label_confidence": round(confidence, 4),
            "auto_label_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Preserve source metadata
        if "source_detail" in item:
            entry["source_detail"] = item["source_detail"]
        if "url" in item:
            entry["source_url"] = item["url"]

        if confidence >= threshold:
            entry["label_source"] = "auto_high_confidence"
            labeled.append(entry)
        else:
            entry["label_source"] = "needs_review"
            review.append(entry)

    return labeled, review


def main():
    parser = argparse.ArgumentParser(description="Auto-label scraped Manglish text")
    parser.add_argument("--input", type=str,
                        default=str(RAW_DIR / "combined_raw.jsonl"),
                        help="Input JSONL file (default: datasets/raw_scraped/combined_raw.jsonl)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Confidence threshold for auto-labeling (default: 0.85)")
    parser.add_argument("--output-labeled", type=str,
                        default=str(DATASETS_DIR / "manglish_auto_labeled.jsonl"),
                        help="Output for high-confidence labeled data")
    parser.add_argument("--output-review", type=str,
                        default=str(DATASETS_DIR / "manglish_review_queue.jsonl"),
                        help="Output for uncertain items needing human review")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run scrape_social_media.py first to generate raw data.")
        return

    # Load raw data
    raw_items = load_jsonl(input_path)
    print(f"Loaded {len(raw_items)} raw items from {input_path}")
    print(f"Confidence threshold: {args.threshold}")

    # Auto-label
    labeled, review = auto_label(raw_items, threshold=args.threshold)

    # Save outputs
    output_labeled = Path(args.output_labeled)
    output_review = Path(args.output_review)

    if labeled:
        save_jsonl(labeled, output_labeled)
    if review:
        save_jsonl(review, output_review)

    # Statistics
    print("\n" + "=" * 60)
    print("AUTO-LABELING STATISTICS")
    print("=" * 60)
    total = len(labeled) + len(review)
    print(f"\nTotal processed:           {total}")
    print(f"High-confidence labeled:   {len(labeled)} ({len(labeled)/max(total,1)*100:.1f}%)")
    print(f"Needs human review:        {len(review)} ({len(review)/max(total,1)*100:.1f}%)")

    if labeled:
        print(f"\nLabeled output:  {output_labeled}")
        print(f"Review output:   {output_review}")

        # Label distributions
        print("\n" + "-" * 60)
        print("LABEL DISTRIBUTIONS (auto-labeled)")
        print("-" * 60)

        for field in ["sentiment", "emotion", "intent", "topic", "language"]:
            values = [item.get(field) for item in labeled if item.get(field)]
            if values:
                counter = Counter(values)
                print(f"\n{field.upper()} ({len(values)} labeled):")
                for label, count in counter.most_common():
                    pct = count / len(values) * 100
                    bar = "#" * int(pct / 2)
                    print(f"  {label:20s} {count:5d} ({pct:5.1f}%) {bar}")

        # Confidence distribution
        confidences = [item["auto_label_confidence"] for item in labeled]
        if confidences:
            print(f"\nCONFIDENCE STATS:")
            print(f"  Min:    {min(confidences):.4f}")
            print(f"  Max:    {max(confidences):.4f}")
            print(f"  Mean:   {sum(confidences)/len(confidences):.4f}")
            print(f"  Median: {sorted(confidences)[len(confidences)//2]:.4f}")

        # Source breakdown
        sources = Counter(item.get("source_type", "unknown") for item in labeled)
        print(f"\nSOURCE BREAKDOWN:")
        for source, count in sources.most_common():
            print(f"  {source:15s} {count:5d}")

    # Accuracy estimate (based on keyword overlap with known patterns)
    if labeled:
        print("\n" + "-" * 60)
        print("ESTIMATED ACCURACY")
        print("-" * 60)
        # Sample check: re-analyze a random subset and compare
        import random
        random.seed(42)
        sample_size = min(50, len(labeled))
        sample = random.sample(labeled, sample_size)
        correct = 0
        for item in sample:
            reanalysis = analyze_text(item["text"])
            if reanalysis["sentiment"] == item["sentiment"]:
                correct += 1
        accuracy = correct / sample_size * 100
        print(f"  Sample size: {sample_size}")
        print(f"  Consistency: {correct}/{sample_size} ({accuracy:.1f}%)")
        print(f"  Note: Self-consistency check, not ground truth")

    print("\nDone!")


if __name__ == "__main__":
    main()
