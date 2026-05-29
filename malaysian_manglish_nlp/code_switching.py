"""
Code-switching detection and analysis for Manglish text.

Detects language alternation between Bahasa Melayu and English,
identifies switch points, and classifies switching patterns.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# === MALAY WORD LIST (500+ common BM words) ===
MALAY_WORDS = {
    # Pronouns
    "aku", "saya", "kamu", "kau", "awak", "dia", "beliau", "mereka", "kami",
    "kita", "engkau", "anda", "hamba", "patik", "beta",
    # Verbs
    "makan", "minum", "tidur", "bangun", "pergi", "datang", "balik", "pulang",
    "buat", "kerja", "belajar", "ajar", "tulis", "baca", "dengar", "tengok",
    "lihat", "nampak", "cakap", "kata", "tanya", "jawab", "fikir", "rasa",
    "suka", "benci", "sayang", "cinta", "rindu", "takut", "berani", "cuba",
    "boleh", "dapat", "kena", "harus", "mesti", "patut", "perlu", "nak",
    "hendak", "mahu", "jalan", "lari", "duduk", "berdiri", "ambil", "letak",
    "bagi", "hantar", "terima", "beli", "jual", "bayar", "hutang", "simpan",
    "guna", "pakai", "buka", "tutup", "masuk", "keluar", "naik", "turun",
    "angkat", "jatuh", "lempar", "tangkap", "pegang", "lepas", "ikat", "potong",
    "masak", "goreng", "rebus", "bakar", "cuci", "basuh", "sapu", "lap",
    "jahit", "lipat", "gosok", "sidai", "jemur", "kering", "basah", "panas",
    "sejuk", "tolong", "bantu", "halang", "larang", "benarkan", "izin",
    "tunggu", "sampai", "tiba", "mulai", "mula", "habis", "tamat", "selesai",
    "gagal", "berjaya", "menang", "kalah", "lawan", "gaduh", "bergaduh",
    "bermain", "main", "nyanyi", "tari", "menari", "lukis", "gambar",
    "rakam", "tangkap", "cari", "jumpa", "hilang", "tinggal", "pindah",
    # Nouns
    "rumah", "sekolah", "kedai", "pasar", "hospital", "masjid", "surau",
    "gereja", "kuil", "pejabat", "kilang", "ladang", "sawah", "kebun",
    "hutan", "sungai", "laut", "pantai", "gunung", "bukit", "tanah",
    "langit", "awan", "hujan", "angin", "ribut", "panas", "matahari",
    "bulan", "bintang", "pokok", "bunga", "daun", "buah", "sayur",
    "ikan", "ayam", "lembu", "kambing", "kucing", "anjing", "burung",
    "ular", "serangga", "nyamuk", "lalat", "semut", "lipas", "tikus",
    "kereta", "motosikal", "bas", "lori", "kapal", "bot", "pesawat",
    "jalan", "lorong", "lebuh", "taman", "kampung", "bandar", "negeri",
    "negara", "dunia", "bumi", "air", "api", "udara", "batu", "pasir",
    "makanan", "minuman", "nasi", "roti", "mee", "kuih", "kek", "gula",
    "garam", "minyak", "cuka", "kicap", "sambal", "santan", "susu",
    "teh", "kopi", "baju", "seluar", "kasut", "stokin", "topi", "tudung",
    "cermin", "meja", "kerusi", "katil", "almari", "pintu", "tingkap",
    "dinding", "lantai", "siling", "tangga", "bumbung", "pagar", "halaman",
    "bilik", "dapur", "tandas", "garaj", "wang", "duit", "ringgit", "sen",
    "harga", "murah", "mahal", "percuma", "diskaun", "untung", "rugi",
    "masa", "waktu", "hari", "minggu", "bulan", "tahun", "pagi", "petang",
    "malam", "tengahari", "subuh", "zohor", "asar", "maghrib", "isyak",
    "isnin", "selasa", "rabu", "khamis", "jumaat", "sabtu", "ahad",
    "orang", "lelaki", "perempuan", "budak", "kanak", "bayi", "remaja",
    "dewasa", "tua", "muda", "abang", "kakak", "adik", "emak", "ayah",
    "ibu", "bapa", "nenek", "atuk", "datuk", "pakcik", "makcik",
    "sepupu", "anak", "cucu", "suami", "isteri", "kawan", "musuh",
    "jiran", "guru", "murid", "pelajar", "doktor", "polis", "tentera",
    "kepala", "mata", "hidung", "mulut", "telinga", "tangan", "kaki",
    "badan", "perut", "dada", "belakang", "bahu", "lutut", "jari",
    "kuku", "rambut", "kulit", "tulang", "darah", "hati", "otak",
    # Adjectives
    "besar", "kecil", "panjang", "pendek", "tinggi", "rendah", "lebar",
    "sempit", "tebal", "nipis", "berat", "ringan", "keras", "lembut",
    "kasar", "halus", "tajam", "tumpul", "baru", "lama", "muda", "tua",
    "cantik", "hodoh", "kemas", "kotor", "bersih", "elok", "baik",
    "jahat", "pandai", "bodoh", "rajin", "malas", "kaya", "miskin",
    "sihat", "sakit", "kuat", "lemah", "cepat", "lambat", "awal", "lewat",
    "betul", "salah", "benar", "palsu", "senang", "susah", "mudah",
    "sukar", "gembira", "sedih", "marah", "tenang", "gelap", "terang",
    "putih", "hitam", "merah", "biru", "hijau", "kuning", "coklat",
    # Adverbs / Function words
    "sangat", "amat", "terlalu", "agak", "sedikit", "banyak", "semua",
    "setiap", "selalu", "kadang", "jarang", "tidak", "tak", "bukan",
    "belum", "sudah", "dah", "akan", "sedang", "tengah", "masih", "lagi",
    "juga", "pun", "sahaja", "saja", "hanya", "cuma", "memang", "sungguh",
    "pasti", "mungkin", "barangkali", "kalau", "jika", "sekiranya",
    "walaupun", "meskipun", "tetapi", "tapi", "namun", "atau", "dan",
    "serta", "dengan", "untuk", "kepada", "daripada", "tentang", "mengenai",
    "dalam", "luar", "atas", "bawah", "depan", "belakang", "sebelah",
    "antara", "sekitar", "dekat", "jauh", "sini", "situ", "sana",
    "ini", "itu", "yang", "apa", "siapa", "mana", "bila", "kenapa",
    "mengapa", "bagaimana", "macam", "berapa", "ke", "di", "dari",
    "pada", "oleh", "sejak", "hingga", "sampai", "supaya", "agar",
    "sebab", "kerana", "pasal", "lepas", "selepas", "sebelum", "semasa",
    # Numbers
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan",
    "sembilan", "sepuluh", "sebelas", "seratus", "seribu", "sejuta",
    # Common informal/shortened
    "nak", "tak", "dah", "je", "jer", "ja", "kot", "kan", "lah", "la",
    "eh", "ah", "oh", "wei", "weh", "woi", "oi", "hah", "hmm",
    "macam", "camtu", "camni", "gitu", "gini", "tu", "ni", "tuh",
    "nanti", "jap", "kejap", "sekejap", "dulu", "lepas", "pastu",
    "sebab", "sbb", "tgk", "tengok", "kat", "dekat", "sini", "situ",
    "sikit", "banyak", "semua", "habis", "dah", "belum", "lagi",
}

# === ENGLISH WORD LIST (500+ common EN words) ===
ENGLISH_WORDS = {
    # Pronouns
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves", "this", "that",
    "these", "those", "who", "whom", "which", "what", "whose",
    # Verbs
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "must", "need", "dare", "ought", "go", "goes", "went", "gone",
    "going", "come", "came", "coming", "get", "got", "getting",
    "make", "made", "making", "take", "took", "taken", "taking",
    "give", "gave", "given", "giving", "know", "knew", "known",
    "think", "thought", "thinking", "see", "saw", "seen", "seeing",
    "want", "wanted", "wanting", "look", "looked", "looking",
    "use", "used", "using", "find", "found", "finding",
    "tell", "told", "telling", "ask", "asked", "asking",
    "work", "worked", "working", "seem", "seemed", "seeming",
    "feel", "felt", "feeling", "try", "tried", "trying",
    "leave", "left", "leaving", "call", "called", "calling",
    "keep", "kept", "keeping", "let", "put", "run", "ran",
    "running", "say", "said", "saying", "turn", "turned",
    "start", "started", "show", "showed", "shown", "hear", "heard",
    "play", "played", "move", "moved", "live", "lived", "believe",
    "bring", "brought", "happen", "happened", "write", "wrote",
    "sit", "sat", "stand", "stood", "lose", "lost", "pay", "paid",
    "meet", "met", "include", "continue", "set", "learn", "learned",
    "change", "changed", "lead", "led", "understand", "understood",
    "watch", "watched", "follow", "followed", "stop", "stopped",
    "speak", "spoke", "spoken", "read", "allow", "add", "spend",
    "spent", "grow", "grew", "open", "opened", "walk", "walked",
    "win", "won", "teach", "taught", "offer", "remember", "love",
    "consider", "appear", "buy", "bought", "wait", "serve", "die",
    "send", "sent", "expect", "build", "built", "stay", "fall", "fell",
    "cut", "reach", "kill", "remain", "suggest", "raise", "pass",
    "sell", "sold", "require", "report", "decide", "pull", "pulled",
    # Nouns
    "time", "year", "people", "way", "day", "man", "woman", "child",
    "children", "world", "life", "hand", "part", "place", "case",
    "week", "company", "system", "program", "question", "work",
    "government", "number", "night", "point", "home", "water",
    "room", "mother", "area", "money", "story", "fact", "month",
    "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend",
    "father", "power", "hour", "game", "line", "end", "member",
    "law", "car", "city", "community", "name", "president", "team",
    "minute", "idea", "body", "information", "back", "parent", "face",
    "others", "level", "office", "door", "health", "person", "art",
    "war", "history", "party", "result", "change", "morning",
    "reason", "research", "girl", "guy", "moment", "air", "teacher",
    "force", "education", "food", "phone", "computer", "laptop",
    "internet", "website", "email", "message", "meeting", "project",
    "problem", "solution", "answer", "school", "class", "student",
    "family", "brother", "sister", "baby", "dog", "cat", "bird",
    "fish", "tree", "flower", "garden", "park", "street", "road",
    "building", "shop", "store", "market", "restaurant", "hotel",
    "airport", "station", "hospital", "church", "bank", "library",
    "movie", "music", "song", "picture", "photo", "video", "news",
    "paper", "table", "chair", "bed", "window", "wall", "floor",
    "kitchen", "bathroom", "bedroom", "clothes", "shirt", "shoes",
    "bag", "key", "card", "ticket", "price", "breakfast", "lunch",
    "dinner", "coffee", "tea", "beer", "wine", "chicken", "rice",
    # Adjectives
    "good", "new", "first", "last", "long", "great", "little", "own",
    "other", "old", "right", "big", "high", "different", "small",
    "large", "next", "early", "young", "important", "few", "public",
    "bad", "same", "able", "free", "sure", "true", "full", "special",
    "easy", "clear", "recent", "certain", "personal", "open", "red",
    "difficult", "available", "likely", "short", "single", "medical",
    "current", "wrong", "private", "past", "foreign", "fine", "common",
    "poor", "natural", "significant", "similar", "hot", "dead", "central",
    "happy", "serious", "ready", "simple", "left", "physical", "general",
    "nice", "beautiful", "ugly", "clean", "dirty", "fast", "slow",
    "cheap", "expensive", "rich", "poor", "strong", "weak", "hard",
    "soft", "dark", "light", "white", "black", "blue", "green", "yellow",
    # Adverbs / Function words
    "not", "also", "very", "often", "however", "too", "usually",
    "really", "already", "always", "never", "sometimes", "together",
    "likely", "simply", "generally", "instead", "actually", "again",
    "rather", "almost", "especially", "ever", "quickly", "probably",
    "just", "enough", "quite", "still", "yet", "soon", "here", "there",
    "where", "when", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "some", "any", "no", "many", "much",
    "the", "a", "an", "and", "but", "or", "nor", "for", "so",
    "if", "then", "than", "because", "since", "while", "although",
    "though", "after", "before", "until", "unless", "about", "above",
    "across", "against", "along", "among", "around", "at", "behind",
    "below", "beneath", "beside", "between", "beyond", "by", "down",
    "during", "except", "from", "in", "inside", "into", "near",
    "of", "off", "on", "onto", "out", "outside", "over", "past",
    "through", "to", "toward", "under", "up", "upon", "with",
    "within", "without", "anyway", "anywhere", "basically", "definitely",
    "hello", "hi", "hey", "bye", "goodbye", "welcome", "congrats",
    "literally", "maybe", "obviously", "okay", "ok", "please", "sorry",
    "thanks", "thank", "yeah", "yes", "no", "nope", "well", "like",
}

# Words that exist in both languages (ambiguous without context)
AMBIGUOUS_WORDS = {
    "air", "main", "long", "lap", "pin", "mat", "ram", "tan",
    "ban", "dam", "jam", "man", "pan", "van", "cap", "cat",
    "had", "has", "hem", "hub", "ion", "lot", "net", "nun",
    "par", "rim", "rum", "sap", "sum", "tab", "tar", "tin",
}

# Malay particles - always indicate Malay context even in English sentences
MALAY_PARTICLES = {
    "la", "lah", "kan", "kot", "je", "jer", "ja", "eh", "ah", "oh",
    "wei", "weh", "woi", "oi", "hah", "meh", "geh", "lor", "ma",
    "hor", "leh", "bah", "kah", "tah", "nah", "pun", "doh", "deh",
}

# English borrowed words commonly used in Malay context (don't count as switches)
BORROWED_INTO_MALAY = {
    "okay", "ok", "sorry", "phone", "handphone", "computer", "laptop",
    "internet", "wifi", "online", "offline", "download", "upload",
    "email", "whatsapp", "instagram", "facebook", "twitter", "tiktok",
    "video", "audio", "camera", "battery", "charger", "cable",
    "bus", "taxi", "hotel", "hospital", "clinic", "pharmacy",
    "bank", "atm", "parking", "petrol", "diesel", "engine",
    "air-cond", "aircond", "tv", "radio", "remote", "channel",
    "shopping", "mall", "supermarket", "restaurant", "cafe",
    "meeting", "office", "manager", "boss", "staff", "team",
    "project", "report", "presentation", "deadline", "schedule",
    "class", "exam", "assignment", "semester", "lecture", "tutorial",
    "doctor", "nurse", "patient", "medicine", "vitamin", "protein",
    "football", "badminton", "gym", "fitness", "training",
    "ticket", "passport", "visa", "flight", "airport",
    "signal", "data", "storage", "memory", "password", "username",
}

# Malay slang/shortforms mapped to language origin
MALAY_SLANG = {
    "nk": "ms", "nak": "ms", "tak": "ms", "dah": "ms", "dh": "ms",
    "je": "ms", "jer": "ms", "ja": "ms", "kot": "ms", "kan": "ms",
    "la": "ms", "lah": "ms", "wei": "ms", "weh": "ms", "woi": "ms",
    "camtu": "ms", "camni": "ms", "gitu": "ms", "gini": "ms",
    "tu": "ms", "ni": "ms", "tuh": "ms", "nih": "ms",
    "pastu": "ms", "lepas": "ms", "sbb": "ms", "sebab": "ms",
    "tgk": "ms", "tengok": "ms", "kat": "ms", "kt": "ms",
    "sikit": "ms", "skit": "ms", "byk": "ms", "banyak": "ms",
    "mcm": "ms", "macam": "ms", "cmne": "ms", "cemana": "ms",
    "nape": "ms", "kenapa": "ms", "ape": "ms", "apa": "ms",
    "sape": "ms", "siapa": "ms", "mane": "ms", "mana": "ms",
    "bile": "ms", "bila": "ms", "brp": "ms", "berapa": "ms",
    "dgn": "ms", "dengan": "ms", "utk": "ms", "untuk": "ms",
    "yg": "ms", "yang": "ms", "tp": "ms", "tapi": "ms",
    "mmg": "ms", "memang": "ms", "blh": "ms", "boleh": "ms",
    "xde": "ms", "takde": "ms", "xnak": "ms", "taknak": "ms",
    "ade": "ms", "ada": "ms", "ngan": "ms", "dgn": "ms",
    "org": "ms", "orang": "ms", "brg": "ms", "barang": "ms",
    "blk": "ms", "balik": "ms", "smpi": "ms", "sampai": "ms",
}

ENGLISH_SLANG = {
    "gonna": "en", "wanna": "en", "gotta": "en", "kinda": "en",
    "sorta": "en", "dunno": "en", "lemme": "en", "gimme": "en",
    "ain't": "en", "y'all": "en", "imma": "en", "tryna": "en",
    "finna": "en", "bruh": "en", "bro": "en", "dude": "en",
    "lol": "en", "omg": "en", "wtf": "en", "smh": "en",
    "tbh": "en", "imo": "en", "idk": "en", "ngl": "en",
    "fr": "en", "rn": "en", "af": "en", "lowkey": "en",
    "highkey": "en", "vibe": "en", "vibes": "en", "slay": "en",
    "cap": "en", "nocap": "en", "sus": "en", "lit": "en",
    "fam": "en", "flex": "en", "ghost": "en", "salty": "en",
}


def _tokenize(text: str) -> List[Dict]:
    """Tokenize text into words with positions."""
    tokens = []
    for match in re.finditer(r"[a-zA-Z'\-]+|[^\s]", text):
        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def _classify_token(token: str) -> str:
    """Classify a single token as 'en', 'ms', or 'mixed'."""
    lower = token.lower().strip("'-")

    if not lower or not lower.isalpha():
        return "mixed"

    # Check particles first (strong Malay signal)
    if lower in MALAY_PARTICLES:
        return "ms"

    # Check slang mappings
    if lower in MALAY_SLANG:
        return "ms"
    if lower in ENGLISH_SLANG:
        return "en"

    # Check borrowed words (neutral - don't count as switch)
    if lower in BORROWED_INTO_MALAY:
        return "mixed"

    # Check ambiguous words
    if lower in AMBIGUOUS_WORDS:
        return "mixed"

    # Check main word lists
    in_malay = lower in MALAY_WORDS
    in_english = lower in ENGLISH_WORDS

    if in_malay and in_english:
        return "mixed"
    elif in_malay:
        return "ms"
    elif in_english:
        return "en"

    # Heuristics for unknown words
    # Malay prefixes
    if any(lower.startswith(p) for p in ("me", "ber", "ter", "pe", "di", "ke", "se")):
        if len(lower) > 4:
            return "ms"
    # Malay suffixes
    if any(lower.endswith(s) for s in ("kan", "an", "nya", "lah", "kah")):
        return "ms"
    # English suffixes
    if any(lower.endswith(s) for s in ("tion", "sion", "ness", "ment", "ing", "ous", "ive", "ful", "less", "able", "ible", "ly", "ed")):
        if len(lower) > 4:
            return "en"

    return "mixed"


def detect_switches(text: str) -> List[Dict]:
    """
    Detect language of each token in text.

    Returns:
        List of dicts: [{"token": "...", "language": "en|ms|mixed", "position": (start, end)}]
    """
    tokens = _tokenize(text)
    results = []

    for tok in tokens:
        lang = _classify_token(tok["token"])
        results.append({
            "token": tok["token"],
            "language": lang,
            "position": (tok["start"], tok["end"]),
        })

    return results


def switch_points(text: str) -> List[int]:
    """
    Find indices where language switches occur.

    Returns:
        List of character positions where a switch happens.
    """
    detections = detect_switches(text)
    points = []

    # Filter out mixed/punctuation tokens for switch detection
    lang_tokens = [(d["position"][0], d["language"]) for d in detections if d["language"] in ("en", "ms")]

    for i in range(1, len(lang_tokens)):
        if lang_tokens[i][1] != lang_tokens[i - 1][1]:
            points.append(lang_tokens[i][0])

    return points


def switch_ratio(text: str) -> float:
    """
    Calculate code-switching ratio (0-1).

    0 = no switching (monolingual), 1 = maximum switching (alternates every word).

    Returns:
        Float between 0 and 1.
    """
    detections = detect_switches(text)
    lang_tokens = [d for d in detections if d["language"] in ("en", "ms")]

    if len(lang_tokens) <= 1:
        return 0.0

    switches = 0
    for i in range(1, len(lang_tokens)):
        if lang_tokens[i]["language"] != lang_tokens[i - 1]["language"]:
            switches += 1

    # Normalize: max possible switches is len - 1
    return switches / (len(lang_tokens) - 1)


def dominant_language(text: str) -> str:
    """
    Determine the dominant language in text.

    Returns:
        "en" | "ms" | "mixed"
    """
    detections = detect_switches(text)
    en_count = sum(1 for d in detections if d["language"] == "en")
    ms_count = sum(1 for d in detections if d["language"] == "ms")

    total = en_count + ms_count
    if total == 0:
        return "mixed"

    en_ratio = en_count / total
    ms_ratio = ms_count / total

    if en_ratio > 0.7:
        return "en"
    elif ms_ratio > 0.7:
        return "ms"
    else:
        return "mixed"


def segment_by_language(text: str) -> List[Dict]:
    """
    Segment text into contiguous language spans.

    Returns:
        List of {"text": "...", "language": "en|ms|mixed", "start": int, "end": int}
    """
    detections = detect_switches(text)
    if not detections:
        return []

    segments = []
    current_lang = detections[0]["language"]
    current_start = detections[0]["position"][0]
    current_end = detections[0]["position"][1]

    for i in range(1, len(detections)):
        d = detections[i]
        if d["language"] == current_lang or d["language"] == "mixed":
            current_end = d["position"][1]
        else:
            seg_text = text[current_start:current_end]
            segments.append({
                "text": seg_text,
                "language": current_lang,
                "start": current_start,
                "end": current_end,
            })
            current_lang = d["language"]
            current_start = d["position"][0]
            current_end = d["position"][1]

    # Append last segment
    seg_text = text[current_start:current_end]
    segments.append({
        "text": seg_text,
        "language": current_lang,
        "start": current_start,
        "end": current_end,
    })

    return segments


def switch_matrix(text: str) -> Dict[str, int]:
    """
    Count language transitions.

    Returns:
        Dict with keys like "en->ms", "ms->en" and their counts.
    """
    detections = detect_switches(text)
    lang_tokens = [d for d in detections if d["language"] in ("en", "ms")]

    matrix = {"en->ms": 0, "ms->en": 0}

    for i in range(1, len(lang_tokens)):
        prev_lang = lang_tokens[i - 1]["language"]
        curr_lang = lang_tokens[i]["language"]
        if prev_lang != curr_lang:
            key = f"{prev_lang}->{curr_lang}"
            matrix[key] = matrix.get(key, 0) + 1

    return matrix


def classify_switch_type(text: str) -> str:
    """
    Classify the type of code-switching in text.

    Returns:
        "intra-sentential" - switching within a sentence
        "inter-sentential" - switching between sentences
        "tag-switching" - only particles/tags from other language
        "none" - no code-switching detected
    """
    detections = detect_switches(text)
    lang_tokens = [d for d in detections if d["language"] in ("en", "ms")]

    if not lang_tokens:
        return "none"

    languages_present = set(d["language"] for d in lang_tokens)
    if len(languages_present) <= 1:
        return "none"

    # Check for tag-switching: only particles from the minority language
    en_tokens = [d for d in lang_tokens if d["language"] == "en"]
    ms_tokens = [d for d in lang_tokens if d["language"] == "ms"]

    # If minority language tokens are all particles, it's tag-switching
    if len(ms_tokens) <= len(en_tokens):
        minority = ms_tokens
    else:
        minority = en_tokens

    if minority:
        all_particles = all(
            d["token"].lower() in MALAY_PARTICLES or
            d["token"].lower() in {"right", "yeah", "okay", "ok", "so", "like", "well"}
            for d in minority
        )
        if all_particles:
            return "tag-switching"

    # Check for inter-sentential: splits by sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) > 1:
        sentence_langs = []
        for sent in sentences:
            sent_det = detect_switches(sent)
            sent_lang_tokens = [d for d in sent_det if d["language"] in ("en", "ms")]
            if sent_lang_tokens:
                en_c = sum(1 for d in sent_lang_tokens if d["language"] == "en")
                ms_c = sum(1 for d in sent_lang_tokens if d["language"] == "ms")
                sentence_langs.append("en" if en_c >= ms_c else "ms")

        if len(set(sentence_langs)) > 1:
            # Check if individual sentences are mostly monolingual
            mono_count = 0
            for sent in sentences:
                sent_det = detect_switches(sent)
                sent_lang_tokens = [d for d in sent_det if d["language"] in ("en", "ms")]
                if sent_lang_tokens:
                    langs_in_sent = set(d["language"] for d in sent_lang_tokens)
                    if len(langs_in_sent) == 1:
                        mono_count += 1
            if mono_count >= len(sentences) * 0.6:
                return "inter-sentential"

    return "intra-sentential"


def resolve_ambiguous(text: str) -> List[Dict]:
    """
    Resolve ambiguous words using surrounding context.

    For words that exist in both BM and EN word lists, uses neighbor context
    and bigram patterns to determine the most likely language.

    Args:
        text: Input text.

    Returns:
        List of dicts: [{"token": "...", "language": "en|ms", "position": (start, end), "resolved_by": "..."}]
        Only returns tokens that were ambiguous and got resolved.

    Example:
        >>> resolve_ambiguous("I makan nasi with air")
        [{'token': 'air', 'language': 'ms', 'position': (21, 24), 'resolved_by': 'neighbor_context'}]
    """
    tokens = _tokenize(text)
    results = []

    # First pass: classify all tokens
    classifications = []
    for tok in tokens:
        lang = _classify_token(tok["token"])
        classifications.append({
            "token": tok["token"],
            "language": lang,
            "position": (tok["start"], tok["end"]),
        })

    # Second pass: resolve ambiguous tokens using context
    for i, cls in enumerate(classifications):
        if cls["language"] != "mixed":
            continue

        lower = cls["token"].lower().strip("'-")

        # Skip non-alpha or borrowed words
        if not lower.isalpha():
            continue
        if lower in BORROWED_INTO_MALAY:
            continue
        if lower not in AMBIGUOUS_WORDS and lower not in MALAY_WORDS and lower not in ENGLISH_WORDS:
            continue

        # Strategy 1: Neighbor context (window of 3 on each side)
        window_start = max(0, i - 3)
        window_end = min(len(classifications), i + 4)
        neighbors = classifications[window_start:i] + classifications[i+1:window_end]

        en_neighbors = sum(1 for n in neighbors if n["language"] == "en")
        ms_neighbors = sum(1 for n in neighbors if n["language"] == "ms")

        resolved_lang = None
        resolved_by = None

        if ms_neighbors > en_neighbors:
            resolved_lang = "ms"
            resolved_by = "neighbor_context"
        elif en_neighbors > ms_neighbors:
            resolved_lang = "en"
            resolved_by = "neighbor_context"
        else:
            # Strategy 2: Bigram context - check immediate neighbors
            prev_lang = classifications[i - 1]["language"] if i > 0 else None
            next_lang = classifications[i + 1]["language"] if i + 1 < len(classifications) else None

            if prev_lang in ("en", "ms") and next_lang in ("en", "ms"):
                # If both neighbors agree, follow them
                if prev_lang == next_lang:
                    resolved_lang = prev_lang
                    resolved_by = "bigram_context"
                else:
                    # At a switch point - use the preceding language
                    resolved_lang = prev_lang
                    resolved_by = "bigram_switch_point"
            elif prev_lang in ("en", "ms"):
                resolved_lang = prev_lang
                resolved_by = "bigram_context"
            elif next_lang in ("en", "ms"):
                resolved_lang = next_lang
                resolved_by = "bigram_context"

        if resolved_lang:
            results.append({
                "token": cls["token"],
                "language": resolved_lang,
                "position": cls["position"],
                "resolved_by": resolved_by,
            })

    return results
