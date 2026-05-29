"""
malaysian_manglish_nlp.topic - Topic modeling for Manglish text.

Classifies Malaysian Manglish text into predefined topics using
keyword-based matching with TF-IDF-like scoring.

Topics:
    food, politics, sports, tech, education, entertainment,
    religion, daily_life, business, health, travel, relationships

Usage:
    from malaysian_manglish_nlp.topic import classify_topic, classify_topics

    classify_topic("mamak punya nasi lemak memang sedap gila")
    # {"topic": "food", "confidence": 0.85, "keywords_matched": ["mamak", "nasi lemak", "sedap"]}

    classify_topics("aku tengok Netflix lepas balik kerja", top_n=3)
    # [{"topic": "entertainment", ...}, {"topic": "daily_life", ...}, ...]
"""

from __future__ import annotations

from typing import Any, Dict, List

import re
import math
from collections import Counter


# Topic keyword dictionaries - BM + EN + slang
TOPIC_KEYWORDS = {
    "food": [
        # Malay food terms
        "makan", "makanan", "masak", "masakan", "lauk", "nasi", "sedap",
        "lazat", "rasa", "perisa", "goreng", "rebus", "bakar", "panggang",
        "kuah", "sambal", "kari", "rendang", "lemak", "pedas", "manis",
        "masam", "masin", "pahit", "sayur", "ikan", "ayam", "daging",
        "udang", "sotong", "telur", "tahu", "tempe", "mi", "mee",
        "bihun", "kuey teow", "laksa", "satay", "roti", "canai",
        "mamak", "warung", "kedai makan", "restoran", "restaurant",
        "gerai", "hawker", "kopitiam", "cafe", "kopi", "teh",
        # Specific dishes
        "nasi lemak", "nasi goreng", "char kuey teow", "roti canai",
        "teh tarik", "cendol", "ais kacang", "rojak", "pisang goreng",
        "curry puff", "karipap", "tom yam", "asam pedas", "sup",
        # English food terms
        "recipe", "cook", "cooking", "food", "eat", "eating", "meal",
        "breakfast", "lunch", "dinner", "supper", "snack", "dessert",
        "hungry", "lapar", "kenyang", "full", "delicious", "yummy",
        "foodie", "menu", "order", "delivery", "grabfood", "foodpanda",
        # Slang
        "tapau", "dabao", "jom makan", "belanja", "sedap gila",
        "power", "mantap", "terbaik", "best",
    ],
    "politics": [
        # Malay political terms
        "kerajaan", "politik", "parti", "parlimen", "parliament",
        "menteri", "perdana menteri", "PM", "TPM", "sultan", "agong",
        "rakyat", "negara", "undang", "perlembagaan", "demokrasi",
        "pembangkang", "opposition", "undi", "mengundi", "pengundi",
        "pilihan raya", "PRU", "PRN", "election", "vote", "voter",
        "calon", "candidate", "kempen", "campaign", "manifesto",
        # Parties
        "UMNO", "PAS", "PKR", "DAP", "Bersatu", "PPBM", "Amanah",
        "GPS", "BN", "Barisan Nasional", "Pakatan", "Perikatan",
        "PN", "PH", "Pakatan Harapan",
        # Political figures (titles only)
        "Dato", "Datuk", "Tan Sri", "Tun",
        # Issues
        "korupsi", "rasuah", "corruption", "skandal", "scandal",
        "subsidi", "subsidy", "cukai", "tax", "GST", "SST",
        "dasar", "policy", "bajet", "budget", "ekonomi",
        # English political terms
        "government", "minister", "prime minister", "political",
        "democracy", "reform", "protest", "rally", "demonstration",
        # Slang
        "gomen", "politikus", "kroni", "kronisme",
    ],
    "sports": [
        # General sports
        "sukan", "sports", "bola", "football", "soccer", "goal",
        "player", "pemain", "pasukan", "team", "liga", "league",
        "perlawanan", "match", "game", "tournament", "kejohanan",
        "piala", "cup", "trophy", "medal", "emas", "gold",
        "perak", "silver", "gangsa", "bronze", "champion", "juara",
        # Football specific
        "EPL", "Premier League", "La Liga", "Serie A", "Bundesliga",
        "Champions League", "World Cup", "Piala Dunia", "FIFA",
        "Harimau Malaya", "JDT", "Selangor FC", "FAM",
        "striker", "goalkeeper", "defender", "midfielder",
        # Badminton
        "badminton", "shuttlecock", "Lee Zii Jia", "Lee Chong Wei",
        "BWF", "All England", "Thomas Cup", "Uber Cup",
        # Other sports
        "F1", "Formula 1", "MotoGP", "cycling", "basikal",
        "swimming", "renang", "running", "lari", "marathon",
        "gym", "fitness", "workout", "exercise", "senaman",
        "basketball", "tennis", "golf", "cricket", "rugby",
        "esports", "gaming", "PUBG", "Mobile Legends", "MLBB",
        "Dota", "Valorant",
        # Slang
        "menang", "kalah", "seri", "draw", "win", "lose",
        "score", "assist", "tackle", "foul", "penalty",
    ],
    "tech": [
        # Devices
        "phone", "telefon", "handphone", "HP", "smartphone",
        "laptop", "komputer", "computer", "PC", "tablet", "iPad",
        "monitor", "keyboard", "mouse", "earphone", "headphone",
        # Software/Internet
        "app", "application", "software", "download", "install",
        "update", "upgrade", "internet", "wifi", "data", "hotspot",
        "website", "browser", "google", "search",
        # Programming
        "coding", "code", "programming", "developer", "programmer",
        "bug", "debug", "deploy", "server", "database", "API",
        "frontend", "backend", "fullstack", "framework",
        "Python", "JavaScript", "Java", "React", "Node",
        # AI/Tech trends
        "AI", "artificial intelligence", "machine learning", "ML",
        "ChatGPT", "GPT", "robot", "automation", "blockchain",
        "crypto", "cryptocurrency", "Bitcoin", "NFT",
        # Hardware
        "spec", "specs", "RAM", "storage", "processor", "GPU",
        "CPU", "SSD", "battery", "bateri", "screen", "display",
        "camera", "kamera", "megapixel",
        # Brands
        "iPhone", "Samsung", "Xiaomi", "Huawei", "Apple",
        "Android", "iOS", "Windows", "Linux", "Mac",
        # Slang
        "lag", "hang", "crash", "slow", "laju", "fast",
        "canggih", "outdated", "latest", "baru",
    ],
    "education": [
        # General
        "belajar", "study", "studying", "pelajar", "student",
        "sekolah", "school", "universiti", "university", "uni",
        "kolej", "college", "kelas", "class", "lecture", "kuliah",
        "lecturer", "pensyarah", "professor", "prof", "cikgu",
        "teacher", "guru", "tutor", "tutorial",
        # Academic
        "exam", "peperiksaan", "test", "ujian", "quiz", "kuiz",
        "assignment", "tugasan", "homework", "kerja rumah",
        "thesis", "tesis", "dissertation", "research", "kajian",
        "semester", "sem", "credit", "kredit", "pointer", "CGPA",
        "GPA", "dean list", "first class", "second class",
        "graduate", "grad", "graduation", "konvo", "convocation",
        "diploma", "degree", "ijazah", "master", "PhD",
        # Malaysian universities
        "UMPSA", "UMP", "UiTM", "UM", "USM", "UKM", "UPM",
        "UTM", "UNIMAS", "UMS", "IPTA", "IPTS",
        "STPM", "SPM", "UPSR", "PT3",
        # Activities
        "revision", "ulangkaji", "hafal", "memorize", "note",
        "nota", "textbook", "buku", "library", "perpustakaan",
        "lab", "makmal", "presentation", "pembentangan",
        "group project", "FYP", "final year project",
        # Slang
        "ponteng", "skip class", "carry", "pointer jatuh",
        "dean list", "repeat", "extend", "drop",
    ],
    "entertainment": [
        # Movies/TV
        "movie", "filem", "wayang", "cinema", "panggung",
        "drama", "series", "season", "episode", "Netflix",
        "Disney", "HBO", "streaming", "tengok", "tonton",
        "watch", "trailer", "review", "rating", "actor",
        "actress", "pelakon", "director", "pengarah",
        # Music
        "lagu", "song", "music", "muzik", "album", "single",
        "concert", "konsert", "gig", "live", "perform",
        "singer", "penyanyi", "band", "rapper", "DJ",
        "Spotify", "playlist", "dengar", "listen",
        # K-pop/J-pop
        "Kpop", "K-pop", "BTS", "Blackpink", "idol",
        "comeback", "MV", "music video", "fandom", "stan",
        "bias", "lightstick", "fanmeet",
        # Gaming
        "game", "gaming", "gamer", "play", "main",
        "console", "PS5", "PlayStation", "Xbox", "Nintendo",
        "Switch", "Steam", "online", "multiplayer",
        # Social media entertainment
        "TikTok", "YouTube", "viral", "trending", "content",
        "creator", "influencer", "vlog", "podcast",
        # Slang
        "best gila", "power", "boring", "sien", "syok",
        "layan", "marathon", "binge", "spoiler",
    ],
    "religion": [
        # Islam
        "solat", "sembahyang", "prayer", "masjid", "mosque",
        "surau", "Quran", "Al-Quran", "ayat", "surah",
        "hadith", "hadis", "sunnah", "fardhu", "wajib",
        "haram", "halal", "makruh", "sunat", "bid'ah",
        "puasa", "fasting", "Ramadan", "Ramadhan", "iftar",
        "sahur", "berbuka", "terawih", "tarawih",
        "zakat", "sedekah", "infaq", "wakaf",
        "haji", "umrah", "Mekah", "Madinah",
        "ustaz", "ustazah", "imam", "muezzin", "bilal",
        "khutbah", "ceramah", "tazkirah", "dakwah",
        "doa", "zikir", "selawat", "istighfar", "taubat",
        "syurga", "neraka", "akhirat", "pahala", "dosa",
        "Allah", "Rasulullah", "Nabi", "Muhammad",
        # General religious terms
        "agama", "religion", "iman", "faith", "taqwa",
        "ibadah", "worship", "church", "gereja", "temple",
        "kuil", "tokong", "Buddhist", "Hindu", "Christian",
        "Kristian", "Bible", "Injil",
        # Events
        "Hari Raya", "Aidilfitri", "Aidiladha", "Maulid",
        "Israk Mikraj", "Nuzul Quran",
        # Slang
        "insyaAllah", "Alhamdulillah", "MasyaAllah",
        "SubhanAllah", "Astaghfirullah", "Bismillah",
    ],
    "daily_life": [
        # Work
        "kerja", "work", "working", "office", "pejabat",
        "boss", "colleague", "rakan sekerja", "meeting",
        "gaji", "salary", "overtime", "OT", "cuti", "leave",
        "resign", "quit", "fired", "kena buang",
        # Home
        "rumah", "house", "home", "bilik", "room", "dapur",
        "kitchen", "toilet", "bathroom", "taman", "apartment",
        "kondo", "condo", "sewa", "rent", "pindah", "move",
        # Daily routine
        "bangun", "wake up", "tidur", "sleep", "mandi",
        "shower", "siap", "ready", "keluar", "balik",
        "sampai", "arrive", "tunggu", "wait",
        # Transport/Traffic
        "traffic", "jam", "jem", "jalan", "road", "highway",
        "tol", "toll", "parking", "drive", "pandu",
        "kereta", "car", "motor", "motorcycle", "bas", "bus",
        "LRT", "MRT", "KTM", "Grab", "taxi",
        # Weather/Environment
        "hujan", "rain", "panas", "hot", "sejuk", "cold",
        "banjir", "flood", "ribut", "storm",
        # Chores
        "basuh", "wash", "kemas", "clean", "sapu", "sweep",
        "sidai", "jemur", "lipat", "iron", "gosok",
        # Slang
        "penat", "tired", "malas", "lazy", "boring",
        "sien", "stress", "hectic", "rushing", "lambat",
        "awal", "early", "lewat", "late",
    ],
    "business": [
        # General business
        "bisnes", "business", "perniagaan", "syarikat", "company",
        "kedai", "shop", "store", "enterprise", "startup",
        "usahawan", "entrepreneur", "founder", "CEO",
        # Finance
        "untung", "profit", "rugi", "loss", "modal", "capital",
        "investment", "pelaburan", "saham", "stock", "share",
        "dividen", "dividend", "ROI", "revenue", "income",
        "pendapatan", "kos", "cost", "margin", "cashflow",
        # Operations
        "customer", "pelanggan", "client", "supplier",
        "pembekal", "vendor", "order", "tempahan",
        "delivery", "penghantaran", "marketing", "pemasaran",
        "sales", "jualan", "promote", "promosi", "iklan", "ads",
        # E-commerce
        "Shopee", "Lazada", "online shop", "dropship",
        "reseller", "agent", "stokis", "pre-order",
        "COD", "postage", "pos", "courier", "J&T", "Poslaju",
        # Property
        "hartanah", "property", "rumah", "tanah", "land",
        "developer", "pemaju", "loan", "pinjaman", "bank",
        # Slang
        "side income", "passive income", "hustle", "grind",
        "cari duit", "duit", "money", "ringgit", "RM",
    ],
    "health": [
        # General health
        "sakit", "sick", "sihat", "healthy", "kesihatan", "health",
        "hospital", "klinik", "clinic", "doctor", "doktor",
        "nurse", "jururawat", "ubat", "medicine", "medication",
        "preskripsi", "prescription", "farmasi", "pharmacy",
        # Symptoms
        "demam", "fever", "batuk", "cough", "selsema", "flu",
        "cold", "sakit kepala", "headache", "sakit perut",
        "stomachache", "pening", "dizzy", "muntah", "vomit",
        "cirit-birit", "diarrhea", "alahan", "allergy",
        "bengkak", "swelling", "luka", "wound", "patah", "fracture",
        # Diseases
        "covid", "coronavirus", "denggi", "dengue", "kencing manis",
        "diabetes", "darah tinggi", "hypertension", "kanser", "cancer",
        "asma", "asthma", "jantung", "heart",
        # Mental health
        "mental health", "kesihatan mental", "stress", "anxiety",
        "depression", "kemurungan", "burnout", "therapy", "therapist",
        "counseling", "kaunseling",
        # Wellness
        "diet", "exercise", "senaman", "vitamin", "supplement",
        "tidur", "sleep", "rehat", "rest", "vaksin", "vaccine",
        "booster", "checkup", "pemeriksaan",
        # Slang
        "MC", "medical leave", "warded", "admit", "discharge",
        "specialist", "pakar", "refer", "appointment",
    ],
    "travel": [
        # General travel
        "travel", "jalan", "jalan-jalan", "trip", "vacation",
        "cuti", "holiday", "percutian", "melancong", "tourism",
        "pelancong", "tourist", "backpack", "backpacker",
        # Transport
        "flight", "penerbangan", "kapal terbang", "airplane",
        "airport", "lapangan terbang", "KLIA", "boarding",
        "check-in", "luggage", "bagasi", "passport", "visa",
        "AirAsia", "MAS", "Malaysia Airlines", "Firefly",
        # Accommodation
        "hotel", "hostel", "resort", "Airbnb", "homestay",
        "booking", "tempahan", "check in", "check out",
        "bilik", "room", "suite",
        # Destinations
        "Langkawi", "Penang", "Pulau Pinang", "Sabah",
        "Sarawak", "Melaka", "Johor", "Cameron Highlands",
        "Tioman", "Perhentian", "Redang", "Pangkor",
        "Genting", "Legoland", "Sunway Lagoon",
        "Thailand", "Singapore", "Indonesia", "Bali",
        "Japan", "Korea", "Europe", "Australia",
        # Activities
        "snorkeling", "diving", "hiking", "mendaki",
        "beach", "pantai", "island", "pulau", "gunung",
        "mountain", "waterfall", "air terjun", "taman negara",
        # Slang
        "jom travel", "road trip", "balik kampung", "kampung",
        "itinerary", "budget travel", "murah", "promo",
    ],
    "relationships": [
        # Romance
        "couple", "pasangan", "boyfriend", "girlfriend",
        "BF", "GF", "partner", "hubby", "wifey",
        "suami", "isteri", "husband", "wife",
        "cinta", "love", "sayang", "dear", "darling",
        "crush", "suka", "like", "minat", "interested",
        "date", "dating", "teman", "courting",
        # Marriage
        "kahwin", "nikah", "marriage", "wedding", "perkahwinan",
        "tunang", "engaged", "engagement", "pertunangan",
        "hantaran", "mas kahwin", "majlis", "reception",
        "pengantin", "bride", "groom", "honeymoon",
        # Breakup
        "breakup", "break up", "putus", "clash", "gaduh",
        "fight", "bergaduh", "argument", "jealous", "cemburu",
        "curang", "cheat", "cheating", "selingkuh",
        "move on", "ex", "mantan", "single",
        # Family
        "family", "keluarga", "mak", "ayah", "abah",
        "ibu", "bapa", "adik", "abang", "kakak",
        "anak", "baby", "pregnant", "mengandung", "hamil",
        # Friendship
        "kawan", "friend", "bestie", "BFF", "geng",
        "squad", "hangout", "lepak", "gathering",
        # Slang
        "jodoh", "takdir", "halal", "halalkan",
        "tackle", "approach", "friendzone", "situationship",
        "talking stage", "red flag", "green flag", "toxic",
    ],
}

# Precompute total keyword count for IDF-like weighting
_TOTAL_TOPICS = len(TOPIC_KEYWORDS)


def _normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def _find_keyword_matches(text: str, keywords: str) -> Dict[str, Any]:
    """Find all keyword matches in text, supporting multi-word phrases."""
    text_lower = _normalize_text(text)
    matches = []

    # Sort keywords by length (longest first) to match phrases before words
    sorted_keywords = sorted(keywords, key=len, reverse=True)

    for keyword in sorted_keywords:
        kw_lower = keyword.lower()
        # Use word boundary matching for single words, substring for phrases
        if ' ' in kw_lower:
            if kw_lower in text_lower:
                matches.append(keyword)
        else:
            # Word boundary match
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(keyword)

    return matches


def _compute_score(matches: Any, total_keywords: str, text_length: str) -> float:
    """
    Compute topic relevance score using keyword density and coverage.

    Combines:
    - Keyword match count (more matches = higher relevance)
    - Coverage ratio (matches / total keywords for topic)
    - Density (matches relative to text length)
    """
    if not matches or text_length == 0:
        return 0.0

    match_count = len(matches)
    word_count = max(text_length, 1)

    # Coverage: what fraction of topic keywords appeared
    coverage = match_count / max(total_keywords, 1)

    # Density: keyword matches relative to text word count
    density = match_count / max(word_count, 1)

    # Combined score with diminishing returns
    # Use log to prevent very long texts from dominating
    raw_score = (coverage * 0.4) + (density * 0.3) + (min(match_count / 5, 1.0) * 0.3)

    # Normalize to 0-1 range with sigmoid-like curve
    confidence = min(raw_score * 3, 0.99)

    return round(confidence, 4)


def classify_topic(text: str) -> Dict[str, Any]:
    """
    Classify text into the most likely topic.

    Args:
        text: Input text to classify.

    Returns:
        dict: {
            "topic": str,           # Most likely topic
            "confidence": float,    # Confidence score 0-1
            "keywords_matched": list  # Keywords found in text
        }
    """
    if not text or not text.strip():
        return {"topic": "daily_life", "confidence": 0.0, "keywords_matched": []}

    text_words = len(_normalize_text(text).split())
    best_topic = "daily_life"
    best_score = 0.0
    best_matches = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = _find_keyword_matches(text, keywords)
        score = _compute_score(matches, len(keywords), text_words)

        if score > best_score:
            best_score = score
            best_topic = topic
            best_matches = matches

    return {
        "topic": best_topic,
        "confidence": round(best_score, 2),
        "keywords_matched": best_matches[:10],  # Limit to top 10
    }


def classify_topics(text: str, top_n: int = 3) -> Dict[str, Any]:
    """
    Classify text into top N most likely topics.

    Args:
        text: Input text to classify.
        top_n: Number of top topics to return (default 3).

    Returns:
        list[dict]: List of top N topics with scores, sorted by confidence.
    """
    if not text or not text.strip():
        return [{"topic": "daily_life", "confidence": 0.0, "keywords_matched": []}]

    text_words = len(_normalize_text(text).split())
    results = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = _find_keyword_matches(text, keywords)
        score = _compute_score(matches, len(keywords), text_words)

        if score > 0:
            results.append({
                "topic": topic,
                "confidence": round(score, 2),
                "keywords_matched": matches[:10],
            })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # If no results, return default
    if not results:
        return [{"topic": "daily_life", "confidence": 0.0, "keywords_matched": []}]

    return results[:top_n]


def classify_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Classify a batch of texts into topics.

    Args:
        texts: List of texts to classify.

    Returns:
        list[dict]: List of classification results.
    """
    if not texts:
        return []

    return [classify_topic(text) for text in texts]


def extract_topic_keywords(text: str) -> Dict[str, Any]:
    """
    Extract all topic-relevant keywords found in text.

    Args:
        text: Input text to analyze.

    Returns:
        list[dict]: List of keywords with their associated topics.
            Each item: {"keyword": str, "topic": str}
    """
    if not text or not text.strip():
        return []

    results = []
    seen = set()

    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = _find_keyword_matches(text, keywords)
        for match in matches:
            if match.lower() not in seen:
                seen.add(match.lower())
                results.append({"keyword": match, "topic": topic})

    return results


def topic_distribution(texts: List[str]) -> Dict[str, Any]:
    """
    Compute topic distribution across a corpus of texts.

    Args:
        texts: List of texts to analyze.

    Returns:
        dict: {
            "distribution": dict,   # topic -> percentage (0-100)
            "total_texts": int,     # Total texts analyzed
            "topic_counts": dict,   # topic -> count of texts classified
        }
    """
    if not texts:
        return {
            "distribution": {},
            "total_texts": 0,
            "topic_counts": {},
        }

    topic_counts = Counter()

    for text in texts:
        result = classify_topic(text)
        if result["confidence"] > 0:
            topic_counts[result["topic"]] += 1

    total = len(texts)
    distribution = {}

    for topic in TOPIC_KEYWORDS:
        count = topic_counts.get(topic, 0)
        distribution[topic] = round((count / total) * 100, 1)

    return {
        "distribution": distribution,
        "total_texts": total,
        "topic_counts": dict(topic_counts),
    }


# Convenience exports
AVAILABLE_TOPICS = list(TOPIC_KEYWORDS.keys())
