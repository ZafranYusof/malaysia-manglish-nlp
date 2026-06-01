"""Enhanced aspect-based sentiment analysis with Malaysian context.

Detects aspects dynamically from text, supports multiple domains
(restaurant, product, app/software, general), and provides per-aspect
sentiment scoring with conflict detection.

Rule-based, zero external dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import re

_RE_WORDS = re.compile(r'[a-zA-Z0-9]+')

_NEGATORS = {
    'tak', 'tidak', 'bukan', 'x', 'xde', 'takde', 'no', 'not',
    "don't", "doesn't", 'never', 'belum', 'jangan', 'tok', 'sik', 'dok',
}

_CONTRAST_MARKERS = {
    'tapi', 'but', 'however', 'cuma', 'cuman', 'except', 'unfortunately',
    'walaupun', 'namun', 'tetapi', 'walau', 'though',
}

_INTENSIFIERS = {
    'gila': 1.5, 'giler': 1.5, 'gile': 1.5, 'sangat': 1.5, 'sgt': 1.5,
    'memang': 1.3, 'mmg': 1.3, 'betul': 1.3, 'btl': 1.3, 'really': 1.4,
    'very': 1.4, 'super': 1.5, 'ultra': 1.5, 'extremely': 1.6,
    'teramat': 1.6, 'habis': 1.3, 'damn': 1.4, 'so': 1.2,
    'tahap': 1.3, 'max': 1.4, 'over': 1.3,
}

_PARTICLES = {'la', 'lah', 'je', 'jer', 'ni', 'tu', 'kan', 'kot', 'pun', 'ke', 'dah'}

# ============================================================
# Domain aspect categories
# ============================================================

def _asp(kw: set, pos: set, neg: set) -> Dict[str, Any]:
    """Helper to build aspect entry."""
    return {'keywords': kw, 'positive': pos, 'negative': neg}


_DOMAIN_ASPECTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'restaurant': {
        'food': _asp(
            {'makanan','makan','rasa','sedap','masak','lauk','nasi','mee','kuih',
             'food','meal','dish','taste','flavor','menu','recipe','sambal','ayam',
             'ikan','sayur','sup','goreng','rebus','pedas','manis','masam','pahit',
             'lemak','santan','kuah','hidangan','roti','satay','rendang','laksa',
             'cendol','teh tarik','kopi','delicious','yummy','tasty','lazat','enak',
             'bland','tawar','masin','fresh','segar'},
            {'sedap','lazat','enak','tasty','delicious','yummy','fresh','segar',
             'flavorful','padu','mantap','solid','perfect','authentic','original',
             'lemak','rich','hearty','generous','banyak','puas'},
            {'tawar','bland','masin','salty','overcooked','undercooked','basi','stale',
             'cold','sejuk','kering','dry','berminyak','oily','hambar','busuk','mentah',
             'raw','cair','watery'},
        ),
        'service': _asp(
            {'service','staff','waiter','waitress','pelayan','layan','layanan',
             'pekerja','cashier','server','attend','response','friendly','rude',
             'helpful','attitude','hospitality','order','serve','customer','budi',
             'senyum','smile','greet','sambut'},
            {'friendly','helpful','fast','cepat','pantas','mesra','ramah','sopan',
             'polite','attentive','professional','senyum','smile','warm','bagus',
             'good','great','efficient','cekap','best'},
            {'rude','kasar','slow','lambat','lewat','ignore','biadab','kurang ajar',
             'berlagak','sombong','arrogant','unfriendly','cold','judes','jutek',
             'teruk','bad','worst','hampeh'},
        ),
        'price': _asp(
            {'harga','mahal','murah','berbaloi','price','cost','expensive','cheap',
             'affordable','worth','value','overpriced','budget','ringgit','rm','bayar',
             'pay','bil','bill','charge','fee','diskaun','discount','promo','offer',
             'pakej','package','berpatutan','reasonable'},
            {'murah','cheap','affordable','berbaloi','worth','value','reasonable',
             'berpatutan','budget','diskaun','discount','promo','free','percuma'},
            {'mahal','expensive','overpriced','ripoff','koyak','pokai','cekik',
             'pricey','unaffordable','membazir','waste','rugi'},
        ),
        'ambiance': _asp(
            {'tempat','suasana','ambiance','atmosphere','vibe','decor','decoration',
             'interior','lighting','view','pemandangan','aircond','music','muzik',
             'seating','kerusi','meja','wifi','outdoor','indoor'},
            {'cantik','beautiful','cozy','selesa','comfortable','luas','spacious',
             'bersih','clean','tenang','quiet','aesthetic','nice','best','cool',
             'chill','relaxing','warm','inviting'},
            {'kotor','dirty','sesak','crowded','bising','noisy','sempit','cramped',
             'panas','hot','pengap','stuffy','buruk','ugly','hodoh','old','lama',
             'rosak','uncomfortable'},
        ),
        'cleanliness': _asp(
            {'bersih','kotor','clean','dirty','hygiene','higien','sanitary','tissue',
             'basuh','cuci','wash','toilet','tandas','restroom','lantai','floor',
             'meja','pinggan','plate','cutlery'},
            {'bersih','clean','hygienic','spotless','kemas','neat','tidy','sanitary',
             'fresh'},
            {'kotor','dirty','filthy','jijik','gross','berminyak','oily','sticky',
             'melekit','berhabuk','dusty','busuk','smelly','bau','berkulat','mouldy'},
        ),
        'portion': _asp(
            {'portion','saiz','size','banyak','sikit','kuantiti','quantity','serving',
             'mangkuk','bowl','pinggan','plate','penuh','full','half','separuh',
             'besar','large','kecil','small','regular'},
            {'banyak','generous','large','besar','penuh','full','hearty','filling',
             'kenyang','satisfying','baloi','puas','berbaloi'},
            {'sikit','small','kecil','tiny','sedikit','meager','ciput','half',
             'separuh','empty','kosong','disappointing'},
        ),
        'speed': _asp(
            {'cepat','lambat','speed','fast','slow','wait','tunggu','masa','time',
             'delivery','hantar','sampai','arrive','express','instant','segera',
             'queue','baris','giliran'},
            {'cepat','fast','pantas','laju','express','quick','segera','immediate',
             'speedy','efficient','cekap'},
            {'lambat','slow','lewat','lama','delay','tunggu','wait','terlambat',
             'late','delayed','terhegeh','lembab'},
        ),
        'staff': _asp(
            {'staff','worker','pekerja','boss','manager','owner','tukang','chef',
             'cook','barista','crew','team','training','latih'},
            {'professional','skilled','trained','berpengalaman','experienced',
             'knowledgeable','pandai','expert','cekap','efficient','rajin',
             'diligent','hardworking'},
            {'malas','lazy','incompetent','cuai','careless','negligent','bodoh',
             'stupid','useless'},
        ),
    },
    'product': {
        'quality': _asp(
            {'kualiti','quality','bagus','rosak','elok','good','bad','broken',
             'excellent','poor','standard','premium','material','bahan','build',
             'finish','craftsmanship','grade'},
            {'bagus','good','excellent','premium','solid','mantap','padu','terbaik',
             'best','superior','top','high','original','genuine','authentic','elok'},
            {'rosak','broken','poor','bad','defect','cacat','fake','tiruan','kw',
             'flimsy','weak','teruk','worst','rubbish','sampah','hampeh'},
        ),
        'design': _asp(
            {'design','reka bentuk','rupa','look','appearance','style','gaya','warna',
             'color','shape','bentuk','aesthetic','minimal','modern','vintage',
             'ergonomic','compact','slim','thin','tebal','thick','berat','weight',
             'light','ringan'},
            {'cantik','beautiful','gorgeous','sleek','elegant','stylish','modern',
             'cool','aesthetic','nice','comel','cute','kemas','neat','clean'},
            {'hodoh','ugly','buruk','tacky','bulky','clumsy','awkward','janggal',
             'pelik','weird','outdated','ketinggalan'},
        ),
        'price': _asp(
            {'harga','price','cost','mahal','murah','expensive','cheap','affordable',
             'budget','value','worth','berbaloi','overpriced','rm','ringgit','bayar',
             'installment','ansuran','deposit'},
            {'murah','cheap','affordable','berbaloi','worth','value','reasonable',
             'budget'},
            {'mahal','expensive','overpriced','pricey','costly','koyak','pokai',
             'membazir','waste'},
        ),
        'durability': _asp(
            {'tahan','durable','durability','last','longevity','awet','kuat','strong',
             'sturdy','robust','tough','wear','tear','rosak','break','pecah','crack',
             'retak','longgar','loose','warranty','waranti','guarantee'},
            {'tahan','durable','awet','kuat','strong','sturdy','solid','robust',
             'tough','reliable'},
            {'rapuh','fragile','brittle','weak','lemah','flimsy','tak tahan',
             'mudah rosak'},
        ),
        'performance': _asp(
            {'performance','prestasi','speed','kelajuan','power','kuasa','lag',
             'lambat','cepat','fast','slow','smooth','lancar','hang','freeze',
             'crash','boot','load','process','fps','benchmark'},
            {'fast','cepat','pantas','laju','smooth','lancar','powerful','kuat',
             'power','responsive','snappy','efficient','cekap','padu','mantap'},
            {'slow','lambat','lembab','lag','laggy','hang','freeze','crash','stuck',
             'jammed','terhegeh','lemah','weak','sluggish'},
        ),
        'battery': _asp(
            {'battery','bateri','charge','cas','power','drain','habis','tahan',
             'last','hours','jam','adapter','percentage','percent'},
            {'tahan','durable','awet','good battery','fast charge','powerful',
             'jimat','efficient'},
            {'drain','cepat habis','short','lemah','weak','bocor','leak','swelling',
             'kembang','rosak','sekejap habis'},
        ),
        'display': _asp(
            {'display','screen','skrin','monitor','panel','resolution','pixel',
             'brightness','cerah','color','warna','contrast','refresh','hz',
             'oled','lcd','amoled','ips','touchscreen'},
            {'clear','jelas','sharp','tajam','bright','cerah','vivid','colorful',
             'crisp','smooth','responsive','beautiful','cantik','stunning'},
            {'dim','malap','blur','kabur','pixelated','pucat','dead pixel','crack',
             'retak','scratched','calar'},
        ),
        'camera': _asp(
            {'camera','kamera','photo','foto','gambar','video','lens','lensa','zoom',
             'focus','resolution','mp','megapixel','night mode','portrait','selfie',
             'flash'},
            {'clear','jelas','sharp','tajam','detailed','bright','cerah','beautiful',
             'cantik','crisp','stunning','professional','pro'},
            {'blur','kabur','grainy','noisy','dark','gelap','overexposed',
             'underexposed','teruk','bad','disappointing'},
        ),
    },
    'app': {
        'ui': _asp(
            {'ui','interface','display','layout','design','menu','button','butang',
             'navigation','navigasi','icon','ikon','theme','tema','font','text',
             'color','warna','screen','skrin','page','halaman','dashboard','home',
             'sidebar','tab'},
            {'clean','bersih','intuitive','mudah','easy','simple','ringkas','cantik',
             'beautiful','modern','smooth','lancar','responsive','kemas','neat',
             'senang guna'},
            {'confusing','keliru','complicated','rumit','ugly','hodoh','cluttered',
             'serabut','messy','outdated','buruk','susah guna'},
        ),
        'performance': _asp(
            {'performance','speed','kelajuan','loading','load','fast','slow','lambat',
             'cepat','lag','freeze','hang','crash','responsive','smooth','stutter',
             'fps','memory','ram','storage','space'},
            {'fast','cepat','pantas','smooth','lancar','responsive','snappy','quick',
             'instant','efficient','optimized','lightweight','ringan'},
            {'slow','lambat','lag','laggy','freeze','hang','crash','stuck','stutter',
             'berat','heavy','bloated','lembab','terhegeh'},
        ),
        'bugs': _asp(
            {'bug','error','glitch','crash','issue','problem','masalah','rosak',
             'broken','fail','gagal','stuck','loop','freeze','hang','blank'},
            {'fixed','resolved','solved','selesai','patched','working','jalan',
             'berfungsi','ok','fine','stable','stabil'},
            {'bug','error','glitch','crash','broken','fail','gagal','rosak','problem',
             'masalah','issue','annoying','frustrating','teruk','not working'},
        ),
        'features': _asp(
            {'feature','fungsi','function','capability','option','tool','alat',
             'setting','tetapan','preference','mode','mod','plugin','addon',
             'extension','integration','sync','export','import','share'},
            {'useful','berguna','helpful','powerful','hebat','comprehensive','lengkap',
             'complete','versatile','innovative','best','great','bagus','convenient',
             'senang'},
            {'limited','terhad','basic','missing','lacking','kurang','useless',
             'unnecessary','confusing'},
        ),
        'pricing': _asp(
            {'price','harga','subscription','langganan','plan','premium','free',
             'percuma','trial','cuba','pay','bayar','cost','fee','charge','monthly',
             'yearly','annual','lifetime','discount','diskaun'},
            {'free','percuma','cheap','murah','affordable','berbaloi','worth','value',
             'reasonable','fair','berpatutan','generous'},
            {'expensive','mahal','overpriced','pricey','subscription','hidden fee',
             'auto renew','scam'},
        ),
        'support': _asp(
            {'support','sokongan','help','bantuan','service','customer','response',
             'reply','balas','ticket','contact','hubungi','email','chat','phone',
             'faq','documentation','docs','guide','tutorial'},
            {'helpful','responsive','cepat balas','friendly','mesra','professional',
             'knowledgeable','solved','selesai','bagus','good'},
            {'slow','lambat','tak balas','ignore','unhelpful','useless','rude',
             'kasar','teruk','bad','worst','hampeh'},
        ),
        'speed': _asp(
            {'speed','kelajuan','fast','slow','lambat','cepat','loading','download',
             'upload','buffer','stream','instant','segera','delay','lengah'},
            {'fast','cepat','pantas','instant','segera','quick','laju','speedy',
             'snappy'},
            {'slow','lambat','lembab','buffering','loading','delay','lengah',
             'terhegeh','waiting'},
        ),
        'reliability': _asp(
            {'reliable','stabil','stable','consistent','uptime','downtime','offline',
             'online','connection','sambung','disconnect','putus','server','cloud',
             'backup','sync'},
            {'reliable','stabil','stable','consistent','smooth','lancar','dependable',
             'trustworthy','selamat','secure','safe'},
            {'unreliable','unstable','crash','down','offline','disconnect','putus',
             'corrupt','rosak','inconsistent'},
        ),
    },
}

_GENERAL_POSITIVE = {
    'best','bagus','good','great','awesome','amazing','excellent','terbaik','padu',
    'mantap','solid','superb','perfect','hebat','cantik','nice','love','suka','happy',
    'gembira','puas','satisfied','recommend','berbaloi','worth','terharu','impressed',
    'wow','gempak','power','legend','outstanding','brilliant','fantastic','wonderful',
    'ok','okay','fine','lancar','smooth','seamless',
}

_GENERAL_NEGATIVE = {
    'teruk','bad','worst','horrible','terrible','awful','sampah','hampeh','rubbish',
    'trash','useless','bodoh','stupid','hate','benci','kecewa','disappointed',
    'frustrated','frust','menyesal','regret','rugi','waste','fail','gagal','broken',
    'rosak','buruk','poor','sad','annoying','irritating','meluat','disgusting',
    'pathetic','crap','suck',
}


# ============================================================
# Core logic
# ============================================================

def _normalize_elongated(word: str) -> str:
    """Collapse repeated chars (3+) to single."""
    return re.sub(r'(.)\1{2,}', r'\1', word)


def _extract_window(words: List[str], idx: int, before: int = 2, after: int = 4) -> List[str]:
    """Extract word window around index."""
    start = max(0, idx - before)
    end = min(len(words), idx + after)
    return words[start:end]


def _score_window(
    window: List[str],
    positive_set: Set[str],
    negative_set: Set[str],
    aspect_idx_in_window: int = -1,
) -> Tuple[float, List[str], List[str]]:
    """Score a word window for sentiment.

    Splits window at contrast markers (tapi, but, etc.) and only
    scores the side containing the aspect keyword.

    Args:
        window: Word window around aspect keyword.
        positive_set: Positive sentiment words for this aspect.
        negative_set: Negative sentiment words for this aspect.
        aspect_idx_in_window: Index of the aspect keyword within window.
            If -1, no contrast splitting is done.

    Returns:
        (score, pos_found, neg_found)
    """
    # Check for contrast marker in window and split
    contrast_idx = None
    for ci, cw in enumerate(window):
        if _normalize_elongated(cw) in _CONTRAST_MARKERS:
            contrast_idx = ci
            break

    scoring_window = window
    if contrast_idx is not None and aspect_idx_in_window >= 0:
        if aspect_idx_in_window <= contrast_idx:
            # Aspect is before contrast → score words before contrast
            scoring_window = window[:contrast_idx]
        else:
            # Aspect is after contrast → score words after contrast
            scoring_window = window[contrast_idx + 1:]

    pos_found: List[str] = []
    neg_found: List[str] = []
    score = 0.0
    negate_next = False
    multiplier = 1.0

    for w in scoring_window:
        nw = _normalize_elongated(w)
        if nw in _NEGATORS:
            negate_next = True
            continue
        if nw in _INTENSIFIERS:
            multiplier = _INTENSIFIERS[nw]
            continue

        is_pos = nw in positive_set or nw in _GENERAL_POSITIVE
        is_neg = nw in negative_set or nw in _GENERAL_NEGATIVE

        if is_pos:
            val = 1.0 * multiplier
            if negate_next:
                val = -val
                neg_found.append(f"NOT({nw})")
            else:
                pos_found.append(nw)
            score += val
        elif is_neg:
            val = -1.0 * multiplier
            if negate_next:
                val = -val
                pos_found.append(f"NOT({nw})")
            else:
                neg_found.append(nw)
            score += val
        else:
            if nw not in _PARTICLES:
                negate_next = False
                multiplier = 1.0
            continue

        negate_next = False
        multiplier = 1.0

    return score, pos_found, neg_found


def _detect_domain_aspects(words: List[str], domain: str) -> List[Dict[str, Any]]:
    """Detect aspects from a specific domain."""
    if domain not in _DOMAIN_ASPECTS:
        return []

    aspects_data = _DOMAIN_ASPECTS[domain]
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for i, word in enumerate(words):
        nw = _normalize_elongated(word)
        for aspect_name, aspect_data in aspects_data.items():
            if aspect_name in seen:
                continue
            if nw not in aspect_data['keywords']:
                continue

            before = 2
            window = _extract_window(words, i, before=before, after=4)
            # Position of aspect keyword within the window
            aspect_idx_in_window = i - max(0, i - before)
            pos_set = aspect_data.get('positive', set())
            neg_set = aspect_data.get('negative', set())
            raw_score, pos_found, neg_found = _score_window(
                window, pos_set, neg_set, aspect_idx_in_window
            )

            total_words = len(pos_found) + len(neg_found)
            if total_words == 0:
                normalized_score = 0.0
                sentiment_label = 'neutral'
            else:
                normalized_score = max(-1.0, min(1.0, raw_score / max(total_words, 1)))
                if normalized_score > 0.15:
                    sentiment_label = 'positive'
                elif normalized_score < -0.15:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'

            results.append({
                'aspect': aspect_name,
                'sentiment': sentiment_label,
                'score': round(normalized_score, 3),
                'keywords': (pos_found + neg_found)[:5],
                'phrase': ' '.join(window),
            })
            seen.add(aspect_name)
            break

    return results


def _detect_general_aspects(words: List[str]) -> List[Dict[str, Any]]:
    """Dynamic aspect extraction for general domain.

    Groups sentiment words by proximity into clusters, labels each
    cluster by its most representative noun-like word.
    """
    sentiment_positions: List[Tuple[int, str, str]] = []

    for i, word in enumerate(words):
        nw = _normalize_elongated(word)
        if nw in _GENERAL_POSITIVE:
            sentiment_positions.append((i, nw, 'positive'))
        elif nw in _GENERAL_NEGATIVE:
            sentiment_positions.append((i, nw, 'negative'))

    if not sentiment_positions:
        return []

    # Cluster sentiment words by proximity (gap <= 5 words)
    clusters: List[List[Tuple[int, str, str]]] = []
    current_cluster: List[Tuple[int, str, str]] = [sentiment_positions[0]]

    for j in range(1, len(sentiment_positions)):
        prev_idx = current_cluster[-1][0]
        curr_idx = sentiment_positions[j][0]
        if curr_idx - prev_idx <= 5:
            current_cluster.append(sentiment_positions[j])
        else:
            clusters.append(current_cluster)
            current_cluster = [sentiment_positions[j]]
    clusters.append(current_cluster)

    # Build aspect from each cluster
    results: List[Dict[str, Any]] = []
    for cluster in clusters:
        # Find context word (noun-like word near sentiment words)
        first_idx = cluster[0][0]
        last_idx = cluster[-1][0]
        context_start = max(0, first_idx - 2)
        context_end = min(len(words), last_idx + 3)
        context_words = words[context_start:context_end]

        # Label: use first non-sentiment, non-particle word as aspect name
        sentiment_words_set = _GENERAL_POSITIVE | _GENERAL_NEGATIVE
        aspect_label = 'general'
        for cw in context_words:
            ncw = _normalize_elongated(cw)
            if ncw not in sentiment_words_set and ncw not in _PARTICLES and ncw not in _NEGATORS and len(ncw) > 2:
                aspect_label = ncw
                break

        pos_count = sum(1 for _, _, p in cluster if p == 'positive')
        neg_count = sum(1 for _, _, p in cluster if p == 'negative')

        total = pos_count + neg_count
        raw_score = pos_count - neg_count
        normalized_score = max(-1.0, min(1.0, raw_score / max(total, 1)))

        if normalized_score > 0.15:
            sentiment_label = 'positive'
        elif normalized_score < -0.15:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        kw_list = [w for _, w, _ in cluster]

        results.append({
            'aspect': aspect_label,
            'sentiment': sentiment_label,
            'score': round(normalized_score, 3),
            'keywords': kw_list[:5],
            'phrase': ' '.join(context_words),
        })

    return results


def _build_summary(aspects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build summary dict from aspect list."""
    if not aspects:
        return {
            'dominant_sentiment': 'neutral',
            'aspect_count': 0,
            'conflicts': False,
            'overall_score': 0.0,
        }

    scores = [a['score'] for a in aspects]
    sentiments = [a['sentiment'] for a in aspects]
    overall = sum(scores) / len(scores)

    pos_count = sentiments.count('positive')
    neg_count = sentiments.count('negative')

    if pos_count > neg_count:
        dominant = 'positive'
    elif neg_count > pos_count:
        dominant = 'negative'
    else:
        dominant = 'neutral'

    # Conflict: has both positive and negative aspects
    conflicts = pos_count > 0 and neg_count > 0

    return {
        'dominant_sentiment': dominant,
        'aspect_count': len(aspects),
        'conflicts': conflicts,
        'overall_score': round(overall, 3),
    }


# ============================================================
# Public API
# ============================================================

def analyze_aspect_sentiment(text: str, domain: str = 'general') -> Dict[str, Any]:
    """Analyze aspect-based sentiment of text.

    Detects aspects from text and scores sentiment for each aspect.
    Supports domain-specific aspect categories or general dynamic extraction.

    Args:
        text: Input text.
        domain: Domain for aspect categories. One of 'restaurant',
                'product', 'app', or 'general' (default).

    Returns:
        dict: Result with keys:
            - aspects (list[dict]): Per-aspect sentiment results
            - summary (dict): Overall summary
            - domain (str): Domain used
            - text (str): Original input text

    Example:
        >>> analyze_aspect_sentiment("makanan sedap tapi service teruk", domain='restaurant')
        {'aspects': [{'aspect': 'food', 'sentiment': 'positive', ...}, ...], ...}
    """
    if not text or not text.strip():
        return {
            'aspects': [],
            'summary': {
                'dominant_sentiment': 'neutral',
                'aspect_count': 0,
                'conflicts': False,
                'overall_score': 0.0,
            },
            'domain': domain,
            'text': text or '',
        }

    words = _RE_WORDS.findall(text.lower())

    if domain == 'general':
        # Try all domain detectors, merge results
        all_aspects: List[Dict[str, Any]] = []
        for d in ('restaurant', 'product', 'app'):
            detected = _detect_domain_aspects(words, d)
            all_aspects.extend(detected)

        # If no domain aspects found, fall back to general extraction
        if not all_aspects:
            all_aspects = _detect_general_aspects(words)
    else:
        all_aspects = _detect_domain_aspects(words, domain)

    summary = _build_summary(all_aspects)

    return {
        'aspects': all_aspects,
        'summary': summary,
        'domain': domain,
        'text': text,
    }


def aspect_sentiment_batch(texts: List[str], domain: str = 'general') -> List[Dict[str, Any]]:
    """Process multiple texts for aspect-based sentiment.

    Args:
        texts: List of input texts.
        domain: Domain for aspect categories.

    Returns:
        list[dict]: Aspect sentiment results per text.
    """
    return [analyze_aspect_sentiment(t, domain=domain) for t in texts]


def get_aspect_categories() -> Dict[str, List[str]]:
    """Return all available domains and their aspect names.

    Returns:
        dict: Mapping of domain name to list of aspect names.

    Example:
        >>> cats = get_aspect_categories()
        >>> 'food' in cats['restaurant']
        True
    """
    result: Dict[str, List[str]] = {}
    for domain, aspects in _DOMAIN_ASPECTS.items():
        result[domain] = sorted(aspects.keys())
    result['general'] = ['(dynamic)']
    return result
