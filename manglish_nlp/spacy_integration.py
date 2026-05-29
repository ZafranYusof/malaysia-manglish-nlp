"""
spaCy integration for manglish-nlp.

Provides a custom spaCy Language class and pipeline components for processing
Malaysian Manglish text with spaCy's familiar API.

Usage:
    from manglish_nlp.spacy_integration import create_manglish_nlp
    nlp = create_manglish_nlp()
    doc = nlp("aku nak pergi makan nasi lemak kat KL")
    for token in doc:
        print(token.text, token.pos_, token.tag_)
    for ent in doc.ents:
        print(ent.text, ent.label_)
    print(doc._.sentiment)
    print(doc._.language)

Requires: spacy>=3.0
    pip install manglish-nlp[spacy]
"""

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
    from spacy.vocab import Vocab
    from spacy.util import registry
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


def _check_spacy():
    """Raise ImportError if spaCy is not installed."""
    if not SPACY_AVAILABLE:
        raise ImportError(
            "spaCy is required for spacy_integration. "
            "Install it with: pip install manglish-nlp[spacy] "
            "or: pip install spacy>=3.0"
        )


# ============================================================================
# Custom Tokenizer
# ============================================================================

class ManglishTokenizer:
    """Custom spaCy tokenizer that uses manglish_nlp's tokenizer."""

    def __init__(self, vocab):
        """Initialize with a spaCy Vocab object."""
        self.vocab = vocab

    def __call__(self, text):
        """Tokenize text using manglish_nlp tokenizer and return a spaCy Doc."""
        from manglish_nlp.tokenizer import word_tokenize

        if not text or not text.strip():
            return Doc(self.vocab, words=[], spaces=[])

        words = word_tokenize(text)
        if not words:
            return Doc(self.vocab, words=[], spaces=[])

        # Compute spaces by checking if each token is followed by a space
        spaces = []
        current_pos = 0
        for i, word in enumerate(words):
            # Find the word in the original text
            word_start = text.find(word, current_pos)
            if word_start == -1:
                word_start = current_pos
            word_end = word_start + len(word)
            current_pos = word_end

            # Check if there's a space after this word
            if i < len(words) - 1:
                spaces.append(word_end < len(text) and text[word_end:word_end + 1] == ' ')
            else:
                spaces.append(False)

        return Doc(self.vocab, words=words, spaces=spaces)


# ============================================================================
# Custom Language Class
# ============================================================================

if SPACY_AVAILABLE:
    @Language.factory("manglish_normalizer")
    def create_normalizer(nlp, name):
        """Factory for the Manglish normalizer component."""
        return ManglishNormalizerComponent(nlp, name)

    @Language.factory("manglish_sentiment")
    def create_sentiment(nlp, name):
        """Factory for the Manglish sentiment component."""
        return ManglishSentimentComponent(nlp, name)

    @Language.factory("manglish_ner")
    def create_ner(nlp, name):
        """Factory for the Manglish NER component."""
        return ManglishNERComponent(nlp, name)

    @Language.factory("manglish_pos")
    def create_pos(nlp, name):
        """Factory for the Manglish POS tagging component."""
        return ManglishPOSComponent(nlp, name)

    @Language.factory("manglish_language_detector")
    def create_language_detector(nlp, name):
        """Factory for the Manglish language detector component."""
        return ManglishLanguageDetectorComponent(nlp, name)

    class ManglishDefaults(Language.Defaults):
        """Default settings for Manglish language."""
        pass

    @registry.languages("ms_manglish")
    class ManglishLanguage(Language):
        """Custom spaCy Language class for Malaysian Manglish.

        Supports the informal mix of Bahasa Melayu, English, and local slang
        used by Malaysians in everyday communication.
        """
        lang = "ms_manglish"
        Defaults = ManglishDefaults

        def __init__(self, vocab=True, **kwargs):
            """Initialize ManglishLanguage with custom tokenizer."""
            super().__init__(vocab=vocab, **kwargs)
            self.tokenizer = ManglishTokenizer(self.vocab)


# ============================================================================
# Pipeline Components
# ============================================================================

class ManglishNormalizerComponent:
    """spaCy pipeline component for normalizing Manglish shortforms.

    Expands common abbreviations (nk→nak, brp→berapa, etc.) and stores
    the normalized form in token._.normalized and doc._.normalized.
    """

    def __init__(self, nlp, name):
        self.name = name
        # Register extensions if not already set
        if not Doc.has_extension("normalized"):
            Doc.set_extension("normalized", default=None)
        if not Token.has_extension("normalized"):
            Token.set_extension("normalized", default=None)

    def __call__(self, doc):
        """Normalize each token and the full document text."""
        from manglish_nlp.normalize import normalize

        # Normalize individual tokens
        for token in doc:
            normalized = normalize(token.text)
            token._.normalized = normalized

        # Normalize full document text
        doc._.normalized = normalize(doc.text)

        return doc


class ManglishSentimentComponent:
    """spaCy pipeline component for Manglish sentiment analysis.

    Adds sentiment scores to Doc and Span objects via custom extensions.
    Understands Malaysian slang, particles, and code-switching patterns.
    """

    def __init__(self, nlp, name):
        self.name = name
        # Register extensions if not already set
        if not Doc.has_extension("sentiment"):
            Doc.set_extension("sentiment", default=None)
        if not Span.has_extension("sentiment"):
            Span.set_extension("sentiment", default=None)

    def __call__(self, doc):
        """Analyze sentiment for the document and each sentence."""
        from manglish_nlp.sentiment import analyze_sentiment

        # Document-level sentiment
        if doc.text.strip():
            doc._.sentiment = analyze_sentiment(doc.text)
        else:
            doc._.sentiment = {"label": "neutral", "score": 0.0}

        # Sentence-level sentiment (if sentences are available)
        try:
            for sent in doc.sents:
                if sent.text.strip():
                    sent._.sentiment = analyze_sentiment(sent.text)
                else:
                    sent._.sentiment = {"label": "neutral", "score": 0.0}
        except ValueError:
            # No sentence boundaries set
            pass

        return doc


class ManglishNERComponent:
    """spaCy pipeline component for Manglish Named Entity Recognition.

    Recognizes entities like PERSON, LOCATION, ORGANIZATION, FOOD, etc.
    that are specific to Malaysian context.
    """

    def __init__(self, nlp, name):
        self.name = name

    def __call__(self, doc):
        """Run NER on the document and set doc.ents."""
        from manglish_nlp.ner import ner_tag

        if not doc.text.strip():
            return doc

        # Get NER results from manglish_nlp
        entities = ner_tag(doc.text)

        # Convert to spaCy Span entities
        spans = []
        for entity in entities:
            if isinstance(entity, dict):
                text = entity.get("text", entity.get("entity", ""))
                label = entity.get("label", entity.get("type", "MISC"))
                start_char = entity.get("start", -1)
                end_char = entity.get("end", -1)

                if start_char >= 0 and end_char > start_char:
                    # Find token indices from char offsets
                    span = doc.char_span(start_char, end_char, label=label)
                    if span is not None:
                        spans.append(span)
                else:
                    # Try to find the entity text in the doc
                    entity_text = text
                    for i in range(len(doc)):
                        candidate = doc[i:i + len(entity_text.split())]
                        if candidate.text == entity_text or candidate.text.lower() == entity_text.lower():
                            span = Span(doc, i, i + len(entity_text.split()), label=label)
                            spans.append(span)
                            break
            elif isinstance(entity, (list, tuple)) and len(entity) >= 2:
                # Format: (text, label) or [text, label]
                text = entity[0]
                label = entity[1]
                # Find in doc
                entity_words = text.split()
                for i in range(len(doc) - len(entity_words) + 1):
                    candidate = doc[i:i + len(entity_words)]
                    if candidate.text == text or candidate.text.lower() == text.lower():
                        span = Span(doc, i, i + len(entity_words), label=label)
                        spans.append(span)
                        break

        # Filter overlapping spans (keep longest)
        if spans:
            spans = spacy.util.filter_spans(spans)

        try:
            doc.ents = spans
        except ValueError:
            # If there's a conflict with existing entities, try to merge
            doc.ents = list(doc.ents) + spans if not doc.ents else spans

        return doc


class ManglishPOSComponent:
    """spaCy pipeline component for Manglish Part-of-Speech tagging.

    Tags tokens with POS tags appropriate for Manglish text,
    handling particles (lah, kan, kot) and code-switching.
    """

    def __init__(self, nlp, name):
        self.name = name

    # Mapping from manglish_nlp POS tags to Universal POS tags
    POS_MAP = {
        "NOUN": "NOUN",
        "VERB": "VERB",
        "ADJ": "ADJ",
        "ADV": "ADV",
        "PRON": "PRON",
        "DET": "DET",
        "ADP": "ADP",
        "CONJ": "CCONJ",
        "CCONJ": "CCONJ",
        "SCONJ": "SCONJ",
        "NUM": "NUM",
        "PART": "PART",
        "PUNCT": "PUNCT",
        "INTJ": "INTJ",
        "PROPN": "PROPN",
        "AUX": "AUX",
        "X": "X",
        "SYM": "SYM",
    }

    def __call__(self, doc):
        """Tag each token with POS information."""
        from manglish_nlp.pos import pos_tag

        if not doc.text.strip():
            return doc

        # Get POS tags from manglish_nlp
        tagged = pos_tag(doc.text)

        # Apply tags to tokens
        if tagged and len(tagged) > 0:
            tag_idx = 0
            for token in doc:
                if tag_idx < len(tagged):
                    tag_item = tagged[tag_idx]
                    if isinstance(tag_item, (list, tuple)) and len(tag_item) >= 2:
                        tag = tag_item[1]
                    elif isinstance(tag_item, dict):
                        tag = tag_item.get("tag", tag_item.get("pos", "X"))
                    else:
                        tag = "X"

                    # Set the fine-grained tag
                    token.tag_ = tag
                    # Map to universal POS
                    token.pos_ = self.POS_MAP.get(tag.upper(), "X")
                    tag_idx += 1
                else:
                    token.tag_ = "X"
                    token.pos_ = "X"

        return doc


class ManglishLanguageDetectorComponent:
    """spaCy pipeline component for detecting language per sentence.

    Classifies text segments as Bahasa Melayu (BM), English (EN),
    or Manglish (mixed), useful for code-switching analysis.
    """

    def __init__(self, nlp, name):
        self.name = name
        # Register extensions if not already set
        if not Doc.has_extension("language"):
            Doc.set_extension("language", default=None)
        if not Span.has_extension("language"):
            Span.set_extension("language", default=None)

    def __call__(self, doc):
        """Detect language for the document and each sentence."""
        from manglish_nlp.language import detect_language

        if not doc.text.strip():
            doc._.language = {"label": "unknown", "confidence": 0.0}
            return doc

        # Document-level language detection
        result = detect_language(doc.text)
        if isinstance(result, dict):
            doc._.language = result
        elif isinstance(result, str):
            doc._.language = {"label": result, "confidence": 1.0}
        else:
            doc._.language = {"label": str(result), "confidence": 1.0}

        # Sentence-level language detection
        try:
            for sent in doc.sents:
                if sent.text.strip():
                    sent_result = detect_language(sent.text)
                    if isinstance(sent_result, dict):
                        sent._.language = sent_result
                    elif isinstance(sent_result, str):
                        sent._.language = {"label": sent_result, "confidence": 1.0}
                    else:
                        sent._.language = {"label": str(sent_result), "confidence": 1.0}
                else:
                    sent._.language = {"label": "unknown", "confidence": 0.0}
        except ValueError:
            # No sentence boundaries set
            pass

        return doc


# ============================================================================
# Factory Function
# ============================================================================

def create_manglish_nlp(components=None):
    """Create a spaCy nlp object configured for Manglish processing.

    Args:
        components: List of component names to include. If None, includes all:
            ['manglish_normalizer', 'manglish_pos', 'manglish_ner',
             'manglish_sentiment', 'manglish_language_detector']

    Returns:
        spaCy Language object with Manglish pipeline components.

    Raises:
        ImportError: If spaCy is not installed.

    Example:
        >>> from manglish_nlp.spacy_integration import create_manglish_nlp
        >>> nlp = create_manglish_nlp()
        >>> doc = nlp("aku nak pergi makan nasi lemak kat KL")
        >>> for token in doc:
        ...     print(token.text, token.pos_, token.tag_)
        >>> print(doc._.sentiment)
        >>> print(doc._.language)
    """
    _check_spacy()

    # Create the custom language instance
    nlp = ManglishLanguage()

    # Default components in recommended order
    default_components = [
        "manglish_normalizer",
        "manglish_pos",
        "manglish_ner",
        "manglish_sentiment",
        "manglish_language_detector",
    ]

    if components is None:
        components = default_components

    # Add sentencizer first for sentence-level components
    if any(c in components for c in ["manglish_sentiment", "manglish_language_detector"]):
        nlp.add_pipe("sentencizer")

    # Add requested components
    for component in components:
        if component in default_components:
            nlp.add_pipe(component)

    return nlp
