# Contributing

We welcome contributions to manglish-nlp! Whether it's bug fixes, new features, documentation improvements, or dataset contributions.

---

## How to Contribute

### 1. Find Something to Work On

- Check [open issues](https://github.com/ZafranYusof/manglish-nlp/issues) for bugs and feature requests
- Look for `good first issue` labels for beginner-friendly tasks
- Check `help wanted` for tasks that need community input
- Propose new features by opening a discussion first

### 2. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/manglish-nlp.git
cd manglish-nlp
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 4. Make Your Changes

Follow the development setup below, make changes, write tests, and ensure everything passes.

### 5. Submit a Pull Request

- Push your branch and open a PR against `main`
- Fill in the PR template
- Link related issues
- Wait for review

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Install in Development Mode

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install with all dev dependencies
pip install -e ".[dev,ml,spacy,api]"

# Install pre-commit hooks
pre-commit install
```

### Project Structure

```
manglish-nlp/
├── src/
│   └── manglish_nlp/
│       ├── __init__.py
│       ├── normalize.py
│       ├── sentiment.py
│       ├── ner.py
│       ├── ...
│       ├── models/          # Model weights and configs
│       ├── data/            # Dictionaries, wordlists
│       └── api/             # FastAPI server
├── tests/
│   ├── test_normalize.py
│   ├── test_sentiment.py
│   ├── ...
│   └── fixtures/            # Test data
├── benchmarks/
│   └── run_all.py
├── docs/                    # This documentation
├── pyproject.toml
├── mkdocs.yml
└── README.md
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_sentiment.py

# Run with coverage
pytest --cov=manglish_nlp --cov-report=html

# Run only fast tests (skip ML model tests)
pytest -m "not slow"

# Verbose output
pytest -v
```

### Writing Tests

Every new feature or bug fix should include tests:

```python
# tests/test_your_module.py
import pytest
import manglish_nlp as mnlp


class TestYourModule:
    def test_basic_usage(self):
        result = mnlp.your_module("input text")
        assert result is not None

    def test_manglish_input(self):
        """Should handle informal Manglish text."""
        result = mnlp.your_module("weh best gila bro")
        assert result["score"] > 0.5

    def test_empty_input(self):
        """Should handle empty string gracefully."""
        result = mnlp.your_module("")
        assert result is not None

    def test_batch_input(self):
        """Should accept list of texts."""
        results = mnlp.your_module(["text1", "text2"])
        assert len(results) == 2

    @pytest.mark.slow
    def test_ml_model(self):
        """Test with ML model (requires [ml] extra)."""
        result = mnlp.your_module("text", model="accurate")
        assert result["score"] > 0.8
```

---

## Code Style

### Formatting

We use:
- **Black** for code formatting (line length 88)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/

# Or run all checks at once
pre-commit run --all-files
```

### Conventions

- Use type hints for all public functions
- Write docstrings (Google style) for all public functions
- Keep functions focused — one function, one job
- Prefer explicit over implicit
- Handle edge cases (empty strings, None, lists)

### Example Function

```python
def sentiment(
    text: str | list[str],
    *,
    detailed: bool = False,
    aspect: bool = False,
    cache: bool = False,
) -> dict | list[dict]:
    """Analyze sentiment of Malaysian text.

    Args:
        text: Input text or list of texts to analyze.
        detailed: If True, return scores for all classes.
        aspect: If True, perform aspect-based sentiment analysis.
        cache: If True, cache results for repeated calls.

    Returns:
        Dictionary with 'label' and 'score' keys, or list of dicts
        for batch input.

    Raises:
        InputError: If text is None or not a string/list.

    Example:
        >>> mnlp.sentiment("Best gila!")
        {'label': 'positive', 'score': 0.94}
    """
    ...
```

---

## Adding a New Module

1. Create `src/manglish_nlp/your_module.py`
2. Add the public function with proper type hints and docstring
3. Register in `src/manglish_nlp/__init__.py`
4. Create `tests/test_your_module.py` with comprehensive tests
5. Add documentation in `docs/modules/` (appropriate category)
6. Update `docs/modules/index.md` module count
7. Add to `docs/api-reference.md` if it's a top-level function

---

## Dataset Contributions

We especially welcome:
- Labeled sentiment data (Malaysian social media)
- NER annotations (Malaysian names, places, orgs)
- Code-switching examples
- Dialect samples (Kelantan, Terengganu, etc.)
- Slang/informal vocabulary additions

### Format

```json
{"text": "Best gila nasi lemak tu!", "label": "positive", "source": "twitter"}
{"text": "Teruk la service", "label": "negative", "source": "review"}
```

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- No harassment or discrimination

---

## Questions?

- Open a [GitHub Discussion](https://github.com/ZafranYusof/manglish-nlp/discussions)
- Tag issues with `question` label
