# Contributing

Thanks for wanting to contribute to `malaysian-manglish-nlp`. This guide covers everything you need to know.

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- pip or conda

### Clone and install

```bash
git clone https://github.com/yourusername/malaysian-manglish-nlp.git
cd malaysian-manglish-nlp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Verify setup

```bash
# Run tests
pytest

# Run linters
ruff check .
mypy malaysian_manglish_nlp/

# Check formatting
ruff format --check .
```

If all three pass, you're good to go.

---

## Code Style

We follow a strict but reasonable style:

| Tool | Purpose | Config |
|------|---------|--------|
| **ruff** | Linting + formatting | `pyproject.toml` |
| **mypy** | Type checking (strict) | `mypy.ini` |
| **pytest** | Testing | `pytest.ini` |

### Rules

1. **Type hints everywhere.** All public functions must have full type annotations.
2. **Docstrings.** Google-style docstrings for all public modules, classes, and functions.
3. **No `import *`.** Explicit imports only.
4. **Line length.** 100 characters max.
5. **Naming.** `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE` constants.
6. **Error handling.** Raise typed exceptions (`ManglishNLPError` subclasses), never bare `except`.
7. **No print statements.** Use `logging` module for debug output.
8. **Test everything.** Every public function needs at least 3 test cases (happy path, edge case, error).

### Example

```python
# Good
def tokenize(text: str, *, keep_punct: bool = False) -> list[str]:
    """Tokenize text into word list.

    Args:
        text: Input text to tokenize.
        keep_punct: Whether to preserve punctuation tokens.

    Returns:
        List of word tokens.

    Raises:
        InputError: If text is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise InputError("text must be a non-empty string")
    ...

# Bad
def tokenize(text):
    tokens = text.split()  # no types, no docs, naive impl
    return tokens
```

---

## Adding a New Module

Step-by-step guide to adding a new NLP module.

### 1. Create the module file

```
malaysian_manglish_nlp/
├── your_module.py          # Main implementation
├── tests/
│   └── test_your_module.py # Tests
└── docs/
    └── your_module.md      # Documentation (optional, API ref auto-generates)
```

### 2. Implement with standard interface

Every module must expose at minimum one callable function:

```python
# malaysian_manglish_nlp/your_module.py
from __future__ import annotations

import logging
from typing import Any

from malaysian_manglish_nlp.exceptions import InputError, ModelError

logger = logging.getLogger(__name__)


def your_function(text: str, **kwargs: Any) -> dict[str, Any]:
    """One-line description.

    Args:
        text: Input text.
        **kwargs: Additional options.

    Returns:
        Result dict with at least a 'score' or 'label' key.

    Raises:
        InputError: If input is invalid.
        ModelError: If model inference fails.
    """
    if not text or not text.strip():
        raise InputError("text cannot be empty")

    # Your logic here
    result = _run_model(text, **kwargs)

    return result


def _run_model(text: str, **kwargs: Any) -> dict[str, Any]:
    """Internal model inference. Private function."""
    ...
```

### 3. Register in `__init__.py`

```python
# malaysian_manglish_nlp/__init__.py
from malaysian_manglish_nlp.your_module import your_function

__all__ = [
    # ... existing exports
    "your_function",
]
```

### 4. Write tests

```python
# tests/test_your_module.py
import pytest
from malaysian_manglish_nlp import your_function
from malaysian_manglish_nlp.exceptions import InputError


class TestYourFunction:
    def test_basic(self):
        result = your_function("test input")
        assert isinstance(result, dict)
        assert "score" in result

    def test_manglish_input(self):
        result = your_function("Best gila lah")
        assert result["score"] > 0

    def test_empty_input_raises(self):
        with pytest.raises(InputError):
            your_function("")

    def test_chinese_mixed(self):
        result = your_function("这个 sangat bagus")
        assert result is not None

    # Add edge cases specific to your module
```

### 5. Add benchmarks (optional but recommended)

```python
# benchmarks/bench_your_module.py
from malaysian_manglish_nlp import your_function
from benchmarks.utils import timed_run

def benchmark():
    texts = load_test_corpus()
    latency, throughput = timed_run(your_function, texts)
    print(f"Latency: {latency:.2f}ms | Throughput: {throughput:.0f} tps")

if __name__ == "__main__":
    benchmark()
```

### 6. Update documentation

Add your function to `docs/api-reference.md` following the existing format.

---

## Adding Tests

### Test structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_sentiment.py        # One file per module
├── test_ner.py
├── test_pipeline.py
├── test_exceptions.py
└── integration/
    ├── test_full_pipeline.py
    └── test_model_loading.py
```

### Test requirements

- **Minimum 3 tests per function**: happy path, edge case, error case
- **Manglish-specific inputs**: Always test with real Manglish text
- **Edge cases**: Empty strings, very long text (>10k chars), special characters, mixed scripts
- **Regression tests**: When fixing a bug, add a test that would have caught it

### Running tests

```bash
# All tests
pytest

# Single module
pytest tests/test_sentiment.py -v

# With coverage
pytest --cov=malaysian_manglish_nlp --cov-report=term-missing

# Parallel (faster for large suites)
pytest -n auto
```

### Coverage target

We aim for **90%+ line coverage** on all public modules. Check current coverage:

```bash
pytest --cov=malaysian_manglish_nlp --cov-report=html
# Open htmlcov/index.html
```

---

## Pull Request Process

### Before you submit

1. **Run full test suite**  -  `pytest` must pass
2. **Run linters**  -  `ruff check .` and `mypy malaysian_manglish_nlp/` must be clean
3. **Format code**  -  `ruff format .`
4. **Update docs**  -  If you changed public API
5. **Add changelog entry**  -  Describe your change in `docs/changelog.md` under `[Unreleased]`

### PR checklist

- [ ] Tests pass locally
- [ ] Linters pass
- [ ] New tests added for new functionality
- [ ] Docstrings updated
- [ ] API reference updated (if public API changed)
- [ ] Changelog entry added
- [ ] No hardcoded secrets or credentials
- [ ] Backward compatible (or migration path documented)

### PR naming convention

```
type: short description

Examples:
feat: add emotion detection module
fix: sentiment misclassifies negated sarcasm
perf: 2x faster tokenization with pre-compiled regex
docs: improve normalize() examples
test: add edge cases for code-switching detection
refactor: simplify pipeline step chaining
```

### Review process

1. **Auto-checks**: CI runs tests + linters on push
2. **Review**: At least 1 maintainer approval required
3. **Squash merge**: All PRs squashed to single commit
4. **Release**: Maintainers handle versioning and publishing

---

## Reporting Bugs

Open a GitHub issue with:

```markdown
## Bug Report

**Module**: `sentiment` / `ner` / etc.
**Version**: `malaysian-manglish-nlp==3.0.0`
**Python**: 3.11.7
**OS**: Windows 11 / Ubuntu 22.04 / macOS 14

### What happened

<clear description>

### Expected behavior

<what should have happened>

### Reproduction

```python
from malaysian_manglish_nlp import sentiment
sentiment("this caused the bug")
# Error: ...
```

### Input that triggered it

<paste exact text>

### Full traceback

<paste full error output>
```

---

## Feature Requests

Open a GitHub issue with:

```markdown
## Feature Request

### What

<one sentence description>

### Why

<use case, why this matters>

### Proposed API

```python
from malaysian_manglish_nlp import new_thing
result = new_thing("input text")
# expected output
```

### Alternatives considered

<what else did you think about?>
```

---

## Community

- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: Questions, ideas, showcases
- **Discord**: Real-time chat with maintainers and users

Be respectful. We're all building something for the Malaysian NLP community.
