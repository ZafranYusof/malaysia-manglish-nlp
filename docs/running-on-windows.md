# Running on Windows

malaysian-manglish-nlp works on Windows, but a few things need extra attention.

---

## Common Issues

### UnicodeEncodeError

The most common Windows issue. Happens when printing Malay text with special characters (é, ñ, etc.) to a console that doesn't support UTF-8.

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2019'
```

**Fix:** Set environment variable before running your script:

```powershell
# PowerShell
$env:PYTHONIOENCODING="utf-8"
python my_script.py
```

```cmd
:: CMD
set PYTHONIOENCODING=utf-8
python my_script.py
```

Or add to your script's top:

```python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

---

### Gensim build issues

Gensim requires a C compiler for its fast Cython extensions. On Windows this means Visual Studio Build Tools.

**Fix:**

1. Install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select "Desktop development with C++"
3. Then install:

```bash
pip install gensim
```

If you don't want to install build tools, Gensim ships pre-built wheels for most Python versions. Make sure pip is up to date:

```bash
python -m pip install --upgrade pip
pip install gensim
```

---

### torch / transformers slow on first import

The first time you import `transformers` or `torch` on Windows, it can take 10–30 seconds. This is normal  -  PyTorch initialises CUDA and loads DLLs.

**Fix:** Nothing to fix. Subsequent imports are fast within the same process.

---

### File path issues

Windows uses `\` as path separator. Python handles this fine, but if you're writing config files or shell scripts:

```python
# Good  -  works everywhere
from pathlib import Path
data_dir = Path(__file__).parent / "data"

# Also fine
import os
data_dir = os.path.join(os.path.dirname(__file__), "data")
```

---

## Recommended Setup

### 1. Use a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install malaysian-manglish-nlp
```

### 2. PowerShell execution policy

If activation fails with a policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install extras as needed

```bash
# Core only (no ML models)
pip install malaysian-manglish-nlp

# With sentiment classifier
pip install malaysian-manglish-nlp[transformers]

# With word embeddings
pip install malaysian-manglish-nlp[embeddings]

# Everything
pip install malaysian-manglish-nlp[all]
```

### 4. Test the install

```python
python -c "import malaysian_manglish_nlp; print(malaysian_manglish_nlp.__version__)"
```

---

## Windows-specific Notes

| Feature | Status on Windows | Notes |
|---------|-------------------|-------|
| Core modules (51) | ✅ Full | No platform dependencies |
| Rule-based sentiment | ✅ Full | Pure Python |
| Fine-tuned classifier | ✅ Full | Needs `[transformers]` extra |
| Word embeddings | ✅ Full | Needs `[embeddings]` extra |
| FastAPI server | ✅ Full | Needs `[api]` extra |
| Graph visualisation | ⚠️ Partial | `pyvis` needs a browser to render |
| Parallel training | ⚠️ Slower | Windows lacks `fork()`; uses `spawn` |

---

## Getting Help

If you hit a Windows-specific bug:

1. Check the [GitHub Issues](https://github.com/ZafranYusof/malaysian-manglish-nlp/issues)
2. Include your Python version, Windows version, and full traceback
3. Run `python -m malaysian_manglish_nlp.debug` to collect system info
