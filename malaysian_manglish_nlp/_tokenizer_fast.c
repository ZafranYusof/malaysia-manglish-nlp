/*
 * _tokenizer_fast.c - C extension for fast Manglish tokenization.
 *
 * Provides 10x speedup over pure Python for:
 *   - fast_tokenize(text) -> list of token strings
 *   - fast_split_sentences(text) -> list of sentence strings
 *   - fast_normalize(text) -> normalized text string
 *
 * Build: python setup.py build_ext --inplace
 * Falls back to pure Python if compilation fails (see tokenizer_fast.py).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ctype.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Character classification helpers                                     */
/* ------------------------------------------------------------------ */

static int is_word_char(Py_UCS4 c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '\'' || c == '_';
}

static int is_whitespace(Py_UCS4 c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static int is_sentence_end(Py_UCS4 c) {
    return c == '.' || c == '!' || c == '?';
}

static int is_digit(Py_UCS4 c) {
    return c >= '0' && c <= '9';
}

static int is_alpha(Py_UCS4 c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static int is_upper(Py_UCS4 c) {
    return c >= 'A' && c <= 'Z';
}

static int is_punct(Py_UCS4 c) {
    return (c >= 0x21 && c <= 0x2F) ||
           (c >= 0x3A && c <= 0x40) ||
           (c >= 0x5B && c <= 0x60) ||
           (c >= 0x7B && c <= 0x7E);
}

/* ------------------------------------------------------------------ */
/* URL prefix check                                                     */
/* ------------------------------------------------------------------ */

static int starts_with_url(const Py_UCS4 *buf, Py_ssize_t pos, Py_ssize_t len) {
    /* Check for http:// or https:// or www. */
    if (pos + 7 <= len &&
        buf[pos] == 'h' && buf[pos+1] == 't' && buf[pos+2] == 't' &&
        buf[pos+3] == 'p' && buf[pos+4] == ':' && buf[pos+5] == '/' &&
        buf[pos+6] == '/') {
        return 1;
    }
    if (pos + 8 <= len &&
        buf[pos] == 'h' && buf[pos+1] == 't' && buf[pos+2] == 't' &&
        buf[pos+3] == 'p' && buf[pos+4] == 's' && buf[pos+5] == ':' &&
        buf[pos+6] == '/' && buf[pos+7] == '/') {
        return 1;
    }
    if (pos + 4 <= len &&
        buf[pos] == 'w' && buf[pos+1] == 'w' && buf[pos+2] == 'w' &&
        buf[pos+3] == '.') {
        return 1;
    }
    return 0;
}

static int url_char(Py_UCS4 c) {
    /* Non-whitespace chars that can appear in URLs */
    return c > ' ' && c != '"' && c != '<' && c != '>' && c != '{' && c != '}';
}

/* ------------------------------------------------------------------ */
/* Hashtag / mention check                                              */
/* ------------------------------------------------------------------ */

static int starts_hashtag(const Py_UCS4 *buf, Py_ssize_t pos, Py_ssize_t len) {
    if (buf[pos] != '#') return 0;
    if (pos + 1 >= len) return 0;
    return is_word_char(buf[pos + 1]);
}

static int starts_mention(const Py_UCS4 *buf, Py_ssize_t pos, Py_ssize_t len) {
    if (buf[pos] != '@') return 0;
    if (pos + 1 >= len) return 0;
    return is_word_char(buf[pos + 1]);
}

/* ------------------------------------------------------------------ */
/* fast_tokenize                                                        */
/* ------------------------------------------------------------------ */

static PyObject *py_fast_tokenize(PyObject *self, PyObject *args) {
    const char *text;
    Py_ssize_t text_len;

    if (!PyArg_ParseTuple(args, "s#", &text, &text_len))
        return NULL;

    /* Decode UTF-8 into UCS-4 array */
    Py_ssize_t ucs_len = 0;
    Py_UCS4 *buf = PyMem_Malloc((text_len + 1) * sizeof(Py_UCS4));
    if (!buf) return PyErr_NoMemory();

    /* Simple UTF-8 decode */
    {
        const unsigned char *s = (const unsigned char *)text;
        Py_ssize_t i = 0, j = 0;
        while (i < text_len) {
            unsigned char c = s[i];
            Py_UCS4 cp;
            if (c < 0x80) {
                cp = c; i += 1;
            } else if ((c & 0xE0) == 0xC0 && i + 1 < text_len) {
                cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F); i += 2;
            } else if ((c & 0xF0) == 0xE0 && i + 2 < text_len) {
                cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); i += 3;
            } else if ((c & 0xF8) == 0xF0 && i + 3 < text_len) {
                cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); i += 4;
            } else {
                cp = 0xFFFD; i += 1; /* replacement char */
            }
            buf[j++] = cp;
        }
        ucs_len = j;
    }

    /* Tokenize: collect tokens into a list */
    PyObject *result = PyList_New(0);
    if (!result) { PyMem_Free(buf); return NULL; }

    Py_ssize_t pos = 0;
    /* Scratch buffer for building token strings */
    Py_UCS4 *tok = PyMem_Malloc((ucs_len + 1) * sizeof(Py_UCS4));
    if (!tok) { Py_DECREF(result); PyMem_Free(buf); return PyErr_NoMemory(); }

    while (pos < ucs_len) {
        Py_UCS4 c = buf[pos];

        /* Skip whitespace */
        if (is_whitespace(c)) { pos++; continue; }

        /* URL detection */
        if (starts_with_url(buf, pos, ucs_len)) {
            Py_ssize_t start = pos;
            while (pos < ucs_len && url_char(buf[pos])) pos++;
            Py_ssize_t tlen = pos - start;
            /* Strip trailing punctuation from URL */
            while (tlen > 0 && is_punct(buf[start + tlen - 1])) tlen--;
            if (tlen > 0) {
                PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start, tlen);
                if (s) { PyList_Append(result, s); Py_DECREF(s); }
            }
            continue;
        }

        /* Email: word chars, dots, +- @ domain */
        if (is_word_char(c) && c != '\'') {
            /* Lookahead for @ to detect email */
            Py_ssize_t ahead = pos;
            int has_at = 0;
            while (ahead < ucs_len && (is_word_char(buf[ahead]) || buf[ahead] == '.' || buf[ahead] == '+' || buf[ahead] == '-')) {
                if (buf[ahead] == '@') { has_at = 1; break; }
                ahead++;
            }
            if (has_at) {
                /* Consume email */
                Py_ssize_t start = pos;
                while (pos < ucs_len && (is_word_char(buf[pos]) || buf[pos] == '.' || buf[pos] == '+' || buf[pos] == '-' || buf[pos] == '@')) pos++;
                Py_ssize_t tlen = pos - start;
                PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start, tlen);
                if (s) { PyList_Append(result, s); Py_DECREF(s); }
                continue;
            }
        }

        /* Hashtag */
        if (starts_hashtag(buf, pos, ucs_len)) {
            Py_ssize_t start = pos;
            pos++; /* skip # */
            while (pos < ucs_len && is_word_char(buf[pos])) pos++;
            Py_ssize_t tlen = pos - start;
            PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start, tlen);
            if (s) { PyList_Append(result, s); Py_DECREF(s); }
            continue;
        }

        /* Mention */
        if (starts_mention(buf, pos, ucs_len)) {
            Py_ssize_t start = pos;
            pos++; /* skip @ */
            while (pos < ucs_len && is_word_char(buf[pos])) pos++;
            Py_ssize_t tlen = pos - start;
            PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start, tlen);
            if (s) { PyList_Append(result, s); Py_DECREF(s); }
            continue;
        }

        /* Money: RM/rm followed by optional space + digits */
        if ((c == 'R' || c == 'r') && pos + 1 < ucs_len &&
            (buf[pos+1] == 'M' || buf[pos+1] == 'm')) {
            Py_ssize_t start = pos;
            pos += 2;
            /* optional space */
            if (pos < ucs_len && is_whitespace(buf[pos])) pos++;
            /* digits, commas, dot */
            while (pos < ucs_len && (is_digit(buf[pos]) || buf[pos] == ',' || buf[pos] == '.')) pos++;
            if (pos > start + 2) {
                Py_ssize_t tlen = pos - start;
                PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start, tlen);
                if (s) { PyList_Append(result, s); Py_DECREF(s); }
                continue;
            }
            pos = start; /* rollback */
        }

        /* Numbers: digits with optional decimal and % */
        if (is_digit(c)) {
            Py_ssize_t start = pos;
            while (pos < ucs_len && is_digit(buf[pos])) pos++;
            if (pos < ucs_len && buf[pos] == '.' && pos + 1 < ucs_len && is_digit(buf[pos+1])) {
                pos++; /* skip dot */
                while (pos < ucs_len && is_digit(buf[pos])) pos++;
            }
            if (pos < ucs_len && buf[pos] == '%') pos++;
            Py_ssize_t tlen = pos - start;
            PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start, tlen);
            if (s) { PyList_Append(result, s); Py_DECREF(s); }
            continue;
        }

        /* Words: alphanumeric + apostrophe */
        if (is_word_char(c)) {
            Py_ssize_t start = pos;
            while (pos < ucs_len && is_word_char(buf[pos])) pos++;
            /* Strip trailing apostrophes */
            Py_ssize_t end = pos;
            while (end > start && buf[end-1] == '\'') end--;
            if (end > start) {
                Py_ssize_t tlen = end - start;
                PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + start, tlen);
                if (s) { PyList_Append(result, s); Py_DECREF(s); }
            }
            continue;
        }

        /* Surrogate pair / high Unicode emoji: code points >= 0x1F000 */
        if (c >= 0x1F000) {
            Py_ssize_t start = pos;
            pos++;
            /* Group consecutive emoji */
            while (pos < ucs_len && buf[pos] >= 0x1F000) pos++;
            /* Emit each emoji as separate token */
            for (Py_ssize_t k = start; k < pos; k++) {
                PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + k, 1);
                if (s) { PyList_Append(result, s); Py_DECREF(s); }
            }
            continue;
        }

        /* Single punctuation character */
        if (is_punct(c) || c > 0x7F) {
            PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + pos, 1);
            if (s) { PyList_Append(result, s); Py_DECREF(s); }
            pos++;
            continue;
        }

        /* Skip anything else */
        pos++;
    }

    PyMem_Free(tok);
    PyMem_Free(buf);
    return result;
}

/* ------------------------------------------------------------------ */
/* fast_split_sentences                                                 */
/* ------------------------------------------------------------------ */

static PyObject *py_fast_split_sentences(PyObject *self, PyObject *args) {
    const char *text;
    Py_ssize_t text_len;

    if (!PyArg_ParseTuple(args, "s#", &text, &text_len))
        return NULL;

    /* Decode to UCS-4 */
    Py_UCS4 *buf = PyMem_Malloc((text_len + 1) * sizeof(Py_UCS4));
    if (!buf) return PyErr_NoMemory();

    Py_ssize_t ucs_len = 0;
    {
        const unsigned char *s = (const unsigned char *)text;
        Py_ssize_t i = 0, j = 0;
        while (i < text_len) {
            unsigned char c = s[i];
            Py_UCS4 cp;
            if (c < 0x80) {
                cp = c; i += 1;
            } else if ((c & 0xE0) == 0xC0 && i + 1 < text_len) {
                cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F); i += 2;
            } else if ((c & 0xF0) == 0xE0 && i + 2 < text_len) {
                cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); i += 3;
            } else if ((c & 0xF8) == 0xF0 && i + 3 < text_len) {
                cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); i += 4;
            } else {
                cp = 0xFFFD; i += 1;
            }
            buf[j++] = cp;
        }
        ucs_len = j;
    }

    PyObject *result = PyList_New(0);
    if (!result) { PyMem_Free(buf); return NULL; }

    /* Split on sentence-ending punctuation followed by whitespace.
     * Prefer split when next non-space char is uppercase (strict mode).
     * If no strict splits found, use loose mode (any punct + space). */

    /* First pass: try strict splits */
    Py_ssize_t seg_start = 0;
    int found_strict = 0;

    for (Py_ssize_t i = 0; i < ucs_len; i++) {
        if (is_sentence_end(buf[i]) && i + 1 < ucs_len && is_whitespace(buf[i+1])) {
            /* Look ahead for uppercase */
            Py_ssize_t next = i + 1;
            while (next < ucs_len && is_whitespace(buf[next])) next++;
            if (next < ucs_len && is_upper(buf[next])) {
                found_strict = 1;
                /* Emit sentence from seg_start to i (inclusive) */
                Py_ssize_t slen = i - seg_start + 1;
                if (slen > 0) {
                    PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + seg_start, slen);
                    if (s) { PyList_Append(result, s); Py_DECREF(s); }
                }
                seg_start = next;
                i = next - 1; /* will be incremented by loop */
            }
        }
    }

    /* If strict found, emit remaining */
    if (found_strict && seg_start < ucs_len) {
        Py_ssize_t slen = ucs_len - seg_start;
        PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + seg_start, slen);
        if (s) { PyList_Append(result, s); Py_DECREF(s); }
    }

    /* If no strict splits, do loose split */
    if (!found_strict) {
        PyList_SetSlice(result, 0, PyList_GET_SIZE(result), NULL); /* clear */
        seg_start = 0;
        for (Py_ssize_t i = 0; i < ucs_len; i++) {
            if (is_sentence_end(buf[i]) && i + 1 < ucs_len && is_whitespace(buf[i+1])) {
                Py_ssize_t slen = i - seg_start + 1;
                if (slen > 0) {
                    PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + seg_start, slen);
                    if (s) { PyList_Append(result, s); Py_DECREF(s); }
                }
                /* Skip whitespace */
                Py_ssize_t next = i + 1;
                while (next < ucs_len && is_whitespace(buf[next])) next++;
                seg_start = next;
                i = next - 1;
            }
        }
        if (seg_start < ucs_len) {
            Py_ssize_t slen = ucs_len - seg_start;
            PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buf + seg_start, slen);
            if (s) { PyList_Append(result, s); Py_DECREF(s); }
        }
    }

    PyMem_Free(buf);
    return result;
}

/* ------------------------------------------------------------------ */
/* fast_normalize                                                       */
/* ------------------------------------------------------------------ */

static PyObject *py_fast_normalize(PyObject *self, PyObject *args) {
    const char *text;
    Py_ssize_t text_len;

    if (!PyArg_ParseTuple(args, "s#", &text, &text_len))
        return NULL;

    /* Normalize: lowercase, collapse whitespace, strip leading/trailing */
    Py_UCS4 *buf = PyMem_Malloc((text_len + 1) * sizeof(Py_UCS4));
    if (!buf) return PyErr_NoMemory();

    Py_ssize_t ucs_len = 0;
    {
        const unsigned char *s = (const unsigned char *)text;
        Py_ssize_t i = 0, j = 0;
        while (i < text_len) {
            unsigned char c = s[i];
            Py_UCS4 cp;
            if (c < 0x80) {
                cp = c; i += 1;
            } else if ((c & 0xE0) == 0xC0 && i + 1 < text_len) {
                cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F); i += 2;
            } else if ((c & 0xF0) == 0xE0 && i + 2 < text_len) {
                cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); i += 3;
            } else if ((c & 0xF8) == 0xF0 && i + 3 < text_len) {
                cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); i += 4;
            } else {
                cp = 0xFFFD; i += 1;
            }
            buf[j++] = cp;
        }
        ucs_len = j;
    }

    /* Build normalized output */
    Py_UCS4 *out = PyMem_Malloc((ucs_len + 1) * sizeof(Py_UCS4));
    if (!out) { PyMem_Free(buf); return PyErr_NoMemory(); }

    Py_ssize_t olen = 0;
    int prev_space = 1; /* start true to strip leading */

    for (Py_ssize_t i = 0; i < ucs_len; i++) {
        Py_UCS4 c = buf[i];

        if (is_whitespace(c)) {
            if (!prev_space) {
                out[olen++] = ' ';
                prev_space = 1;
            }
            continue;
        }

        /* Lowercase ASCII */
        if (c >= 'A' && c <= 'Z') {
            c = c + 32;
        }

        out[olen++] = c;
        prev_space = 0;
    }

    /* Strip trailing space */
    if (olen > 0 && out[olen - 1] == ' ') olen--;

    PyObject *result = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, out, olen);

    PyMem_Free(out);
    PyMem_Free(buf);

    return result;
}

/* ------------------------------------------------------------------ */
/* Module definition                                                    */
/* ------------------------------------------------------------------ */

static PyMethodDef TokenizerMethods[] = {
    {"fast_tokenize", py_fast_tokenize, METH_VARARGS,
     "Tokenize text into a list of tokens (fast C implementation)."},
    {"fast_split_sentences", py_fast_split_sentences, METH_VARARGS,
     "Split text into sentences (fast C implementation)."},
    {"fast_normalize", py_fast_normalize, METH_VARARGS,
     "Normalize text: lowercase, collapse whitespace, strip."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef tokenizermodule = {
    PyModuleDef_HEAD_INIT,
    "_tokenizer_fast",
    "Fast C tokenizer for manglish-nlp.",
    -1,
    TokenizerMethods
};

PyMODINIT_FUNC PyInit__tokenizer_fast(void) {
    return PyModule_Create(&tokenizermodule);
}
