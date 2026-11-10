"""Reusable text-cleaning pipeline for the IMDB reviews (Phase 1).

Usage (from the project root):
    python src/cleaning.py

Pipeline order (each step is a separate function so the notebook can measure
the effect of every step):

    1. Unicode normalization   normalize_unicode()
    2. Noise removal           strip_noise()          HTML, URLs, ratings, repeated letters
    3. Case folding            (str.lower inside strip_noise order, see below)
    4. Contraction expansion   expand_contractions()  keeps negations ("don't" -> "do not")
    5. Tokenization            tokenize()
    6. Stop-word removal       remove_stopwords()     negations are kept
    7. Lemmatization           lemmatize_tokens()     (we use lemmatization, NOT stemming)
"""
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import ROOT, load_raw  # noqa: E402

OUTPUT_DIR = ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "cleaned_data.csv"
STATS_PATH = OUTPUT_DIR / "cleaning_stats.json"

# --------------------------------------------------------------------------
# Regular expressions (also demonstrated in notebooks/exploration.ipynb)
# --------------------------------------------------------------------------
HTML_RE = re.compile(r"<[^>]+>")                              # <br />, <i>, ...
URL_RE = re.compile(r"https?://\S+|www\.\S+")                 # links
RATING_RE = re.compile(r"\b\d{1,2}(?:\.\d)?\s*/\s*10\b")      # "8/10", "7.5 / 10"
REPEAT_RE = re.compile(r"([a-z])\1{2,}")                      # "sooooo" -> "soo"
TOKEN_RE = re.compile(r"[a-z]+")                              # letters only

# Contractions: order matters (irregular forms first).
CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]

# Negations carry sentiment ("not good"), so we keep them.
NEGATIONS = {"no", "nor", "not"}


def _ensure_nltk() -> None:
    for pkg, path in [("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS
_lemmatizer = WordNetLemmatizer()


# --------------------------------------------------------------------------
# Individual steps
# --------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """NFKD-normalize, drop accents (cafe), unify curly quotes."""
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def strip_noise(text: str) -> str:
    """Remove HTML, URLs and x/10 ratings, case-fold, squash repeated letters.

    HTML is removed BEFORE lower-casing/tokenizing, otherwise '<br />' would
    leave a stray 'br' token in the vocabulary.
    """
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = RATING_RE.sub(" ", text)
    text = text.lower()                       # case folding
    text = REPEAT_RE.sub(r"\1\1", text)       # sooooo -> soo
    return text


def expand_contractions(text: str) -> str:
    for pattern, repl in CONTRACTIONS:
        text = pattern.sub(repl, text)
    return text


def tokenize(text: str) -> list:
    """Letters-only tokens of length >= 2 (drops digits, punctuation, 's' leftovers)."""
    return [t for t in TOKEN_RE.findall(text) if len(t) > 1]


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS]


@lru_cache(maxsize=None)
def _lemma(word: str) -> str:
    # noun first, then verb: movies -> movie, liked -> like
    return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, "n"), "v")


def lemmatize_tokens(tokens: list) -> list:
    return [_lemma(t) for t in tokens]


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def preprocess(text: str) -> list:
    """Raw review -> list of clean tokens (the reusable pipeline function)."""
    text = normalize_unicode(text)
    text = strip_noise(text)
    text = expand_contractions(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)


def clean_text(text: str) -> str:
    """Same as preprocess() but returns a single space-joined string."""
    return " ".join(preprocess(text))


def main() -> None:
    df = load_raw()
    n_raw = len(df)

    # Exact duplicate reviews would leak between train/test later, so drop them.
    df = df.drop_duplicates(subset="review").reset_index(names="orig_id")
    n_dedup = len(df)

    vocab_raw = len({w for t in df["review"] for w in t.split()})

    tokens = [preprocess(t) for t in df["review"]]
    vocab_clean = len({w for toks in tokens for w in toks})

    out = pd.DataFrame({
        "orig_id": df["orig_id"],
        "sentiment": df["sentiment"],
        "label": (df["sentiment"] == "positive").astype(int),
        "clean_text": [" ".join(t) for t in tokens],
        "n_tokens": [len(t) for t in tokens],
    })
    empty = int((out["n_tokens"] == 0).sum())
    out = out[out["n_tokens"] > 0]

    OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    stats = {
        "rows_raw": n_raw,
        "duplicates_removed": n_raw - n_dedup,
        "rows_empty_after_cleaning_removed": empty,
        "rows_final": len(out),
        "vocab_before_whitespace_tokens": vocab_raw,
        "vocab_after_cleaning": vocab_clean,
        "vocab_reduction_pct": round(100 * (1 - vocab_clean / vocab_raw), 1),
        "avg_tokens_after_cleaning": round(out["n_tokens"].mean(), 1),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Saved cleaned dataset -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
