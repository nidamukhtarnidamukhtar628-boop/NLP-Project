"""Phase 2: word representations (count-based and learned).

Usage (from the project root, after Phase 1 produced output/cleaned_data.csv):

    python src/representations.py            # build + save anything that is missing
    python src/representations.py --force    # rebuild everything from scratch

What it builds (all from the SAME cleaned corpus of Phase 1):

    Count-based  (fitted with scikit-learn, saved with joblib)
        bow_vectorizer.joblib          Bag of Words, unigrams, full vocabulary
        tfidf_vectorizer.joblib        TF-IDF, unigrams, full vocabulary
        tfidf_ngram_vectorizer.joblib  TF-IDF, unigrams + bigrams, min_df=5

    Learned embeddings (trained with gensim, saved as KeyedVectors)
        word2vec_cbow.kv               Word2Vec, CBOW      (sg=0)
        word2vec_skipgram.kv           Word2Vec, Skip-gram (sg=1)
        fasttext.kv                    fastText (Skip-gram + character n-grams)

Models are trained ONCE here and only loaded afterwards (see src/vectorize.py).
Training is never triggered from the loading path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText, KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.fasttext import FastTextKeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import ROOT  # noqa: E402

CLEAN_PATH = ROOT / "output" / "cleaned_data.csv"
MODELS_DIR = ROOT / "models"
SUMMARY_PATH = ROOT / "output" / "phase2_summary.json"

# --------------------------------------------------------------------------
# Where everything is saved
# --------------------------------------------------------------------------
COUNT_PATHS = {
    "bow": MODELS_DIR / "bow_vectorizer.joblib",
    "tfidf": MODELS_DIR / "tfidf_vectorizer.joblib",
    "tfidf_ngram": MODELS_DIR / "tfidf_ngram_vectorizer.joblib",
}
EMBED_PATHS = {
    "cbow": MODELS_DIR / "word2vec_cbow.kv",
    "skipgram": MODELS_DIR / "word2vec_skipgram.kv",
    "fasttext": MODELS_DIR / "fasttext.kv",
}

# --------------------------------------------------------------------------
# Hyperparameters.  CBOW and Skip-gram share EXACTLY the same values, only `sg`
# differs.  fastText uses the same shared values plus its sub-word settings.
# --------------------------------------------------------------------------
SHARED_PARAMS = dict(
    vector_size=100,   # embedding dimension
    window=5,          # context words on each side of the centre word
    min_count=5,       # ignore words that occur fewer than 5 times
    epochs=10,         # passes over the corpus
    negative=5,        # negative sampling: 5 noise words per positive pair
    sample=1e-3,       # down-sample very frequent words
    workers=1,         # 1 worker + fixed seed = (almost) reproducible runs
    seed=42,
)
FASTTEXT_EXTRA = dict(
    min_n=3,           # shortest character n-gram
    max_n=6,           # longest character n-gram
    bucket=100_000,    # hash buckets for n-grams (default 2M is too big for git)
)

# The cleaned text is already lower-case words separated by single spaces, so
# the vectorizers must split on whitespace only and must not lower-case again.
TOKEN_PATTERN = r"\S+"


# --------------------------------------------------------------------------
# Count-based vectorizers
# --------------------------------------------------------------------------
def make_bow() -> CountVectorizer:
    """Bag of Words: raw counts, unigrams, whole vocabulary."""
    return CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False,
                           dtype=np.int32)


def make_tfidf() -> TfidfVectorizer:
    """TF-IDF over unigrams (scikit-learn default: smooth idf + L2 norm)."""
    return TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False,
                           dtype=np.float32)


def make_tfidf_ngram() -> TfidfVectorizer:
    """TF-IDF over unigrams + bigrams; min_df=5 keeps the feature count sane."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                           token_pattern=TOKEN_PATTERN, lowercase=False,
                           dtype=np.float32)


COUNT_FACTORIES = {"bow": make_bow, "tfidf": make_tfidf,
                   "tfidf_ngram": make_tfidf_ngram}


# --------------------------------------------------------------------------
# Corpus helpers
# --------------------------------------------------------------------------
def load_corpus() -> pd.DataFrame:
    """Load the Phase 1 output (columns: orig_id, sentiment, label, clean_text, n_tokens)."""
    if not CLEAN_PATH.exists():
        sys.exit(f"{CLEAN_PATH} not found. Run `python src/cleaning.py` first (Phase 1).")
    return pd.read_csv(CLEAN_PATH)


class SentenceStream:
    """Streams each review as a list of tokens, so the corpus is never copied in memory."""

    def __init__(self, texts):
        self.texts = list(texts)

    def __iter__(self):
        for text in self.texts:
            yield text.split()

    def __len__(self):
        return len(self.texts)


class EpochLogger(CallbackAny2Vec):
    """Prints one line per finished training epoch."""

    def __init__(self, name: str, total: int):
        self.name, self.total, self.epoch, self.t0 = name, total, 0, time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"  [{self.name}] epoch {self.epoch}/{self.total} "
              f"({time.time() - self.t0:.0f}s elapsed)", flush=True)


# --------------------------------------------------------------------------
# Training the embedding models
# --------------------------------------------------------------------------
def train_word2vec(sentences, sg: int) -> Word2Vec:
    name = "skipgram" if sg else "cbow"
    return Word2Vec(sentences, sg=sg, **SHARED_PARAMS,
                    callbacks=[EpochLogger(name, SHARED_PARAMS["epochs"])])


def train_fasttext(sentences) -> FastText:
    # sg=1 so the ONLY difference to the Skip-gram Word2Vec model is the
    # character n-gram (sub-word) information.
    return FastText(sentences, sg=1, **SHARED_PARAMS, **FASTTEXT_EXTRA,
                    callbacks=[EpochLogger("fasttext", SHARED_PARAMS["epochs"])])


# --------------------------------------------------------------------------
# Loading (never trains)
# --------------------------------------------------------------------------
def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Create it once with:  python src/representations.py")
    return path


def load_count_vectorizer(name: str):
    return joblib.load(_require(COUNT_PATHS[name]))


def load_embeddings(name: str) -> KeyedVectors:
    path = _require(EMBED_PATHS[name])
    if name == "fasttext":
        return FastTextKeyedVectors.load(str(path))
    return KeyedVectors.load(str(path))


# --------------------------------------------------------------------------
# Summary file (so notebook, README and report quote identical numbers)
# --------------------------------------------------------------------------
def _read_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}


def _write_summary(summary: dict) -> None:
    SUMMARY_PATH.parent.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def _mb(path: Path) -> float:
    files = [path] if path.is_file() else []
    # gensim stores big arrays in side files next to the main file (name.kv.vectors*.npy)
    files += [p for p in path.parent.glob(path.name + ".*")]
    return round(sum(p.stat().st_size for p in files) / 1e6, 1)


# --------------------------------------------------------------------------
# Main: build and save everything
# --------------------------------------------------------------------------
def build_count_based(texts, force: bool, summary: dict) -> None:
    for name, factory in COUNT_FACTORIES.items():
        path = COUNT_PATHS[name]
        if path.exists() and not force:
            print(f"[skip] {path.name} already exists")
            continue
        print(f"[fit]  {name} ...", flush=True)
        t0 = time.time()
        vec = factory()
        X = vec.fit_transform(texts)
        secs = time.time() - t0
        # stop_words_ lists every pruned term (millions with n-grams) and is not needed
        if hasattr(vec, "stop_words_"):
            vec.stop_words_ = None
        joblib.dump(vec, path, compress=3)
        n_docs, n_feat = X.shape
        summary[name] = {
            "shape": [int(n_docs), int(n_feat)],
            "nonzero": int(X.nnz),
            "sparsity_pct": round(100 * (1 - X.nnz / (n_docs * n_feat)), 4),
            "fit_seconds": round(secs, 1),
            "file_mb": _mb(path),
        }
        print(f"       shape={X.shape}, sparsity={summary[name]['sparsity_pct']}%, "
              f"saved {path.name} ({summary[name]['file_mb']} MB)")
        del X
        _write_summary(summary)


def build_embeddings(texts, force: bool, summary: dict) -> None:
    todo = [n for n, p in EMBED_PATHS.items() if force or not p.exists()]
    for name in EMBED_PATHS:
        if name not in todo:
            print(f"[skip] {EMBED_PATHS[name].name} already exists")
    if not todo:
        return
    sentences = SentenceStream(texts)
    for name in todo:
        print(f"[train] {name} ...", flush=True)
        t0 = time.time()
        if name == "fasttext":
            model = train_fasttext(sentences)
        else:
            model = train_word2vec(sentences, sg=1 if name == "skipgram" else 0)
        secs = time.time() - t0
        # Only the vectors are kept (smaller files, enough for Phase 3).
        model.wv.save(str(EMBED_PATHS[name]))
        params = dict(SHARED_PARAMS, sg=1 if name in ("skipgram", "fasttext") else 0)
        if name == "fasttext":
            params.update(FASTTEXT_EXTRA)
        summary[name] = {
            "vocab_size": len(model.wv),
            "vector_size": SHARED_PARAMS["vector_size"],
            "train_seconds": round(secs, 1),
            "file_mb": _mb(EMBED_PATHS[name]),
            "params": params,
            "corpus_tokens": int(model.corpus_total_words),
        }
        print(f"        vocab={len(model.wv):,}, {secs:.0f}s, "
              f"saved {EMBED_PATHS[name].name} ({summary[name]['file_mb']} MB)")
        del model
        _write_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save all Phase 2 representations.")
    parser.add_argument("--force", action="store_true", help="rebuild even if files exist")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    df = load_corpus()
    texts = df["clean_text"].tolist()
    print(f"Corpus: {len(texts):,} cleaned reviews")

    summary = _read_summary()
    summary["corpus_documents"] = len(texts)
    build_count_based(texts, args.force, summary)
    build_embeddings(texts, args.force, summary)
    _write_summary(summary)
    print(f"\nDone. Models are in {MODELS_DIR}, summary in {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
