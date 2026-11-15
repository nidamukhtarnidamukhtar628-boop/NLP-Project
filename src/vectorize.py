"""Phase 2, Part D2: the interface Phase 3 will call.

    from vectorize import vectorize, vectorize_many

    v = vectorize("This movie wasn't good at all!!!", method="tfidf")
    X = vectorize_many(list_of_raw_reviews, method="fasttext")

A raw document is first passed through the SAME Phase 1 cleaning pipeline
(`cleaning.clean_text`) and is then turned into a vector. Models are only
LOADED from models/ (created once by `python src/representations.py`);
nothing is ever retrained here.

Supported methods
-----------------
    "bow"          Bag of Words counts                  sparse, 79,320 dims
    "tfidf"        TF-IDF, unigrams                     sparse, 79,320 dims
    "tfidf_ngram"  TF-IDF, unigrams + bigrams (min_df=5) sparse, 182,415 dims
    "cbow"         mean Word2Vec (CBOW) vector           dense, 100 dims
    "skipgram"     mean Word2Vec (Skip-gram) vector      dense, 100 dims
    "fasttext"     mean fastText vector                  dense, 100 dims

Return types
------------
    count-based  -> scipy.sparse CSR matrix of shape (1, n_features)
                    (or a dense 1-D numpy array if dense=True)
    embeddings   -> 1-D numpy float32 array of length 100

A document embedding is the MEAN of the vectors of its words. Words that are
not in the Word2Vec vocabulary are skipped; fastText never skips a word, it
builds a vector for it from its character n-grams. If no word of a document
can be embedded the result is a zero vector.

Command line demo:
    python src/vectorize.py "This movie wasn't good at all!!!"
"""
import sys
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cleaning import clean_text  # noqa: E402  (Phase 1 pipeline)
from representations import (COUNT_PATHS, EMBED_PATHS, SHARED_PARAMS,  # noqa: E402
                             load_count_vectorizer, load_embeddings)

COUNT_METHODS = tuple(COUNT_PATHS)       # bow, tfidf, tfidf_ngram
EMBEDDING_METHODS = tuple(EMBED_PATHS)   # cbow, skipgram, fasttext
METHODS = COUNT_METHODS + EMBEDDING_METHODS

_cache: dict = {}   # models are loaded once per process


def available_methods() -> tuple:
    """Names accepted by `method=`."""
    return METHODS


def _get(method: str):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose one of: {', '.join(METHODS)}")
    if method not in _cache:
        _cache[method] = (load_count_vectorizer(method) if method in COUNT_METHODS
                          else load_embeddings(method))
    return _cache[method]


def _mean_vector(tokens: list, kv) -> np.ndarray:
    dim = SHARED_PARAMS["vector_size"]
    if method_is_fasttext(kv):
        vectors = [kv[t] for t in tokens]                     # every word gets a vector
    else:
        vectors = [kv[t] for t in tokens if t in kv.key_to_index]  # skip OOV words
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def method_is_fasttext(kv) -> bool:
    return hasattr(kv, "buckets_word")   # only FastTextKeyedVectors has sub-word buckets


def vectorize_many(docs: Iterable[str], method: str = "tfidf", clean: bool = True,
                   dense: bool = False) -> Union[sp.csr_matrix, np.ndarray]:
    """Vectorize many documents at once.

    docs   : raw review texts (or already cleaned texts if clean=False)
    method : one of available_methods()
    clean  : apply the Phase 1 cleaning first (default). Pass False only for text
             that is ALREADY the output of cleaning.clean_text().
    dense  : count-based methods only, return a dense array instead of a sparse matrix
    """
    model = _get(method)
    texts = [clean_text(d) if clean else d for d in docs]
    if method in COUNT_METHODS:
        X = model.transform(texts).tocsr()
        return X.toarray() if dense else X
    return np.vstack([_mean_vector(t.split(), model) for t in texts]) if texts \
        else np.zeros((0, SHARED_PARAMS["vector_size"]), dtype=np.float32)


def vectorize(doc: str, method: str = "tfidf", clean: bool = True,
              dense: bool = False) -> Union[sp.csr_matrix, np.ndarray]:
    """Vectorize ONE raw document. See the module docstring for return types."""
    X = vectorize_many([doc], method=method, clean=clean, dense=dense)
    if method in COUNT_METHODS:
        return X.ravel() if dense else X
    return X[0]


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "This movie wasn't good at all!!! <br />Terrible acting, 2/10."
    print("Raw     :", text)
    print("Cleaned :", clean_text(text))
    for m in METHODS:
        try:
            v = vectorize(text, m)
        except FileNotFoundError as err:
            print(f"{m:12s} unavailable: {err}")
            continue
        if m in COUNT_METHODS:
            print(f"{m:12s} sparse {v.shape}, {v.nnz} non-zero entries")
        else:
            print(f"{m:12s} dense  {v.shape}, first 5 values {np.round(v[:5], 3)}")
