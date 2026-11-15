# NLP Project: Phase 1 and Phase 2

Course: Introduction to NLP

- Phase 1: Data Acquisition, Text Processing and Cleaning (Assignment 1, tag `assignment-1`)
- Phase 2: Word Representations: BoW, TF-IDF, N-grams, Word2Vec (CBOW and Skip-gram), fastText (Assignment 2, tag `assignment-2`)

## Team

Team name: ETERNALS

| Name | Roll Number | GitHub Username |
|---|---|---|
| Muhammad Abdullah | L1F23BSCS0328 | |
| Muhammad Freed Alam | L1F23BSCS0337 | freedalam |
| Nida Mukthar | L1F23BSCS1127 | nidamukhtarnidamukhtar628-boop |
| Muhammad Baar | L1F23BSCS0334 | muhammad-baar |
| Muhammad Mahmood Akhter | L1F23BSCS0138 | |

## Project Overview

We build a sentiment analysis pipeline for movie reviews: given an IMDb review, decide whether the writer liked or disliked the film. We chose this topic because the data is public, large, balanced and realistically messy, and because sentiment depends on small words ("not good"), which makes it a good test for every text representation covered in class.

- Phase 1 loads the dataset, explores it, compares tokenizers and builds a reusable cleaning pipeline that produces an analysis-ready corpus.
- Phase 2 turns that same cleaned corpus into numerical representations (count-based and learned embeddings), probes what each one captures, and packages them for Phase 3.

## Dataset

| Field | Details |
|---|---|
| Name | IMDB Dataset of 50K Movie Reviews |
| Source / Link | https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews (originally Maas et al., 2011) |
| Size | 50,000 reviews (49,582 after removing 418 exact duplicates) |
| Language | English |
| Licence | <check the Kaggle page and write it here> |
| Prediction target | sentiment: positive / negative (25,000 each in the raw data) |

The raw data is not committed to this repository. It is downloaded or loaded by `src/load_data.py`.

### Dataset Description

Our project uses the IMDb Dataset of 50K Movie Reviews, obtained from Kaggle. This dataset contains 50,000 English movie reviews, divided equally into 25,000 positive reviews and 25,000 negative reviews. It consists of two main columns: `review`, which contains the actual text written by users about movies, and `sentiment`, which identifies whether the review expresses a positive or negative opinion. The dataset is used for Natural Language Processing (NLP) and sentiment analysis, where our goal is to process movie reviews, identify meaningful words and patterns, and understand how different text representation techniques work. Throughout this project, we use the dataset for text exploration, tokenization, text cleaning, Bag of Words (BoW), TF-IDF, N-grams, Word2Vec (CBOW and Skip-gram), and fastText embeddings. The same dataset is also used in the later stages of the project for machine-learning and deep-learning experiments.

## Repository Structure

```
nlp-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                           # raw IMDB CSV (ignored by git)
├── models/                         # Phase 2: saved vectorizers and embeddings (about 82 MB)
│   ├── bow_vectorizer.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── tfidf_ngram_vectorizer.joblib
│   ├── word2vec_cbow.kv
│   ├── word2vec_skipgram.kv
│   └── fasttext.kv
├── notebooks/
│   ├── exploration.ipynb           # Phase 1: statistics, tokenizers, regex, cleaning analysis
│   └── representations.ipynb       # Phase 2: Parts A, B, C (and a demo of Part D)
├── src/
│   ├── load_data.py                # Phase 1: downloads / loads the raw dataset
│   ├── cleaning.py                 # Phase 1: reusable cleaning pipeline, writes the cleaned CSV
│   ├── representations.py          # Phase 2 (D1): builds, trains and saves all representations
│   └── vectorize.py                # Phase 2 (D2): vectorize(raw_document, method) interface
└── output/
    ├── cleaned_data.csv            # Phase 1 output, input of Phase 2
    ├── cleaning_stats.json
    ├── phase2_summary.json         # shapes, sparsity, hyperparameters, training times
    ├── phase2_results.json         # key numbers produced by the Phase 2 notebook
    └── figures/                    # plots used in the reports
```

## Setup

Requirements: Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/nidamukhtarnidamukhtar628-boop/NLP-Project.git
cd NLP-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies (includes gensim and scikit-learn for Phase 2)
pip install -r requirements.txt

# 4. Download the NLTK resources
python -m nltk.downloader punkt punkt_tab stopwords wordnet
```

`src/cleaning.py` also downloads `stopwords` and `wordnet` automatically. `punkt_tab` is only needed by the Phase 1 notebook. We do not use spaCy.

## How to Run (Reproducing the Output)

Run these steps in order from the project root:

```bash
# Phase 1
python src/load_data.py            # raw data into data/ (downloads it if missing)
python src/cleaning.py             # -> output/cleaned_data.csv   (about 2 minutes)

# Phase 2
python src/representations.py      # -> models/ and output/phase2_summary.json (about 15 minutes on 1 CPU core)
python src/vectorize.py "This movie was great!"   # quick test of the vectorize interface

# Notebooks (run all cells, from the notebooks/ folder)
jupyter notebook notebooks/exploration.ipynb         # Phase 1
jupyter notebook notebooks/representations.ipynb     # Phase 2
```

`representations.py` only builds what is missing; add `--force` to rebuild everything. If the automatic download fails, download the CSV from the Kaggle link and save it as `data/IMDB_Dataset.csv`.

Reproducibility note: the embeddings are trained with a fixed seed and one worker, so repeated runs should give the same or nearly the same vectors; small differences can occur on other machines or library versions.

## Phase 1 Summary (Assignment 1)

### Task 1: Dataset and Environment

Dataset documented above. Loading script: `src/load_data.py`. Environment recorded in `requirements.txt`.

### Task 2: Exploration and Basic Statistics

| Statistic | Value |
|---|---|
| Number of documents | 50,000 (418 exact duplicates, 0 missing values) |
| Average document length | 1,309 characters / 231.2 words (median 173, max 2,470) |
| Target distribution | 25,000 positive / 25,000 negative (balanced) |
| Tokens / types (light tokenization) | 11,485,517 tokens / 109,526 types (type-token ratio 0.0095) |
| Words seen only once | 42,545 (38.8% of the vocabulary) |
| Coverage | top 100 types = 50.7% of tokens, top 1,000 = 76.4% |

The rank-frequency plot is close to a straight line on log-log axes (Zipf's law): a few function words dominate and a long tail of rare words forms most of the vocabulary. Messy text found: HTML `<br />` in 58.4% of reviews, ALL-CAPS words in 22.7%, non-ASCII characters in 9.3%, x/10 ratings in 6.0%, elongated words in 1.4%, URLs in 0.4%.

### Task 3: Tokenization and Regular Expressions

Tokenizers compared on a 2,000-review sample: whitespace split (459,645 tokens, 49,370 types), regex letter runs (461,553 tokens, 26,337 types) and NLTK `word_tokenize` (533,330 tokens, 28,341 types, about 20 times slower than the regex).

| Pattern | Purpose |
|---|---|
| `<[^>]+>` | remove HTML tags such as `<br />` |
| `https?://\S+\|www\.\S+` | remove URLs |
| `\b\d{1,2}(?:\.\d)?\s*/\s*10\b` | remove numeric ratings like 8/10 |
| `([a-z])\1{2,}` | collapse elongated letters (sooooo to soo) |
| `[a-z]+` | tokenizer: keep alphabetic runs |

Tokenization challenges: contractions and negation (can't), punctuation glued to words in whitespace splitting, and special strings (ratings, URLs, prices, hyphenated words).

### Task 4: Cleaning and Normalization Pipeline

Steps applied in this order:

1. Unicode normalization (NFKD, strip accents, unify curly quotes)
2. Noise removal (HTML, URLs, ratings), before tokenizing so `<br />` does not leave a `br` token
3. Case folding and collapsing elongated letters
4. Contraction expansion (`don't` to `do not`), before stop-word removal so negations survive
5. Tokenization (letters only, length 2 or more)
6. Stop-word removal with the NLTK list minus `no`, `nor`, `not` (negations flip sentiment)
7. Lemmatization (WordNet, noun then verb), chosen over stemming because it returns real words

| Vocabulary size | Count |
|---|---|
| Before cleaning (raw whitespace tokens) | 438,729 |
| After cleaning | 79,320 (-81.9%) |

Tokens drop from 11.47M to 6.01M (about 121 per review). Exact duplicates are removed so they cannot leak between train and test later.

Known limitations: x/10 ratings and digits are removed (a strong sentiment signal is lost); the noun-then-verb lemmatizer turns -ing adjectives such as "boring" into the verb "bore".

### Task 5: Output and Reproducibility

`python src/cleaning.py` writes `output/cleaned_data.csv` (49,582 rows; columns `orig_id`, `sentiment`, `label`, `clean_text`, `n_tokens`; 24,884 positive / 24,698 negative) and `output/cleaning_stats.json`. A fresh clone, install and run reproduces it.

## Phase 2 Summary (Assignment 2): Word Representations

All representations use the same cleaned corpus from Phase 1 (49,582 reviews, 6,007,695 tokens), unchanged. The details, code and outputs are in `notebooks/representations.ipynb`.

### Part A: Count-based representations

| Representation | Matrix shape | Sparsity (% zeros) |
|---|---|---|
| A1. Bag of Words (unigrams) | 49,582 x 79,320 | 99.8815 % (4,660,849 non-zeros; 37.5 MB sparse vs 15.7 GB dense) |
| A2. TF-IDF (unigrams) | 49,582 x 79,320 | 99.8815 % |
| A3. TF-IDF unigrams + bigrams, df >= 5 (saved for Phase 3) | 49,582 x 182,415 | 99.9192 % |

A2, high and low TF-IDF weights (example review, `orig_id` 5, every word occurs once so only idf differs): high weights go to rare specific words, "selflessness" (document frequency 4, idf 10.20, weight 0.293), "lukas" (42, 8.05, 0.231), "midget" (70, 7.55, 0.217). Common words are pushed down: "not" (in 40,457 reviews, idf 1.20, weight 0.035), "one" (0.044), "make" (0.048). A word that appears in almost every review cannot help tell reviews apart, so idf lowers its weight.

A3, n-gram growth:

| n-gram range | Features | Growth | Sparsity | Features seen once | Features with df >= 5 |
|---|---|---|---|---|---|
| (1,1) | 79,320 | 1x | 99.881 % | 40.6 % | 27,136 |
| (1,2) | 2,681,743 | 33.8x | 99.992 % | 75.5 % | 182,415 |
| (1,3) | 7,973,854 | 100.5x | 99.996 % | 88.4 % | 213,757 |

The vocabulary explodes with n and the matrix becomes sparser, because most bigrams and trigrams occur only once. They do capture useful phrases ("not waste time", "make no sense", "worst movie ever"), and pruning rare n-grams (df >= 5) removes most of the cost.

A4, hand-worked numerical (4 short real reviews, N = 4): for the word "primary" in `d4` (tf = 2, df = 1), idf = ln((1+4)/(1+1)) + 1 = 1.9163, so tf-idf = 2 x 1.9163 = 3.8326, and after L2 normalisation (norm 5.4201) it is 0.7071. The bigram "primary plot" has tf = 1, df = 1 and weight 1.9163. All values match the scikit-learn output (checked with `assert` in the notebook). The same arithmetic must be written by hand, scanned (CamScanner) and attached to the PDF report.

### Part B: Learned embeddings

Shared hyperparameters (CBOW and Skip-gram are identical except for `sg`):

| Model | sg | vector_size | window | min_count | epochs | negative | Vocabulary | Training time |
|---|---|---|---|---|---|---|---|---|
| B1. Word2Vec CBOW | 0 | 100 | 5 | 5 | 10 | 5 | 29,485 | 78 s |
| B2. Word2Vec Skip-gram | 1 | 100 | 5 | 5 | 10 | 5 | 29,485 | 280 s |
| B4. fastText (Skip-gram + character n-grams 3 to 6, 100,000 buckets) | 1 | 100 | 5 | 5 | 10 | 5 | 29,485 | 482 s |

B3, CBOW vs Skip-gram: CBOW goes from context to centre word. It averages the surrounding word vectors and predicts the middle word (one training example per window, fast, smooth). Skip-gram goes from centre word to context. It predicts each surrounding word separately (up to 2 x window examples per window), which gives rare words more updates but is slower (3.6 times in our run). We expected Skip-gram to be better for the rare words of our corpus; the results only partly confirm this (see Part C).

B4, fastText and sub-words: Word2Vec gives each whole word one independent vector and has no vector at all for an unseen word. fastText sums the vectors of a word's character n-grams, so spelling-similar words get similar vectors and a misspelled or unseen word still gets a vector. Demonstration (neighbours): `fantastik` is not in the Word2Vec vocabulary (`KeyError` in CBOW and Skip-gram), while fastText returns "fantastic" (0.85); `terriblee` gives "terrible" (0.90); the rare word `performace` (5 occurrences) gives random actor names in Word2Vec but "performance" (0.80) and "performer" (0.80) in fastText. fastText is not perfect: `aweful` returns "rueful" and "tuneful" instead of "awful".

### Part C: Probing and comparison

- C1, neighbours: "good" gives great, decent, nice; "bad" gives awful, terrible, horrible; "actor" gives actress (0.73), performer, cast; "funny" gives hilarious (0.80 to 0.84); "horror" gives slasher (0.78 to 0.81), genre. The top neighbours make sense in all models; Skip-gram and fastText add rare names and titles further down the list.
- C2, similarities (CBOW / Skip-gram / fastText): movie-film 0.87 / 0.88 / 0.87, good-great 0.76 / 0.77 / 0.76 (similar); actor-refrigerator -0.22 / 0.12 / 0.25, director-banana -0.13 / 0.17 / 0.15 (unrelated). good-bad is 0.71 / 0.75 / 0.74: embeddings capture shared context, not polarity.
- C3, analogies: actor - man + woman gives "actress" in all models (but "actress" is already the nearest neighbour of "actor"); comedy - funny + scary gives "horror"; terrible - bad + good gives "decent" / "great"; director - actor + actress fails (names and spelling-similar words), reported honestly because the question is ill-posed for this corpus.
- C4, count-based vs learned: count-based vectors are very wide (79,320 to 182,415 dimensions), 99.9 % zeros, interpretable, and give an all-zero vector for unseen words. Embeddings have 100 dense dimensions, capture word meaning and similarity, but lose word order when averaged and need enough training text; only fastText handles unseen words. A quick sanity check (10,000 reviews, untuned logistic regression) gave accuracy BoW 0.837, TF-IDF 0.843, TF-IDF with bigrams 0.847, CBOW 0.850, Skip-gram 0.860, fastText 0.860. The gaps are small (single split, standard error about 0.8 points), so we claim no winner; the real comparison belongs to Phase 3. Our corpus is small for embeddings (6.0M tokens), so rare-word results are interpreted cautiously.

### Part D: Package for the next phase

D1, saved models (`python src/representations.py`, total about 82 MB, every file under 100 MB so GitHub accepts them):

| File | Content | Size |
|---|---|---|
| `models/bow_vectorizer.joblib` | Bag of Words, unigrams, 79,320 features | 0.7 MB |
| `models/tfidf_vectorizer.joblib` | TF-IDF, unigrams, 79,320 features | 0.8 MB |
| `models/tfidf_ngram_vectorizer.joblib` | TF-IDF, unigrams + bigrams, df >= 5, 182,415 features | 2.0 MB |
| `models/word2vec_cbow.kv` | Word2Vec CBOW vectors | 12.7 MB |
| `models/word2vec_skipgram.kv` | Word2Vec Skip-gram vectors | 12.7 MB |
| `models/fasttext.kv` | fastText vectors including character n-grams | 52.7 MB |

Embeddings are stored as gensim `KeyedVectors` (vectors only, which is all Phase 3 needs). To retrain, run the script with `--force`. Models are never trained when loading.

D2, the vectorize interface (`src/vectorize.py`): raw document in, vector out. The document is first cleaned with the Phase 1 pipeline, then vectorized.

```python
import sys; sys.path.insert(0, "src")
from vectorize import vectorize, vectorize_many, available_methods

available_methods()   # ('bow', 'tfidf', 'tfidf_ngram', 'cbow', 'skipgram', 'fasttext')

v = vectorize("This movie wasn't good at all!!!", method="tfidf")    # sparse (1, 79320)
v = vectorize("This movie wasn't good at all!!!", method="fasttext") # dense (100,)
X = vectorize_many(list_of_raw_reviews, method="skipgram")            # dense (n_docs, 100)
```

Count-based methods return a scipy sparse matrix (a dense array with `dense=True`). Embedding methods return the mean of the document's word vectors (a zero vector if no word can be embedded; fastText always produces a vector). Pass `clean=False` only for text that is already the output of `cleaning.clean_text()`.

## Git Workflow

- Each member works on their own feature branch (`feature/<name>`).
- Changes are merged into `main` through Pull Requests.
- Commit messages are short and descriptive.
- Phase 1 submission is tagged `assignment-1`; Phase 2 submission is tagged `assignment-2` (in the same repository).

## Contribution Statement

### Phase 1

| Member | Roll Number | Contribution |
|---|---|---|
| Muhammad Abdullah | L1F23BSCS0328 | Tokenization, cleaning |
| Muhammad Freed Alam | L1F23BSCS0337 | Sourcing of data, cleaning and load data |
| Nida Mukthar | L1F23BSCS1127 | Environment setup of Git and README updates |
| Muhammad Baar | L1F23BSCS0334 | Dataset setup and finding |
| Muhammad Mahmood Akhter | L1F23BSCS0138 | Output setup and testing |

### Phase 2

| Member | Roll Number | Contribution |
|---|---|---|
| Muhammad Abdullah | L1F23BSCS0328 | Part A: BoW, TF-IDF, N-grams, hand-worked numerical |
| Muhammad Freed Alam | L1F23BSCS0337 | Part B: Word2Vec CBOW, Skip-gram, fastText (shared with Nida) |
| Nida Mukthar | L1F23BSCS1127 | Part B: Word2Vec CBOW, Skip-gram, fastText (shared with Freed) |
| Muhammad Baar | L1F23BSCS0334 | Part C: probing and comparison |
| Muhammad Mahmood Akhter | L1F23BSCS0138 | Part D: saved models, vectorize interface |

## Submission

- Repository: https://github.com/nidamukhtarnidamukhtar628-boop/NLP-Project
- Phase 1 tag: https://github.com/nidamukhtarnidamukhtar628-boop/NLP-Project/releases/tag/assignment-1
- Phase 2 tag: https://github.com/nidamukhtarnidamukhtar628-boop/NLP-Project/releases/tag/assignment-2
