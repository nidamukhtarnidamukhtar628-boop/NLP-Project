# Assignment-1

For my Natural Language Processing Assignment-1 NLP Project: Phase 1 (Assignment 1)

**Data Acquisition, Text Processing & Cleaning**
Course: Introduction to NLP

| Team Name | Roll Number | GitHub Username |
|---|---|---|
| Muhammad Abdullah | L1F23BSCS0328 | |
| Muhammad Freed Alam | L1F23BSCS0337 | |
| Nida Mukthar | L1F23BSCS1127 | |

**Team name:** ETERNALS

## Project Overview

<2-3 sentences: what your project is about and why you chose this topic.>

This repository covers Phase 1: loading a public text dataset, exploring it, comparing tokenization methods, and building a reusable cleaning pipeline that produces an analysis-ready corpus for Phase 2.

## Dataset

| Field | Details |
|---|---|
| Name | |
| Source / Link | |
| Size | <number of rows / documents> |
| Language | <English / Urdu / ...> |
| Licence | <e.g. CC BY 4.0> |
| Prediction target | <e.g. sentiment label, category> |

The raw data is **not** committed to this repository. It is downloaded or loaded by `src/load_data.py`.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # raw data (ignored by git)
├── notebooks/
│   └── exploration.ipynb  # Task 2 & 3: statistics, tokenization, regex
├── src/
│   ├── load_data.py       # downloads / loads the raw dataset
│   └── cleaning.py        # reusable cleaning pipeline (Task 4 & 5)
└── output/
    └── cleaned_data.csv   # cleaned dataset used in Phase 2
```

## Setup

Requirements: **Python 3.11+**

```bash
# 1. Clone the repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK / spaCy resources
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm   # change model if needed
```

## How to Run (Reproducing the Output)

Run these steps in order from the project root:

```bash
# Step 1: Download / load the raw data into data/
python src/load_data.py

# Step 2: Run the cleaning pipeline and save the cleaned dataset
python src/cleaning.py

# Step 3 (optional): Open the exploration notebook
jupyter notebook notebooks/exploration.ipynb
```

After running, the cleaned dataset is saved to `output/cleaned_data.csv`.

## Tasks Summary

### Task 1: Dataset & Environment

Dataset documented above. Loading script: `src/load_data.py`. Environment recorded in `requirements.txt`.

### Task 2: Exploration & Basic Statistics

- Number of documents:
- Average document length (words):
- Target distribution:
- Vocabulary size / tokens vs types:
- Zipf / long-tail observations:

### Task 3: Tokenization & Regular Expressions

- Tokenizers compared: <e.g. regex tokenizer vs NLTK / spaCy>
- Regex patterns used:

| Pattern | Purpose |
|---|---|
| <pattern 1> | <e.g. remove URLs> |
| <pattern 2> | <e.g. find hashtags> |
| <pattern 3> | <e.g. match numbers / prices> |

- Tokenization challenges: <e.g. contractions, hyphenation, emojis>

### Task 4: Cleaning & Normalization Pipeline

Steps applied in this order:

1. Unicode normalization (<NFC / script-specific>):
2. Case folding: <yes/no, justification>
3. Noise removal (<HTML, URLs, whitespace, ...>):
4. Stop-word removal: <yes/no, justification; negations preserved?>
5. Stemming or lemmatization (<which one>):

| Vocabulary size | Count |
|---|---|
| Before cleaning | |
| After cleaning | |

### Task 5: Output & Reproducibility

The cleaned dataset is saved by `src/cleaning.py` to `output/cleaned_data.csv`. A fresh clone, install and run reproduces it using the steps above.

## Git Workflow

- Each member works on their own feature branch (`feature/<name>`).
- Changes are merged into `main` through Pull Requests.
- Commit messages are short and descriptive.
- The final submission is tagged `assignment-1`.

## Contribution Statement

| Member | Contribution |
|---|---|
| <Member 1> | |
| <Member 2> | |
| <Member 3> | |