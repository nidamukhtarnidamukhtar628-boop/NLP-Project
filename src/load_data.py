"""Load the raw IMDB 50K movie-review dataset.

Usage (from the project root):
    python src/load_data.py

The raw CSV is NOT committed to git. If it is missing from data/, this script
tries to download it from Kaggle with `kagglehub`. If that is not possible,
it prints manual download instructions.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "IMDB_Dataset.csv"
KAGGLE_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"


def download_raw() -> Path:
    """Download the dataset from Kaggle into data/ (needs `kagglehub`)."""
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub is not installed and data/IMDB_Dataset.csv is missing.\n"
            f"Either `pip install kagglehub` or download the CSV manually from\n"
            f"{KAGGLE_URL}\nand place it at {RAW_PATH}"
        )
    folder = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    DATA_DIR.mkdir(exist_ok=True)
    shutil.copy(folder / "IMDB Dataset.csv", RAW_PATH)
    return RAW_PATH


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Return the raw dataframe with columns: review, sentiment."""
    path = Path(path)
    if not path.exists():
        print(f"{path} not found, trying to download ...")
        download_raw()
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} rows from {RAW_PATH}")
    print(df["sentiment"].value_counts().to_string())
    print(df.head(3))
