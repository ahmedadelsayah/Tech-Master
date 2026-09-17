"""
Stage 1 — Data (Load, Inspect, Clean, Split)
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_data():
    """Load the raw spam dataset from data/spam.csv."""
    base_dir = os.path.dirname(os.path.abspath(__file__))  # project root (this file's folder)
    csv_path = os.path.join(base_dir, "data", "spam.csv")

    df = pd.read_csv(csv_path, encoding="latin1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    return df


def inspect_data(df):
    """Print basic dataset info."""
    print("First 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def clean_data(df):
    """Drop missing/duplicate rows and clean message text."""
    before_na = df.shape[0]
    df = df.dropna(subset=["message", "label"])
    print(f"\nDropped {before_na - df.shape[0]} rows with missing values")

    before_dupes = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped {before_dupes - df.shape[0]} duplicate rows")

    df["message"] = df["message"].apply(clean_text)

    print("\nDataset shape after cleaning:")
    print(df.shape)

    print("\nLabel distribution after cleaning:")
    print(df["label"].value_counts())

    return df


def split_data(df):
    """Separate features/labels and split into train/test sets."""
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\nTraining set size:", len(X_train))
    print("Testing set size:", len(X_test))

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def run():
    """Run the full Stage 1 pipeline and return train/test splits."""
    df = load_data()
    inspect_data(df)
    df = clean_data(df)
    return split_data(df)
