"""
Stage 2 — Feature Extraction (TF-IDF)
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(X_train, X_test):
    """Fit a TF-IDF vectorizer on training data and transform train/test."""
    vectorizer = TfidfVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return vectorizer, X_train_vec, X_test_vec
