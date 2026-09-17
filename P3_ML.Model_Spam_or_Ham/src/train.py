"""
Stage 3 — Model Training & Tuning
"""

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def train_models(X_train_vec, y_train):
    """Train the base models (Naive Bayes, Logistic Regression, SVM)."""
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)

    svm_model = SVC()
    svm_model.fit(X_train_vec, y_train)

    return nb_model, lr_model, svm_model


def tune_naive_bayes(X_train_vec, y_train):
    """Use GridSearchCV to tune MultinomialNB's alpha parameter."""
    print("\n--- Tuning MultinomialNB ---")
    params = {"alpha": [0.1, 0.5, 1.0]}
    search = GridSearchCV(MultinomialNB(), params, cv=5)
    search.fit(X_train_vec, y_train)
    return search.best_estimator_


def build_models_dict(best_nb_model, lr_model, svm_model):
    """Combine trained models into a single dictionary for evaluation."""
    return {
        "NB (tuned)": best_nb_model,
        "Logistic Regression": lr_model,
        "SVM": svm_model,
    }


def run(X_train_vec, y_train):
    """Run the full Stage 3 pipeline and return the models dictionary."""
    nb_model, lr_model, svm_model = train_models(X_train_vec, y_train)
    best_nb_model = tune_naive_bayes(X_train_vec, y_train)
    return build_models_dict(best_nb_model, lr_model, svm_model)
