
import os
import pandas as pd
import numpy as np 
import re                #for text cleaning
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42 

# ============================================================
# STAGE 1 — Data (Load, Inspect, Clean, Split)
# ============================================================

# Get the absolute path to the directory where this script (main.py) lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Join that base directory with "data" and "spam.csv"
csv_path = os.path.join(BASE_DIR, "data", "spam.csv")

# Load dataset using the dynamic path
df = pd.read_csv(csv_path, encoding="latin1")

# Keep only the required columns
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]


# Inspect dataset
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


# Handle missing values
before_na = df.shape[0]
df = df.dropna(subset=["message", "label"])

print(f"\nDropped {before_na - df.shape[0]} rows with missing values")


# Handle duplicate rows
before_dupes = df.shape[0]
df = df.drop_duplicates()

print(f"Dropped {before_dupes - df.shape[0]} duplicate rows")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text

df["message"] = df["message"].apply(clean_text)


# Check dataset after cleaning
print("\nDataset shape after cleaning:")
print(df.shape)

print("\nLabel distribution after cleaning:")
print(df["label"].value_counts())


# Separate features and labels
X = df["message"]
y = df["label"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())


# ============================================================
# STAGE 2 — Feature Extraction
# ============================================================

# TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# ============================================================
# STAGE 3 — Model Training & Tuning
# ============================================================
#  train and tune MultinomialNB, LogisticRegression, and SVM models
MultinominalNB_model = MultinomialNB()
MultinominalNB_model.fit(X_train_vec, y_train)

LogisticRegression_model = LogisticRegression(max_iter=1000)
LogisticRegression_model.fit(X_train_vec, y_train)

SVM_model = SVC(probability=True)
SVM_model.fit(X_train_vec, y_train)

# 2. Use GridSearchCV for tuning MultinomialNB on X_train_vec
print("\n--- Tuning MultinomialNB ---")
params = {'alpha': [0.1, 0.5, 1.0]}
search = GridSearchCV(MultinomialNB(), params, cv=5)
search.fit(X_train_vec, y_train)
best_nb_model = search.best_estimator_

#   models dictionary to hold all trained models
models = {
    "NB": MultinominalNB_model,
    "Logistic Regression": LogisticRegression_model,
    "SVM": SVM_model,
}

# ============================================================
# STAGE 4 — Evaluation, Comparison & Final Validation
# ============================================================

results = {}

# 1. Evaluate each model
for name, model in models.items():
    # Make predictions on the unseen test set
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics (setting pos_label='spam' since labels are strings)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        "recall": recall_score(y_test, y_pred, pos_label="spam"),
        "f1": f1_score(y_test, y_pred, pos_label="spam")
    }

# 2. Print a comparison table
print("\n--- Model Comparison ---")
print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 75)
for name, metrics in results.items():
    print(f"{name:<25} | {metrics['accuracy']:.4f}     | {metrics['precision']:.4f}     | {metrics['recall']:.4f}     | {metrics['f1']:.4f}")

# 3. Select the best model (Example: choosing based on highest Precision)
best_model_name = max(results, key=lambda k: results[k]['f1'])
best_model = models[best_model_name]

# 5. Print justification
print(f"\nSelected Best Model: {best_model_name}")
print("Reason for selection: Highest F1 Score among the evaluated models, indicating a good balance between precision and recall for spam detection.")


# 4. Print final validation metrics for the best model
print(f"\n--- Final Validation: {best_model_name} ---")
y_pred_best = best_model.predict(X_test_vec)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# comparison by visualization

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('beige')
fig.suptitle("Evaluation, Comparison & Final Validation", fontsize=16, fontweight='bold', color='darkblue', y=0.95)
ax.axis('off')
gs=plt.GridSpec(2, 2, figure=fig)

# Plotting comparison table

ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
table_data = [[name, f"{metrics['accuracy']:.4f}", f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}"] for name, metrics in results.items()]
table = ax1.table(cellText=table_data, colLabels=["Model", "Accuracy", "Precision", "Recall", "F1 Score"], loc='center', cellLoc='center', colColours=['lightblue']*5)
table.auto_set_font_size(False)
table.set_fontsize(12)

#plotting confusion matrix for the best model

ax2 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred_best)
im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax2.set_title(f"Confusion Matrix: {best_model_name}")
ax2.set_xticks(np.arange(len(set(y_test))))
ax2.set_yticks(np.arange(len(set(y_test))))
ax2.set_xticklabels(set(y_test))
ax2.set_yticklabels(set(y_test))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.colorbar(im, ax=ax2)   

#ploting the best model by classification report

ax3 = fig.add_subplot(gs[1, 1])
report = classification_report(y_test, y_pred_best, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.iloc[:-1, :-1]  # Exclude 'accuracy'
report_df.plot(kind='bar', ax=ax3)
ax3.set_title(f"Classification Report: {best_model_name}")

All_axes = [ax1, ax2, ax3]

plt.savefig("model_comparison.png", dpi=300,facecolor='beige')



 # Save the best model & vectorizer for app.py (Streamlit)
joblib.dump(best_model, "best_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved successfully!")

joblib.dump(results, "results.pkl")

cm = confusion_matrix(y_test, y_pred_best)
joblib.dump(cm, "confusion_matrix.pkl")


if __name__ == "__main__":
    print("Run each stage in order — no stage should run before the previous one is complete.")
