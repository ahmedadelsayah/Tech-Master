"""
Stage 4 — Evaluation, Comparison & Final Validation

" " "
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import preprocess
import features
import train


def evaluate_models(models, X_test_vec, y_test):
    """Compute accuracy/precision/recall/F1 for each model."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_vec)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="spam"),
            "recall": recall_score(y_test, y_pred, pos_label="spam"),
            "f1": f1_score(y_test, y_pred, pos_label="spam"),
        }
    return results


def print_comparison_table(results):
    print("\n--- Model Comparison ---")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 75)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['accuracy']:.4f}     | {metrics['precision']:.4f}     "
              f"| {metrics['recall']:.4f}     | {metrics['f1']:.4f}")


def select_best_model(models, results):
    """Pick the model with the highest precision (fewest false positives)."""
    best_model_name = max(results, key=lambda k: results[k]["precision"])
    best_model = models[best_model_name]

    print(f"\nSelected Best Model: {best_model_name}")
    print("Reason: For spam detection, precision is crucial to avoid flagging "
          "legitimate messages (false positives) as spam.")

    return best_model_name, best_model


def final_validation(best_model_name, best_model, X_test_vec, y_test):
    print(f"\n--- Final Validation: {best_model_name} ---")
    y_pred_best = best_model.predict(X_test_vec)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
    return y_pred_best


def plot_results(results, y_test, y_pred_best, best_model_name, save_path="model_comparison.png"):
    """Build the comparison table + confusion matrix + classification report figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("beige")
    fig.suptitle("Evaluation, Comparison & Final Validation", fontsize=16,
                 fontweight="bold", color="darkblue", y=0.95)
    ax.axis("off")
    gs = plt.GridSpec(2, 2, figure=fig)

    # Comparison table
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis("off")
    table_data = [
        [name, f"{m['accuracy']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"]
        for name, m in results.items()
    ]
    table = ax1.table(
        cellText=table_data,
        colLabels=["Model", "Accuracy", "Precision", "Recall", "F1 Score"],
        loc="center", cellLoc="center", colColours=["lightblue"] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Confusion matrix
    ax2 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, y_pred_best)
    im = ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.set_title(f"Confusion Matrix: {best_model_name}")
    ax2.set_xticks(np.arange(len(set(y_test))))
    ax2.set_yticks(np.arange(len(set(y_test))))
    ax2.set_xticklabels(set(y_test))
    ax2.set_yticklabels(set(y_test))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.colorbar(im, ax=ax2)

    # Classification report bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    report = classification_report(y_test, y_pred_best, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.iloc[:-1, :-1]  # exclude 'accuracy' row
    report_df.plot(kind="bar", ax=ax3)
    ax3.set_title(f"Classification Report: {best_model_name}")

    plt.savefig(save_path, dpi=300, facecolor="beige")
    plt.close(fig)  # no plt.show() — keeps the script non-blocking


def run(models, X_test_vec, y_test):
    """Run the full Stage 4 pipeline and return the best model + its name."""
    results = evaluate_models(models, X_test_vec, y_test)
    print_comparison_table(results)

    best_model_name, best_model = select_best_model(models, results)
    y_pred_best = final_validation(best_model_name, best_model, X_test_vec, y_test)
    plot_results(results, y_test, y_pred_best, best_model_name)

    return best_model_name, best_model


def main():
    """Run the entire pipeline end-to-end: Stage 1 -> 2 -> 3 -> 4."""
    # Stage 1 — Data
    X_train, X_test, y_train, y_test = preprocess.run()

    # Stage 2 — Feature Extraction
    vectorizer, X_train_vec, X_test_vec = features.extract_features(X_train, X_test)

    # Stage 3 — Model Training & Tuning
    models = train.run(X_train_vec, y_train)

    # Stage 4 — Evaluation, Comparison & Final Validation
    best_model_name, best_model = run(models, X_test_vec, y_test)

    # Save the best model & vectorizer for app.py (Streamlit)
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("\nModel and vectorizer saved successfully!")


if __name__ == "__main__":
    main()
