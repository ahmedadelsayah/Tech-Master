import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spam Detection System", layout="wide")

# Load everything saved by main.py
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
results = joblib.load("results.pkl")
cm = joblib.load("confusion_matrix.pkl")

st.title("📧 Spam Detection System")

# ---------- Input & Prediction ----------
message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if message:
        vec = vectorizer.transform([message])
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        spam_prob = proba[classes.index("spam")]
        ham_prob = proba[classes.index("ham")]

        if spam_prob >= 0.5:
            st.error("⚠️ SPAM")
        else:
            st.success("✅ HAM")

        st.write(f"Spam Probability: {spam_prob*100:.1f}%")
        st.progress(float(spam_prob))
        st.write(f"Ham Probability: {ham_prob*100:.1f}%")
        st.progress(float(ham_prob))
    else:
        st.warning("Please enter a message to classify.")

st.divider()

# ---------- Confusion Matrix / Accuracy / Model Comparison ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"])
    ax.set_yticklabels(["Ham", "Spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="black", fontsize=14, fontweight="bold")
    st.pyplot(fig)

with col2:
    st.subheader("Accuracy")
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    st.metric(best_model_name, f"{results[best_model_name]['accuracy']*100:.1f}%")

with col3:
    st.subheader("Model Comparison")
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    names = list(results.keys())
    scores = [results[n]["accuracy"] for n in names]
    colors = ["darkblue" if n == best_model_name else "lightgray" for n in names]
    ax2.bar(names, scores, color=colors)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Accuracy")
    for i, v in enumerate(scores):
        ax2.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=9)
    st.pyplot(fig2)