# Optimized Spam Detection System

## Project 03 — TechMaster Academy (ML Fundamentals)

A Machine Learning system that classifies SMS/Email messages as **Spam** or **Ham** (legitimate messages).

The project compares **three different classification models**, evaluates their performance using multiple metrics, and selects the best-performing model based on **Precision**.

---

# 📌 Project Overview

Spam messages are unwanted or potentially harmful messages that can appear in SMS or email communication.

In this project, we build a complete Machine Learning pipeline that:

1. Loads and cleans the dataset.
2. Splits the data into training and testing sets.
3. Converts text messages into numerical features using **TF-IDF**.
4. Trains three different classification models.
5. Tunes the models where required.
6. Evaluates and compares their performance.
7. Selects the best model.
8. Saves the best model and vectorizer.
9. Provides an interactive **Streamlit** application for real-time predictions.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Build a complete Machine Learning classification pipeline.
- Detect whether a message is **Spam** or **Ham**.
- Apply text preprocessing and feature extraction.
- Use **TF-IDF** to convert text into numerical features.
- Train multiple Machine Learning classification models.
- Compare the models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Select the best model based on **Precision**.
- Build an interactive interface using Streamlit.

---

# 👥 Team Members & Responsibilities

## 👩 Eman Mohamed — Stage 1

### File:
```text
preprocess.py
```

### Responsibilities:

- Load the dataset.
- Inspect the dataset.
- Clean the data.
- Handle duplicate or missing data when required.
- Prepare the messages and labels.
- Split the dataset into:
  - Training data
  - Testing data

### Stage Output:

Clean and prepared data ready for feature extraction.

---

# 👨 Ahmed Adel — Stage 2 & Stage 3

## Stage 2 — Feature Extraction

### File:
```text
features.py
```

### Responsibilities:

- Convert text messages into numerical features.
- Use **TF-IDF (Term Frequency–Inverse Document Frequency)**.
- Fit the vectorizer on the training data.
- Transform the training and testing messages.
- Prepare the features for Machine Learning models.

---

## Stage 3 — Model Training

### File:
```text
train.py
```

### Responsibilities:

Train and prepare the three classification models:

### 1. Multinomial Naive Bayes

- Used for text classification.
- Tuned using `GridSearchCV`.
- The `alpha` parameter is optimized.

### 2. Logistic Regression

- Used as one of the classification models.
- Trained using the extracted TF-IDF features.

### 3. SVM (SVC)

- Support Vector Machine classifier.
- Trained using the extracted TF-IDF features.

### Stage Output:

Trained models ready to be evaluated in Stage 4.

---

# 👩 Haneen Riad — Stage 4

## Model Evaluation & Comparison

### File:
```text
evaluate.py
```

### Responsibilities:

- Run the complete Machine Learning pipeline.
- Evaluate all three models.
- Calculate performance metrics.
- Compare the models.
- Create the Confusion Matrix.
- Generate the Classification Report.
- Select the best model.

### Evaluation Metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Best Model Selection

The main criterion for selecting the best model is:

> **Highest Precision**

This is important because in spam detection, incorrectly classifying a legitimate message (**Ham**) as Spam is an important type of error.

### Stage Output:

```text
model_comparison.png
best_model.pkl
vectorizer.pkl
```

---

# 👩 Jaidaa Ahmed — Streamlit

## Interactive Application

### File:

```text
app.py
```

### Responsibilities:

Build an interactive web interface using **Streamlit**.

The application allows the user to:

- Enter a message.
- Click the Predict button.
- Get the prediction:
  - SPAM
  - HAM
- View the probability for each class.
- View the Confusion Matrix.
- View the final model Accuracy.
- View the comparison between the three models.

---

# 📁 Project Structure

```text
optimized-spam-detector/
│
├── data/
│   └── spam.csv
│
├── exploration.ipynb
│
├── preprocess.py
│
├── features.py
│
├── train.py
│
├── evaluate.py
│
├── app.py
│
├── requirements.txt
│
├── best_model.pkl
│
├── vectorizer.pkl
│
├── model_comparison.png
│
└── README.md
```

---

# 🔄 Project Pipeline

The complete project follows this workflow:

```text
spam.csv
   │
   ▼
Stage 1
preprocess.py
   │
   │ Load
   │ Inspect
   │ Clean
   │ Split
   ▼
Stage 2
features.py
   │
   │ TF-IDF
   ▼
Stage 3
train.py
   │
   ├── Multinomial Naive Bayes
   ├── Logistic Regression
   └── SVM (SVC)
   │
   ▼
Stage 4
evaluate.py
   │
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1-Score
   ├── Confusion Matrix
   └── Classification Report
   │
   ▼
Best Model
   │
   ├── best_model.pkl
   └── vectorizer.pkl
   │
   ▼
Streamlit
app.py
   │
   ▼
SPAM / HAM Prediction
```

---

# ⚙️ Installation

## 1. Clone or download the project

Make sure the project folder contains all required files.

---

## 2. Create a Virtual Environment

### Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux:

```bash
python -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# 📦 Required Libraries

The project uses:

```text
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
Joblib
```

The exact dependencies should be listed in:

```text
requirements.txt
```

---

# ▶️ How to Run the Project

## Step 1 — Run the Machine Learning Pipeline

Run:

```bash
python evaluate.py
```

This runs the pipeline in the following order:

```text
preprocess.py
      ↓
features.py
      ↓
train.py
      ↓
evaluate.py
```

---

## Step 2 — Check the Generated Files

After the evaluation is completed, the following files should be available:

```text
best_model.pkl
vectorizer.pkl
model_comparison.png
```

The first two files are required by the Streamlit application.

---

# 🌐 Run Streamlit Application

After `evaluate.py` finishes successfully, run:

```bash
streamlit run app.py
```

The application will normally open in the browser at:

```text
http://localhost:8501
```

---

# 🖥️ Streamlit Features

The application provides the following features:

## 1. Input Message

The user can enter an SMS or email message.

Example:

```text
Congratulations! You have won a free prize.
```

---

## 2. Prediction

The application predicts whether the message is:

```text
SPAM
```

or:

```text
HAM
```

---

## 3. Prediction Probability

The application displays the probability associated with each class.

For example:

```text
SPAM: 95%
HAM: 5%
```

---

## 4. Confusion Matrix

The application displays a visual Confusion Matrix showing the performance of the final model on the test dataset.

---

## 5. Accuracy

The final model's accuracy on unseen test data is displayed.

---

## 6. Model Comparison

The application displays a comparison between the three trained models.

The comparison includes:

```text
Accuracy
Precision
Recall
F1-Score
```

---

# 🤖 Machine Learning Models

The project uses three classification models.

| Model | Description |
|---|---|
| Multinomial Naive Bayes | Text classification model tuned using GridSearchCV |
| Logistic Regression | Classification model trained using TF-IDF features |
| SVM (SVC) | Support Vector Machine classifier |

---

# 📊 Evaluation Metrics

## Accuracy

Measures the percentage of all predictions that are correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Precision

Measures how many messages predicted as Spam are actually Spam.

```text
Precision =
True Positives / (True Positives + False Positives)
```

Precision is particularly important in this project because we want to avoid classifying legitimate messages as Spam.

---

## Recall

Measures how many actual Spam messages were correctly detected.

```text
Recall =
True Positives / (True Positives + False Negatives)
```

---

## F1-Score

The F1-Score combines Precision and Recall.

```text
F1 =
2 × (Precision × Recall) /
(Precision + Recall)
```

---

# 🎯 Model Selection

The project uses **Precision as the main criterion** for selecting the best model.

The model with the highest Precision is selected as the final model.

The selected model is saved as:

```text
best_model.pkl
```

The TF-IDF vectorizer used with the model is saved as:

```text
vectorizer.pkl
```

---

# 📈 Model Comparison

The project generates:

```text
model_comparison.png
```

This file contains the comparison results between the three models, including the evaluation results and visualizations such as the Confusion Matrix and Classification Report.

---

# 💾 Saved Files

## `best_model.pkl`

Contains the best-performing trained Machine Learning model.

It is loaded by the Streamlit application to make predictions.

---

## `vectorizer.pkl`

Contains the trained TF-IDF vectorizer.

It is required to convert new user messages into the same numerical feature representation used during model training.

---

## `model_comparison.png`

Contains the model comparison and evaluation visualization.

---

# 🧪 Example

Input:

```text
Congratulations! You have won a free lottery ticket. Call now!
```

Possible output:

```text
Prediction: SPAM
```

Another example:

```text
Hey, are we still meeting today at 5 PM?
```

Possible output:

```text
Prediction: HAM
```

---

# 🔗 Team Workflow

The team members should work in this order:

```text
Eman Mohamed
     │
     ▼
Stage 1
preprocess.py
     │
     ▼
Ahmed Adel
     │
     ├── Stage 2
     │   features.py
     │
     └── Stage 3
         train.py
     │
     ▼
Haneen Riad
     │
     ▼
Stage 4
evaluate.py
     │
     ├── best_model.pkl
     ├── vectorizer.pkl
     └── model_comparison.png
     │
     ▼
Jaidaa Ahmed
     │
     ▼
Streamlit
app.py
```

---

# 👥 Team Summary

| Team Member | Responsibility | Main File |
|---|---|---|
| **Eman Mohamed** | Data Loading, Inspection, Cleaning & Splitting | `preprocess.py` |
| **Ahmed Adel** | TF-IDF Feature Extraction + Model Training | `features.py`, `train.py` |
| **Haneen Riad** | Evaluation, Comparison & Best Model Selection | `evaluate.py` |
| **Jaidaa Ahmed** | Interactive Streamlit Application | `app.py` |

---

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Joblib**
- **Streamlit**

---

# 📓 Optional Exploration

The project may also contain:

```text
exploration.ipynb
```

This notebook can be used for exploratory data analysis (EDA), understanding the dataset, and experimenting with the data before building the final pipeline.

---

# 🚫 Out of Scope

The following technologies are not included in this project:

- Deep Learning
- Neural Networks
- Transformers
- LLMs
- Cloud Deployment
- Production APIs
- MLOps

---

# ⚠️ Important Notes

### Before running Streamlit

You must run:

```bash
python evaluate.py
```

first.

This is necessary because `app.py` depends on:

```text
best_model.pkl
vectorizer.pkl
```

Without these files, the Streamlit application cannot load the final trained model.

---

# 🚀 Final Project Result

At the end of the project, the system provides:

```text
SMS/Email Message
       │
       ▼
   TF-IDF
       │
       ▼
Best ML Model
       │
       ▼
 ┌─────────────┐
 │             │
 ▼             ▼
SPAM          HAM
```

The final application allows users to test messages interactively and view the performance of the Machine Learning models.

---

## Project Team

**TechMaster Academy — ML Fundamentals**

### Team Members

- Eman Mohamed — Stage 1
- Ahmed Adel — Stage 2 & Stage 3
- Haneen Riad — Stage 4
- Jaidaa Ahmed — Streamlit