# 🎓 RETINA AI: Student Dropout Risk Prediction

## 📌 Overview

This project was developed for the **RETINA AI Competition: Predict Student Dropout Risk with Deep Learning**.

The goal is to identify students at risk of dropping out by analyzing attendance patterns and behavioral indicators extracted from educational records. The solution combines **deep learning**, **feature engineering**, and **ensemble learning** to classify students into:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

The proposed framework utilizes attendance time-series data, statistical attendance features, and machine learning models to accurately predict dropout risk at an early stage.

---

# 🚀 Project Highlights

✅ Multimodal Learning Approach

✅ Attendance Time-Series Modeling

✅ Extensive Feature Engineering

✅ Bidirectional LSTM Network

✅ CNN-LSTM Hybrid Architecture

✅ LightGBM Gradient Boosting

✅ 5-Fold Stratified Cross Validation

✅ Ensemble Learning

✅ Weighted F1 Score Optimization

---

# 📂 Dataset Description

The competition dataset consists of attendance records collected across multiple semesters.

### Attendance Data

| Column         | Description               |
| -------------- | ------------------------- |
| student_id     | Unique student identifier |
| semester       | Semester number           |
| week           | Academic week             |
| subject        | Subject category          |
| attendance_pct | Attendance percentage     |

### Dataset Statistics

* ~1 Million Attendance Records
* 15,000 Students
* 3 Semesters
* 8 Weeks per Semester
* Multiple Subject Categories

---

# 🔍 Problem Statement

Student dropout is a significant challenge for educational institutions.

The objective is to predict whether a student belongs to:

| Class       | Meaning              |
| ----------- | -------------------- |
| Low Risk    | Unlikely to Drop Out |
| Medium Risk | Moderate Risk        |
| High Risk   | Likely to Drop Out   |

Early prediction enables institutions to provide targeted interventions and improve student retention.

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

### Attendance Sequences

* Chronological ordering of attendance records
* Time-index creation
* Student-level sequence generation
* Forward-fill missing values
* Backward-fill missing values
* Zero-padding for fixed sequence length

### Tabular Features

* Label Encoding of target classes
* Standard Scaling of numerical features

---

# 🧠 Feature Engineering

More than 30 student-level features were created from raw attendance data.

## Global Statistics

* Mean Attendance
* Standard Deviation
* Median Attendance
* Minimum Attendance
* Maximum Attendance
* Interquartile Range (IQR)
* Attendance Range

## Threshold Features

* Percentage Below 50%
* Percentage Below 75%
* Percentage Above 90%
* Critical Attendance Count

## Subject-Level Features

For:

* Core_1
* Core_2
* Elective

Computed:

* Mean Attendance
* Standard Deviation
* Minimum Attendance

## Temporal Features

* Weekly Attendance Trend
* Polynomial Trend Slope

## Semester Features

* Semester Mean Attendance
* Semester Standard Deviation
* Semester-to-Semester Changes

## Consistency Features

* Coefficient of Variation
* Attendance Stability Indicators

---

# 🏗️ Model Architecture

## 1️⃣ Multimodal Bi-LSTM Network

### Sequence Branch

Input Shape:

24 Timesteps × 3 Features

Architecture:

Input
↓
Bi-LSTM (64)
↓
Bi-LSTM (32)
↓
Dense (64)
↓
BatchNorm
↓
Dropout (0.3)
↓
Dense (32)

### Tabular Branch

Input
↓
Dense (128)
↓
BatchNorm
↓
Dropout (0.3)
↓
Dense (64)
↓
BatchNorm
↓
Dropout (0.2)
↓
Dense (32)

### Fusion Layer

Concatenate
↓
Dense (64)
↓
BatchNorm
↓
Dropout (0.2)
↓
Dense (32)
↓
Softmax (3 Classes)

---

## 2️⃣ CNN-LSTM Hybrid Model

The project also includes an alternative architecture:

Input Sequence
↓
Conv1D (64)
↓
Conv1D (32)
↓
LSTM (64)
↓
Dense (32)

This model captures local attendance patterns before temporal modeling.

---

## 3️⃣ LightGBM Classifier

A gradient boosting model was trained on engineered features.

### Parameters

* n_estimators = 500
* learning_rate = 0.05
* max_depth = 6
* num_leaves = 31

---

# 🔄 Ensemble Strategy

Final predictions are generated using weighted averaging.

Prediction = 0.6 × Neural Network + 0.4 × LightGBM

Benefits:

* Better generalization
* Reduced overfitting
* Improved robustness

---

# 📊 Evaluation

## Validation Strategy

* 5-Fold Stratified Cross Validation

## Evaluation Metric

### Weighted F1 Score

Chosen because:

* Handles class imbalance
* Balances precision and recall
* Suitable for multi-class classification

---

# 📈 Key Insights

* Students with attendance below 50% for multiple weeks were highly correlated with dropout risk.
* Semester-to-semester attendance decline was one of the strongest predictors.
* Trend slope consistently ranked among the most important features.
* Combining deep learning and gradient boosting outperformed individual models.

---

# 🔬 Future Improvements

Potential enhancements include:

* Integration of counsellor notes using NLP
* Transformer-based text models (BERT)
* Academic performance features
* Demographic information
* Attention-based sequence models
* Explainable AI techniques (SHAP/LIME)

---

# 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Scikit-Learn
* TensorFlow / Keras
* LightGBM
* Matplotlib
* Seaborn

---

# 📁 Project Structure

RETINA-AI-Dropout-Prediction/

├── RETINA_AI_Dropout_Prediction.ipynb

├── data/

│ ├── Attendance_series_1_.csv

│ ├── student_info.csv

│ └── counsellor_notes.csv

├── models/

├── outputs/

├── README.md

└── requirements.txt

---

# 🏆 Competition

**RETINA AI — Predict Student Dropout Risk with Deep Learning**

This project demonstrates how attendance behavior can be transformed into actionable insights using deep learning and machine learning techniques for early student dropout prediction.

---

## 👨‍💻 Author

**Mayuri Khairnar**

Computer Science Engineering Student

K. K. Wagh Institute of Engineering Education and Research

Nashik, Maharashtra, India
