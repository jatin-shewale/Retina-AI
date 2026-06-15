# RETINA AI: Student Dropout Risk Prediction
## Kaggle Competition Writeup

---

## Project Title
**Multimodal Student Dropout Risk Predictor — Fusing Academic, Attendance & NLP Signals**

---

## Problem Understanding

Student dropout is one of the most persistent challenges in higher education. A student rarely drops out due to a single reason — it's almost always a combination of declining grades, poor attendance, financial stress, and emotional burnout building up over multiple semesters. Traditional approaches that look at CGPA alone miss this multi-dimensional nature of the problem entirely.

In this challenge, we are given three completely different types of data about the same 15,000 students: structured academic records, weekly attendance time-series across subjects and semesters, and free-text observations from counsellors. The task is to classify each student into one of three dropout risk levels — Low, Medium, or High — on a held-out test set of 3,000 students.

What makes this interesting is that each data source tells a different part of the story. A student might have decent CGPA but worrying attendance. Another might have average grades but a counsellor note that says "discussed dropping out due to financial stress." An approach that uses only one modality will miss these cases. So the real challenge here is not just building a model, but building the right features from each source and fusing them meaningfully.

---

## Data Preprocessing Steps

### Academic Data (train.csv / test.csv)
- Two columns had missing values: `parent_education` (238 missing in train) and `commute_time_mins` (387 missing).
- Both were imputed using the **median strategy** fitted only on training data to prevent leakage.
- Categorical columns (`branch`, `gender`, `hostel_status`, `family_income`, `parent_education`) were encoded with manual ordinal mappings based on domain logic (e.g., family income: Low=0, Medium=1, High=2) rather than one-hot encoding, keeping the feature space compact.

### Attendance Time-Series (Attendance_series.csv)
- 1,048,575 rows across 15,000 students, 3 semesters, 8 weeks, 3 subjects (Core_1, Core_2, Elective).
- Aggregated to per-student features: global mean, min, max, standard deviation, and median attendance.
- Per-semester averages were computed and a trend feature was derived (last semester attendance minus first semester attendance). A negative trend signals deteriorating engagement.
- Counted the number of weeks where attendance fell below 75% — a common academic threshold — as a separate risk indicator.
- Per-subject means were also calculated to check if poor attendance was subject-specific or global.

### Counsellor Notes (Counsellor_notes.csv)
- Free text with one note per student.
- Applied rule-based keyword matching (no target leakage) using curated dictionaries for high-risk phrases ("dropping out", "financial stress", "demotivated", "multiple backlogs"), medium-risk phrases ("struggling", "monitor", "action plan", "needs improvement"), and positive phrases ("performing well", "no further action").
- Derived a composite `note_risk_score` = 2 × high_risk_flag + 1 × medium_risk_flag − 1 × positive_flag.
- Also computed note length and word count as proxy features (longer notes typically indicate more concern).

---

## Feature Engineering

We created 35+ features across three modalities:

**Academic / Engineered:**
- `cgpa_mean` — average CGPA across all 4 semesters
- `cgpa_min` — worst semester GPA
- `cgpa_std` — consistency of performance
- `cgpa_trend` — cgpa_sem4 minus cgpa_sem1 (positive = improving)
- `cgpa_late_dip` — binary flag if GPA fell between sem2 and sem4
- `total_backlogs` — total backlog count across sem1–sem3
- `backlog_trend` — did backlogs worsen over time?
- `has_any_backlog` — binary flag
- `cgpa_backlog_ratio` — higher ratio = better student despite backlogs
- `risk_score_proxy` — total_backlogs minus cgpa_mean (simple domain-derived risk signal)

**Attendance (Time-Series):**
- `att_mean`, `att_min`, `att_max`, `att_std`, `att_median` — global stats
- `low_att_weeks` — count of weeks below 75%
- `att_sem1_mean`, `att_sem2_mean`, `att_sem3_mean` — per-semester averages
- `att_trend` — last vs first semester attendance
- `att_core_1_mean`, `att_core_2_mean`, `att_elective_mean` — per-subject averages

**NLP:**
- `high_risk_flag`, `medium_risk_flag`, `positive_flag` — binary keyword signals
- `note_length`, `word_count`
- `note_risk_score` — composite score

---

## Model Architecture / Approach

We trained three models individually and combined them with a **weighted soft-voting ensemble**.

### Individual Models

**XGBoost (weight: 40%)**
- 500 estimators, max_depth=6, learning_rate=0.05
- L1 (alpha=0.1) and L2 (lambda=1.0) regularisation
- Subsample and colsample_bytree = 0.8 to reduce overfitting
- `multi:softprob` objective for 3-class probabilities

**LightGBM (weight: 40%)**
- 500 estimators, 63 leaves, learning_rate=0.05
- Same subsample settings
- Generally slightly faster than XGBoost with comparable accuracy

**Random Forest (weight: 20%)**
- 300 trees, max_depth=10
- `class_weight='balanced'` to handle class imbalance (60/25/15 split)
- Acts as a diversity booster in the ensemble

### Ensemble Strategy
Instead of hard-voting (which uses only the predicted label), we average the **class probabilities** from all three models. This means a model that is "slightly confident" about Medium Risk but "very confident" about Low Risk will pass that nuance along — resulting in better calibrated final predictions.

### Leakage Prevention
This is a critical concern in competitions and real deployments. We took the following steps:

1. The `SimpleImputer` was fit **only on X_train** and then applied to X_val and X_test. This prevents information from validation/test distributions leaking into imputation.
2. Categorical encodings were **manual ordinal maps** based on domain knowledge, not learned from target distribution.
3. NLP features used **rule-based keyword dictionaries**, not any form of target encoding.
4. All attendance aggregations are pure statistical summaries with no reference to labels.
5. Cross-validation used `StratifiedKFold` — in each fold, only the training portion is visible to the model.

---

## Evaluation Methodology

- **Primary metric:** Accuracy on the hidden test set (Kaggle leaderboard)
- **Local validation:** 80/20 stratified train-validation split
- **Cross-validation:** 5-fold `StratifiedKFold` on the full training set to estimate generalisation
- `classification_report` with per-class precision, recall, and F1 for all 3 classes
- Confusion matrix for all four models (XGBoost, LightGBM, Random Forest, Ensemble)

---

## Results and Observations

- The ensemble consistently outperformed each individual model on validation.
- `cgpa_mean`, `total_backlogs`, and `att_mean` were the top 3 features by importance — confirming that academic performance and attendance are the strongest dropout signals.
- The NLP `note_risk_score` and `high_risk_flag` ranked in the top 10, showing that even simple keyword-based NLP adds value.
- `cgpa_trend` (whether CGPA improved or declined across semesters) was more predictive than raw CGPA in the final semester.
- Students with `total_backlogs > 2` and `att_mean < 0.75` were almost exclusively classified as High Risk.
- The Random Forest with `class_weight='balanced'` significantly improved recall on the minority High Risk class.

---

## Conclusion

We built a complete multimodal dropout risk prediction system that fuses three different data sources — academic records, weekly attendance time-series, and counsellor notes — into a single feature-rich representation, and trained a soft-voting ensemble of XGBoost, LightGBM, and Random Forest on top.

The key insight is that no single modality tells the full story. A student with decent grades but worsening attendance and a counsellor note mentioning financial stress is a real dropout risk that a grade-only model would miss entirely.

This kind of system, when deployed in an institution's student information portal, could automatically flag at-risk students at the start of each semester and give academic advisors an actionable, ranked list of students who need intervention — before it's too late.

---

## Mandatory Attachments

- `class_distribution.png` — Target class distribution (bar + pie)
- `train_val_split.png` — Train/validation split screenshot (training log equivalent)
- `confusion_matrices.png` — Confusion matrices for all 4 models
- `feature_importance.png` — Feature importance plot (coloured by modality)
- `model_comparison.png` — Accuracy comparison bar chart
- `cv_scores.png` — 5-fold cross-validation fold scores
- `cgpa_backlog_trends.png` — EDA: CGPA and backlog trends by risk group
- `submission.csv` — Final predictions

---

## Project Link
[Kaggle Notebook](https://www.kaggle.com/code/jatinshewale/notebook-retina-ai/edit)
[GitHub Link](https://github.com/jatin-shewale/Retina-AI.git)

---

*Submitted for RETINA AI: Predict Student Dropout Risk with Deep Learning — Kaggle Hackathon*
