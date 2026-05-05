# 🎓 Student Grade Prediction — ML Project

## Overview
A supervised machine learning project that predicts students' final academic grade (G3) using demographic, behavioral, and academic features from the UCI Student Performance dataset. This project was completed as part of the **Intro to Machine Learning** course.

## Team
- Askarkbekova Ingkar — Report Lead & Presentation Lead
- Member 2 — EDA Lead
- Member 3 — Modelling Lead

## Dataset
- **Source:** [UCI ML Repository — Student Performance (ID 320)](https://archive.ics.uci.edu/dataset/320)
- **File used:** `student-por.csv`
- **Size:** 649 rows × 33 columns, no missing values
- **Task:** Regression — predict continuous target G3 (final grade, scale 0–20)

## Project structure
```
student-grade-prediction-ml/
├── student-por.csv
├── Week1_Proposal.pdf
├── Week2_EDA_Preprocessing.ipynb
├── Week3_Modelling.ipynb
├── Week4_FinalReport.pdf
└── README.md
```

## Pipeline summary
- **EDA:** distribution analysis, correlation heatmap, 7 visualizations — identified G1 & G2 as dominant predictors (r ≈ 0.83 and 0.92 with G3)
- **Preprocessing:** one-hot encoding (`drop_first=True`), feature engineering (`avg_grade`, `grade_trend`), StandardScaler
- **Train/test split:** 80/20, `random_state=42`
- **Models:** Linear Regression · KNN Regressor · Decision Tree Regressor
- **Evaluation:** RMSE, MAE, R², 5-fold cross-validation

## Results
```
Model                        RMSE    MAE     R²      CV R² (mean)
Linear Regression (baseline) 1.2149  0.7651  0.8487  0.8356  ← best
Decision Tree (max_depth=5)  1.3355  0.8011  0.8171  0.7618
KNN (k=5)                    2.1556  1.5569  0.5235  0.4901
```
Linear Regression achieved the best performance. The data's predominantly linear structure — driven by prior grades G1 and G2 — made complex models unnecessary. KNN underperformed due to the curse of dimensionality after one-hot encoding.

## Key findings
- G1 and G2 are by far the strongest predictors of G3; behavioral and social features add minimal signal in their presence
- 15 students with G3 = 0 form a structurally distinct subpopulation (likely dropouts or exam no-shows), causing systematic prediction error
- Past failures carry additional signal beyond G2 alone — students with failures cluster at lower grades even when G2 is moderate
- A two-stage hurdle model (classifier for dropout detection + regressor for active students) is proposed as a future improvement

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
Install with: `pip install pandas numpy matplotlib seaborn scikit-learn`
