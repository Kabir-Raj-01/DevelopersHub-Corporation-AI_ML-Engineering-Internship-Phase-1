# AI/ML Engineering Internship Tasks — DevelopersHub Corporation

> Completed by: Kabir Raj

---

## Overview

This repository contains solutions to 3 AI/ML Engineering Internship tasks from DevelopersHub Corporation. Each task is implemented as a complete, well-commented Jupyter Notebook covering data loading, preprocessing, EDA, model training, evaluation, and interpretation.

---

## Task Summary

| # | Task | Dataset | Models | Key Metric |
|---|------|---------|--------|------------|
| 1 | Iris Dataset Visualization | Iris (seaborn) | — (EDA only) | Visualizations |
| 2 | Stock Price Prediction | AAPL via yfinance | Linear Regression, Random Forest | MAE, R² |
| 3 | Heart Disease Prediction | Heart Disease UCI | Logistic Regression, Decision Tree | ROC-AUC, Accuracy |
| 4 | Health Query Chatbot | — (LLM API) | Claude / GPT via API | Qualitative |
| 5 | Mental Health Chatbot | EmpatheticDialogues | DistilGPT2 (fine-tuned) | Training Loss |
| 6 | House Price Prediction | California Housing | Linear Regression, Gradient Boosting | MAE, RMSE, R² |

---

## Task 1: Exploring and Visualizing the Iris Dataset

**Objective:** Load, inspect, and visualize the Iris dataset to understand data trends and distributions.

**Dataset:** Iris Dataset (150 samples × 5 columns, 3 species)

**Approach:**
- Loaded via `seaborn.load_dataset('iris')`
- Inspected with `.head()`, `.info()`, `.describe()`
- Visualized with: pairplot, scatter plots, histograms, box plots, correlation heatmap

**Key Results:**
- Petal length & petal width are the most discriminating features between species
- Dataset is perfectly balanced (50 samples per species) with no missing values
- Petal length ↔ petal width correlation: r = 0.96

**Skills:** pandas, seaborn, matplotlib, EDA

---

## Task 2: Stock Price Prediction (AAPL)

**Objective:** Predict the next day's closing price using historical OHLCV data.

**Dataset:** Apple (AAPL) stock — 2022–2024 via `yfinance`

**Features:** Open, High, Low, Close, Volume + engineered features (moving averages, lag values, daily return, price range)

**Models:**
- Linear Regression
- Random Forest Regressor (n_estimators=200)

**Key Results:**
- Random Forest outperforms Linear Regression due to non-linear relationships
- Most important feature: `Close_Lag1` (yesterday's closing price)
- Both models achieve high R² (>0.95) on the test set

**Skills:** yfinance API, time-series feature engineering, regression, sklearn

---

## Task 3: Heart Disease Prediction

**Objective:** Classify whether a patient is at risk of heart disease.

**Dataset:** Heart Disease UCI Dataset (303 samples, 13 features, binary target)

**Models:**
- Logistic Regression
- Decision Tree Classifier (max_depth=5)

**Evaluation:**
- Confusion matrix
- ROC-AUC curve
- Classification report (precision, recall, F1)
- Feature importance analysis

**Key Results:**
- Top features: Chest Pain Type, Max Heart Rate (thalach), ST Depression (oldpeak), Thal
- Exercise-related features are strongest predictors — consistent with medical literature
- Logistic Regression and Decision Tree both achieve ~80%+ accuracy

**Skills:** Binary classification, sklearn, ROC-AUC, feature importance
