# ML Evaluation

Evaluation of classical machine-learning algorithms on two independent tasks: a **classification** track predicting airline passenger satisfaction, and a **regression** track predicting e-commerce delivery time. Each track runs multiple algorithms through the same clean/split pipeline, tunes hyperparameters via grid/randomized search, and compares results on a shared leaderboard.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/                          # Zipped source datasets
│   ├── airlinepassengers_dataset.zip
│   └── olist_dataset.zip
├── notebooks/
│   ├── classification.ipynb       # Airline passenger satisfaction
│   └── regression.ipynb           # Olist order delivery time
├── models/                        # Saved best model per track (joblib)
│   ├── best_classification_model.pkl
│   └── best_regression_model.pkl
└── app/                           # Reserved for GUI/deployment code
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Unzip the datasets in `data/` before running the notebooks, then open `notebooks/` with Jupyter or VS Code.

## Classification track — Airline Passenger Satisfaction

Predicts `satisfaction` (satisfied / neutral or dissatisfied) from ~26k passenger survey records.

**Pipeline:** dataset audit → EDA → cleaning (median imputation, IQR winsorizing of delay outliers) → feature engineering (`Loyal Business` interaction flag, `Digital Score`) → 80/20 stratified split → scaling for distance-sensitive models.

**Algorithms:** Logistic Regression, K-Nearest Neighbors (grid-searched k), Gaussian Naive Bayes, Decision Tree (grid-searched max_depth), Support Vector Machine (grid-searched C/kernel).

| Model | F1 | AUC |
|---|---|---|
| **SVM (C=10, RBF)** | **0.9532** | **0.9887** |
| Decision Tree (depth=10) | 0.9424 | 0.9780 |
| KNN (k=9) | 0.9210 | 0.9738 |
| Logistic Regression | 0.8715 | — |
| Gaussian Naive Bayes | 0.8669 | — |

SVM is the best overall performer; the Decision Tree is the best interpretable alternative. Naive Bayes underperforms because its feature-independence assumption is violated by the engineered features.

## Regression track — Olist Delivery Time

Predicts `delivery_days` for ~96k delivered Olist marketplace orders.

**Pipeline:** dataset audit → EDA → leakage-safe cleaning (drops `review_score` and any post-dispatch timestamps) → domain feature engineering (Haversine `distance_km`/`log_distance`, `is_interstate`, `is_same_city`, `estimated_days`, `freight_ratio`, weekend/day-of-week flags) → 80/20 stratified split (by target decile) → train-only category encoding and scaling.

**Algorithms:** Linear, Ridge, Lasso, Elastic Net, Polynomial, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regression, KNN Regression, with `RandomizedSearchCV` tuning of the top 2 (Gradient Boosting, Random Forest).

| Model | Test R² | MAE (days) |
|---|---|---|
| **Gradient Boosting (tuned)** | **0.4353** | **3.32** |
| Random Forest (tuned) | 0.4279 | 3.35 |
| Random Forest (baseline, n=150) | 0.4245 | 3.36 |
| Gradient Boosting (baseline) | 0.4220 | 3.37 |

Gradient Boosting edges out Random Forest after tuning. `total_freight`, `distance_km`, and product weight/dimensions are the strongest predictors of delivery duration.

## Models

The best model from each track is persisted with `joblib` in `models/` for reuse without retraining.
