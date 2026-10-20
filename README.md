git checkout 9eb1dac -- .
# Machine Learning Project

This project compares machine-learning algorithms across three tracks: airline satisfaction classification, e-commerce delivery-time regression, and gas-turbine clustering. A single Streamlit app lets you use all three saved models.

## Contents

- [Problem statements and prediction targets](#problem-statements-and-prediction-targets)
- [Selected models and metrics](#selected-models-and-metrics)
- [Classification algorithms](#classification-algorithms)
- [Regression algorithms](#regression-algorithms)
- [Clustering algorithms](#clustering-algorithms)
- [Project files](#project-files)
- [Install and run the app](#install-and-run-the-app)
- [Run the notebooks and regenerate models](#run-the-notebooks-and-regenerate-models)

## Problem statements and prediction targets

### Classification — airline passenger satisfaction

Given passenger details, travel information, delays, and service ratings, predict the passenger's satisfaction class: **satisfied** or **neutral/dissatisfied**. The notebook engineers features such as `Loyal Business` and `Digital Score`, then compares ten classifiers.

Notebook: [`notebooks/classification.ipynb`](notebooks/classification.ipynb)

### Regression — Olist delivery duration

Given order, product, freight, route, and purchase-time details, predict **actual delivery duration in days** (`delivery_days`), measured from purchase to delivery. The checkout's promised delivery date and features derived from it are excluded from the inputs. Training rows are also no longer filtered by whether they met the promise.

Notebook: [`notebooks/regression.ipynb`](notebooks/regression.ipynb)

### Clustering — gas-turbine operating conditions

Group observations by nine turbine operating measurements. CO and NOx emissions are kept out of clustering inputs and used afterward to describe the resulting groups. This is an unsupervised task, so there is no predicted target label.

Notebook: [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb)

## Selected models and metrics

| Track | Selected model | Evaluation metrics |
|---|---|---|
| Classification | **Random Forest** | Accuracy 0.9588 · Precision 0.9588 · Recall 0.9588 · F1 0.9588 · ROC-AUC 0.9933 |
| Regression | **Tuned Gradient Boosting Regressor** | Test R² 0.3359 · MAE 3.8315 days |
| Clustering | **K-Means (k=3)**, used by the app | Silhouette 0.3066 (5,000-row estimate) · Davies–Bouldin 1.2180 · Calinski–Harabasz 20,436.26 (full 36,726-row dataset) |

Classification metrics are from the saved test evaluation. Regression metrics are from the held-out test split after retraining without checkout ETA features. Clustering has no single best-model score: K-Means was selected for the interactive app because it can assign a cluster to a new observation. The notebook also exports Agglomerative clustering; its Ward-linkage evaluation on a 2,000-row sample was Silhouette 0.2845, Davies–Bouldin 1.1750, and Calinski–Harabasz 933.61. These clustering metric sets use different evaluation scopes and should not be compared directly.

## Classification algorithms

The classification notebook evaluates these algorithms in order:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine (SVM)
6. Random Forest Classifier — selected model
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier
10. Multi-layer Perceptron (MLP) Classifier

## Regression algorithms

The regression notebook evaluates these algorithms in order:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor — selected after hyperparameter tuning
9. Support Vector Regression (SVR)
10. K-Nearest Neighbors (KNN) Regressor

Gradient Boosting and Random Forest are tuned with `RandomizedSearchCV`. The selected Gradient Boosting model's saved test metrics are R² 0.3359 and MAE 3.8315 days.

## Clustering algorithms

The clustering notebook compares:

1. **K-Means** — compares candidate cluster counts, then fits the selected `k=3` model on all cleaned rows. The app uses this model to assign new observations.
2. **Agglomerative Hierarchical Clustering** — compares Ward, complete, and average linkage on a fixed sample; the selected Ward model is exported for reuse. Agglomerative estimators do not provide a `predict` method for unseen observations.

## Project files

```text
.
├── app/                 # Shared entry point and separate Streamlit track sections
│   ├── app.py
│   ├── classification_app.py
│   ├── clustering_app.py
│   ├── regression_app.py
│   └── common.py
├── data/                # Source dataset archives
├── models/              # Saved Joblib model bundles
├── notebooks/           # Data analysis, training, and evaluation notebooks
├── README.md
└── requirements.txt
```

Saved models:

- `models/best_classification_model.pkl`
- `models/best_regression_model.pkl`
- `models/gas_turbine_clustering_model.pkl`
- `models/gas_turbine_agglomerative_model.pkl`

## Install and run the app

Run these commands from the project root. Python and pip must be installed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/app.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead of the `source` command.

Streamlit opens the app in your browser. If it does not, visit [http://localhost:8501](http://localhost:8501). Use the navigation at the top of the page to switch between Regression, Clustering, and Classification.

## Run the notebooks and regenerate models

The app uses the saved models and does not need the training datasets. To rerun a notebook, install the requirements, extract its source dataset as described in the notebook, and run its cells from top to bottom.

The regression export cell writes `models/best_regression_model.pkl`. The clustering export cell writes the K-Means bundle used by the app and the selected Agglomerative bundle. The classification notebook exports its selected classifier and preprocessing objects to `models/best_classification_model.pkl`.
