"""Shared model paths and cached loaders for the Streamlit app sections."""

from pathlib import Path

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_MODEL_PATH = ROOT / "models" / "best_classification_model.pkl"
CLUSTERING_MODEL_PATH = ROOT / "models" / "gas_turbine_clustering_model.pkl"
REGRESSION_MODEL_PATH = ROOT / "models" / "best_regression_model.pkl"


@st.cache_resource
def load_classification_bundle():
    return joblib.load(CLASSIFICATION_MODEL_PATH)


@st.cache_resource
def load_clustering_bundle():
    return joblib.load(CLUSTERING_MODEL_PATH)


@st.cache_resource
def load_regression_bundle():
    return joblib.load(REGRESSION_MODEL_PATH)
