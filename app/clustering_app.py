"""Clustering section for the unified Streamlit app."""

import pandas as pd
import streamlit as st

from common import CLUSTERING_MODEL_PATH, load_clustering_bundle


FEATURE_LABELS = {
    "AT": "Ambient temperature",
    "AP": "Ambient pressure",
    "AH": "Ambient humidity",
    "AFDP": "Air-filter differential pressure",
    "GTEP": "Gas-turbine exhaust pressure",
    "TIT": "Turbine inlet temperature",
    "TAT": "Turbine after temperature",
    "TEY": "Turbine energy yield",
    "CDP": "Compressor discharge pressure",
}
DEFAULT_VALUES = {
    "AT": 17.801,
    "AP": 1012.6,
    "AH": 80.47,
    "AFDP": 3.9377,
    "GTEP": 25.104,
    "TIT": 1085.9,
    "TAT": 549.88,
    "TEY": 133.73,
    "CDP": 11.965,
}
BOUNDS = {
    "AT": (-6.2348, 37.103),
    "AP": (985.85, 1036.6),
    "AH": (24.085, 100.2),
    "AFDP": (2.0874, 7.6106),
    "GTEP": (17.698, 40.716),
    "TIT": (1000.8, 1100.9),
    "TAT": (511.04, 550.61),
    "TEY": (100.02, 179.5),
    "CDP": (9.8518, 15.159),
}


def render() -> None:
    st.title("Gas-Turbine Operating-Condition Clustering")
    st.write(
        "Enter nine operating measurements. K-Means assigns the observation to a cluster; "
        "emissions are shown only to help interpret that group."
    )

    if not CLUSTERING_MODEL_PATH.exists():
        st.error(f"Saved model not found: {CLUSTERING_MODEL_PATH}")
        st.info(
            "Run notebooks/clustering.ipynb from top to bottom. "
            "Its model-export cell saves the trained model bundle into the models folder."
        )
        st.stop()
    bundle = load_clustering_bundle()
    st.caption(
        f"K-Means with k={bundle['n_clusters']} · fitted on "
        f"{bundle['training_rows']:,} cleaned observations · current notebook configuration."
    )

    with st.form("turbine_form"):
        columns = st.columns(3)
        values = {}
        for index, feature in enumerate(bundle["feature_columns"]):
            low, high = BOUNDS[feature]
            values[feature] = columns[index % 3].number_input(
                FEATURE_LABELS[feature],
                min_value=float(low),
                max_value=float(high),
                value=float(DEFAULT_VALUES[feature]),
                format="%.4f",
            )
        submitted = st.form_submit_button("Assign cluster", type="primary")

    if submitted:
        X = pd.DataFrame(
            [[values[name] for name in bundle["feature_columns"]]],
            columns=bundle["feature_columns"],
        )
        X_scaled = bundle["scaler"].transform(X)
        cluster = int(bundle["model"].predict(X_scaled)[0])
        st.success(f"Assigned to **Cluster {cluster}**")
        st.caption("Cluster numbers are identifiers; they do not indicate rank or quality.")

        counts = bundle["cluster_counts"]
        st.metric("Training observations in this cluster", f"{counts[cluster]:,}")
        st.markdown("**Typical values for this cluster** (training-set means)")
        profile = pd.Series(bundle["cluster_means"][cluster], name="Mean")
        st.dataframe(profile.to_frame().style.format("{:.3f}"), use_container_width=True)
        medians = pd.Series(bundle["emission_medians"][cluster], name="Median")
        st.markdown("**Emissions in the training data** (post-clustering medians)")
        st.dataframe(medians.to_frame().style.format("{:.3f}"), use_container_width=True)
        st.caption("CO and NOx are not inputs to the clustering model.")
