"""Classification section for the unified Streamlit app."""

import numpy as np
import pandas as pd
import streamlit as st

from common import CLASSIFICATION_MODEL_PATH, load_classification_bundle


AIRLINE_RATINGS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]


def make_airline_features(values: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Recreate the notebook's engineered and one-hot encoded feature row."""
    row = {
        "Age": values["Age"],
        "Flight Distance": values["Flight Distance"],
        **{name: values[name] for name in AIRLINE_RATINGS},
        "Departure Delay in Minutes": values["Departure Delay in Minutes"],
        "Arrival Delay in Minutes": values["Arrival Delay in Minutes"],
        "Loyal Business": int(
            values["Customer Type"] == "Loyal Customer"
            and values["Type of Travel"] == "Business travel"
        ),
        "Digital Score": np.mean(
            [
                values["Inflight wifi service"],
                values["Ease of Online booking"],
                values["Online boarding"],
            ]
        ),
        "Gender_Male": int(values["Gender"] == "Male"),
        "Customer Type_disloyal Customer": int(
            values["Customer Type"] == "disloyal Customer"
        ),
        "Type of Travel_Personal Travel": int(
            values["Type of Travel"] == "Personal Travel"
        ),
        "Class_Eco": int(values["Class"] == "Eco"),
        "Class_Eco Plus": int(values["Class"] == "Eco Plus"),
    }
    return pd.DataFrame([[row[column] for column in feature_columns]], columns=feature_columns)


def render() -> None:
    st.title("Airline Passenger Satisfaction")
    st.write("Enter passenger and service details to estimate the satisfaction class.")

    if not CLASSIFICATION_MODEL_PATH.exists():
        st.error(f"Saved model not found: {CLASSIFICATION_MODEL_PATH}")
        st.stop()
    bundle = load_classification_bundle()
    st.caption(f"Saved model: {bundle['model_name']} classifier")

    with st.form("airline_form"):
        a, b, c = st.columns(3)
        values = {
            "Gender": a.selectbox("Gender", ["Female", "Male"]),
            "Customer Type": b.selectbox(
                "Customer type", ["Loyal Customer", "disloyal Customer"]
            ),
            "Type of Travel": c.selectbox(
                "Travel type", ["Business travel", "Personal Travel"]
            ),
            "Class": a.selectbox("Travel class", ["Business", "Eco", "Eco Plus"]),
            "Age": b.number_input("Age", min_value=7, max_value=85, value=35, step=1),
            "Flight Distance": c.number_input(
                "Flight distance", min_value=31, max_value=4983, value=1000, step=50
            ),
            "Departure Delay in Minutes": a.number_input(
                "Departure delay (minutes)", min_value=0, max_value=1128, value=0, step=5
            ),
            "Arrival Delay in Minutes": b.number_input(
                "Arrival delay (minutes)", min_value=0.0, max_value=1115.0, value=0.0, step=5.0
            ),
        }
        st.markdown("**Service ratings** (0 = lowest, 5 = highest)")
        rating_columns = st.columns(3)
        for idx, name in enumerate(AIRLINE_RATINGS):
            values[name] = rating_columns[idx % 3].slider(name, 0, 5, 3)
        submitted = st.form_submit_button("Predict satisfaction", type="primary")

    if submitted:
        X = make_airline_features(values, bundle["feature_columns"])
        if bundle.get("scale_features", bundle.get("scaler") is not None):
            X_for_model = bundle["scaler"].transform(X)
        else:
            X_for_model = X
        prediction = bundle["model"].predict(X_for_model)
        label = bundle["label_encoder"].inverse_transform(prediction)[0]
        st.success(f"Predicted satisfaction: **{label.title()}**")

        if hasattr(bundle["model"], "predict_proba"):
            probabilities = bundle["model"].predict_proba(X_for_model)[0]
            probability_table = pd.DataFrame(
                {"Class": bundle["label_encoder"].classes_, "Probability": probabilities}
            ).sort_values("Probability", ascending=False)
            st.dataframe(
                probability_table.style.format({"Probability": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
