"""Regression form, feature preparation, and prediction for the unified app."""

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from common import REGRESSION_MODEL_PATH, load_regression_bundle


BRAZIL_STATES = {
    "AC": "Norte", "AL": "Nordeste", "AP": "Norte", "AM": "Norte",
    "BA": "Nordeste", "CE": "Nordeste", "DF": "Centro-Oeste", "ES": "Sudeste",
    "GO": "Centro-Oeste", "MA": "Nordeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste", "MG": "Sudeste", "PA": "Norte", "PB": "Nordeste",
    "PR": "Sul", "PE": "Nordeste", "PI": "Nordeste", "RJ": "Sudeste",
    "RN": "Nordeste", "RS": "Sul", "RO": "Norte", "RR": "Norte",
    "SC": "Sul", "SP": "Sudeste", "SE": "Nordeste", "TO": "Norte",
}
PRODUCT_CATEGORIES = [
    "bed_bath_table", "health_beauty", "sports_leisure", "computers_accessories",
    "furniture_decor", "housewares", "watches_gifts", "telephony", "auto", "toys",
    "cool_stuff", "garden_tools", "perfumery", "baby", "electronics", "other",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return float(2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def make_regression_features(values: dict, bundle: dict) -> pd.DataFrame:
    """Build the engineered, encoded, and scaled row used by the regression notebook."""
    distance = haversine_km(
        values["seller_lat"], values["seller_lng"],
        values["customer_lat"], values["customer_lng"],
    )
    purchase_dow = values["purchase_date"].weekday()
    product_volume = values["product_length"] * values["product_height"] * values["product_width"]
    grouped_category = (
        values["category"] if values["category"] in bundle["top_categories"] else "other"
    )

    numeric_values = {
        "distance_km": distance,
        "log_distance": float(np.log1p(distance)),
        "is_interstate": int(values["customer_state"] != values["seller_state"]),
        "is_same_city": int(values["same_city"]),
        "total_freight": values["freight"],
        "freight_ratio": values["freight"] / (values["price"] + 1.0),
        "freight_per_km": values["freight"] / (distance + 1.0),
        "log_weight": float(np.log1p(values["weight"])),
        "product_volume_cm3": float(product_volume),
        "purchase_dow": purchase_dow,
        "is_weekend_order": int(purchase_dow in [4, 5, 6]),
    }
    numeric = pd.DataFrame(
        [[numeric_values[name] for name in bundle["numeric_columns"]]],
        columns=bundle["numeric_columns"],
    )
    categories = pd.DataFrame(
        [[BRAZIL_STATES[values["customer_state"]], grouped_category]],
        columns=["customer_region", "category_grouped"],
    )
    encoded = pd.DataFrame(
        bundle["one_hot_encoder"].transform(categories),
        columns=bundle["one_hot_encoder"].get_feature_names_out(),
    )
    features = pd.concat([numeric, encoded], axis=1)
    continuous = bundle["continuous_columns"]
    features.loc[:, continuous] = bundle["scaler"].transform(features[continuous])
    return features.reindex(columns=bundle["feature_columns"], fill_value=0)


def render() -> None:
    st.title("Olist Delivery Time")
    st.write("Estimate actual delivery duration from order details, without requiring the checkout ETA.")
    if not REGRESSION_MODEL_PATH.exists():
        st.error(f"Saved model not found: {REGRESSION_MODEL_PATH}")
        st.stop()
    bundle = load_regression_bundle()
    st.caption(
        f"Model: {bundle['model_name']} · test R²: {bundle['test_r2']:.3f} · "
        f"test MAE: {bundle['test_mae']:.2f} days"
    )

    with st.form("delivery_form"):
        st.subheader("Route")
        route1, route2 = st.columns(2)
        purchase_date = route1.date_input("Order date", value=date(2017, 6, 1))
        same_city = route2.checkbox("Seller and customer are in the same city")

        route4, route5, route6 = st.columns(3)
        state_options = sorted(BRAZIL_STATES)
        sp_index = state_options.index("SP")
        seller_state = route4.selectbox("Seller state", state_options, index=sp_index)
        customer_state = route5.selectbox("Customer state", state_options, index=sp_index)
        category = route6.selectbox("Product category", PRODUCT_CATEGORIES)

        st.subheader("Shipping and product")
        ship1, ship2, ship3 = st.columns(3)
        price = ship1.number_input("Order price (R$)", min_value=0.0, value=120.0, step=10.0)
        freight = ship2.number_input("Freight cost (R$)", min_value=0.0, value=20.0, step=5.0)
        weight = ship3.number_input("Product weight (g)", min_value=0.0, value=1200.0, step=100.0)
        product1, product2, product3 = st.columns(3)
        length = product1.number_input("Length (cm)", min_value=0.0, value=25.0, step=1.0)
        height = product2.number_input("Height (cm)", min_value=0.0, value=15.0, step=1.0)
        width = product3.number_input("Width (cm)", min_value=0.0, value=20.0, step=1.0)

        with st.expander("Route coordinates (Brazil)"):
            st.caption("Coordinates are used to estimate the route's great-circle distance.")
            coord1, coord2 = st.columns(2)
            seller_lat = coord1.number_input(
                "Seller latitude", min_value=-34.0, max_value=6.0, value=-23.55, format="%.5f"
            )
            seller_lng = coord1.number_input(
                "Seller longitude", min_value=-74.0, max_value=-34.0, value=-46.63, format="%.5f"
            )
            customer_lat = coord2.number_input(
                "Customer latitude", min_value=-34.0, max_value=6.0, value=-22.91, format="%.5f"
            )
            customer_lng = coord2.number_input(
                "Customer longitude", min_value=-74.0, max_value=-34.0, value=-43.20, format="%.5f"
            )
        submitted = st.form_submit_button("Estimate delivery time", type="primary")

    if submitted:
        values = {
            "purchase_date": purchase_date,
            "same_city": same_city,
            "seller_state": seller_state,
            "customer_state": customer_state,
            "category": category,
            "price": price,
            "freight": freight,
            "weight": weight,
            "product_length": length,
            "product_height": height,
            "product_width": width,
            "seller_lat": seller_lat,
            "seller_lng": seller_lng,
            "customer_lat": customer_lat,
            "customer_lng": customer_lng,
        }
        prediction = float(bundle["model"].predict(make_regression_features(values, bundle))[0])
        st.success(f"Predicted delivery duration: **{prediction:.1f} days**")
        st.caption("Prediction target: actual days from purchase to customer delivery.")
