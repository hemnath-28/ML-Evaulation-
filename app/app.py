"""Shared Streamlit entry point and navigation for all three model tracks."""

import streamlit as st

import classification_app
import clustering_app
import regression_app


def main() -> None:
    st.set_page_config(page_title="ML Model Studio", page_icon="📊", layout="wide")
    st.title("Machine Learning Model Studio")
    st.caption("Choose a model track to make a prediction or explore a cluster.")

    track = st.radio(
        "Choose a model",
        ["Regression", "Clustering", "Classification"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    if track == "Regression":
        regression_app.render()
    elif track == "Clustering":
        clustering_app.render()
    else:
        classification_app.render()


if __name__ == "__main__":
    main()
