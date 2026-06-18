"""Streamlit frontend for the Autism Prediction System."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

RAW_FEATURE_COLUMNS = [
    "age",
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "relation",
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
]

CATEGORICAL_OPTIONS = {
    "gender": ["m", "f"],
    "ethnicity": [
        "White-European",
        "Latino",
        "Asian",
        "Middle Eastern ",
        "Pasifika",
        "Black",
        "Others",
        "South Asian",
        "?",
    ],
    "jundice": ["yes", "no"],
    "austim": ["yes", "no"],
    "contry_of_res": [
        "United States",
        "Brazil",
        "Spain",
        "Egypt",
        "India",
        "United Kingdom",
        "Jordan",
        "Netherlands",
        "Azerbaijan",
        "Canada",
        "Australia",
        "New Zealand",
        "Afghanistan",
        "France",
        "Malaysia",
        "Germany",
        "Ireland",
        "Italy",
        "Sweden",
        "China",
        "Bahamas",
        "Armenia",
        "United Arab Emirates",
        "Argentina",
        "Pakistan",
        "Saudi Arabia",
        "Qatar",
        "Turkey",
        "Oman",
        "Russia",
        "South Africa",
        "Iraq",
        "Nigeria",
        "Vietnam",
        "Iran",
        "Tunisia",
        "Lebanon",
        "Mexico",
        "Romania",
        "Belgium",
        "Poland",
        "Kuwait",
        "Indonesia",
        "Philippines",
        "Sri Lanka",
        "Bangladesh",
        "Serbia",
        "Norway",
        "Portugal",
        "Finland",
        "Albania",
        "Greece",
        "Cyprus",
        "Japan",
        "Bahrain",
        "Morocco",
        "Hungary",
        "Austria",
        "Ethiopia",
        "Nepal",
        "Ukraine",
        "Syria",
        "Korea",
        "Denmark",
        "Croatia",
        "Singapore",
        "Thailand",
        "Iceland",
        "Malta",
        "Myanmar",
        "Kenya",
        "Tanzania",
        "Algeria",
        "Costa Rica",
        "Palestine",
        "Yemen",
        "Ghana",
        "Chile",
        "Colombia",
        "Peru",
        "Ecuador",
        "Sudan",
        "Libya",
        "Uruguay",
        "Bolivia",
        "Czech Republic",
        "Slovakia",
        "Slovenia",
        "Lithuania",
        "Latvia",
        "Estonia",
        "Luxembourg",
        "Monaco",
        "Andorra",
        "Liechtenstein",
        "San Marino",
        "Vatican City",
        "Other",
    ],
    "relation": ["Self", "Parent", "Health care professional", "Relative", "Others", "?"],
}


@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("autism_pipeline.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Run the notebook pipeline save cell or `python train_and_save.py` first.")
        st.stop()


def build_input_form() -> dict:
    st.subheader("Patient / Screening Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=26.0, step=1.0)
        gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
        ethnicity = st.selectbox("Ethnicity", CATEGORICAL_OPTIONS["ethnicity"])
        jundice = st.selectbox("Jaundice at birth", CATEGORICAL_OPTIONS["jundice"])
        austim = st.selectbox("Family history of autism", CATEGORICAL_OPTIONS["austim"])
        country = st.selectbox("Country of residence", CATEGORICAL_OPTIONS["contry_of_res"])
        relation = st.selectbox("Relation to respondent", CATEGORICAL_OPTIONS["relation"])

    with col2:
        st.markdown("**A1–A10 Screening Scores** (0 = No, 1 = Yes)")
        scores = {}
        for i in range(1, 11):
            scores[f"A{i}_Score"] = st.selectbox(
                f"A{i}_Score",
                options=[0, 1],
                key=f"a{i}",
            )

    return {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jundice": jundice,
        "austim": austim,
        "contry_of_res": country,
        "relation": relation,
        **scores,
    }


def render_shap_explanation(pipeline, input_df: pd.DataFrame):
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(input_df)

    model_name = type(classifier).__name__
    tree_models = ("XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier")

    if any(name in model_name for name in tree_models):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        background = shap.sample(transformed, min(50, transformed.shape[0]), random_state=1)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_values = explainer.shap_values(transformed, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = explainer.expected_value[1]

    feature_names = preprocessor.get_feature_names_out()
    mean_abs = np.abs(shap_values[0])
    top_idx = np.argsort(mean_abs)[::-1][:10]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [feature_names[i] for i in top_idx[::-1]],
        [mean_abs[i] for i in top_idx[::-1]],
        color="#4e6e8e",
    )
    ax.set_xlabel("SHAP value magnitude")
    ax.set_title("Top 10 Influential Features for This Prediction")
    st.pyplot(fig)
    plt.close(fig)

    st.caption("Force plot for this individual")
    shap.force_plot(
        base_value,
        shap_values[0],
        transformed[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    force_fig = plt.gcf()
    st.pyplot(force_fig)
    plt.close(force_fig)


def main():
    st.set_page_config(page_title="Autism Prediction System", page_icon="🧠", layout="wide")
    st.title("Autism Spectrum Disorder Prediction System")
    st.warning("⚠️ **Clinical Disclaimer:** This tool is for screening purposes only and does not replace professional medical diagnosis.")
    st.markdown(
        "Enter screening questionnaire responses and demographic data to receive "
        "an ML-based ASD likelihood estimate with SHAP explanation."
    )

    pipeline = load_pipeline()
    user_input = build_input_form()
    
    # Restrict predictions to adults only
    if user_input["age"] < 18.0:
        st.error("Prediction is restricted to adults only (age must be >= 18).")
        st.stop()

    input_df = pd.DataFrame([user_input])

    if st.button("Predict", type="primary"):
        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])
        label = "YES — ASD likely" if prediction == 1 else "NO — ASD unlikely"

        st.success(f"**Prediction:** {label}")
        st.info(f"**Probability (ASD):** {probability:.2%}")

        with st.expander("SHAP Explanation", expanded=True):
            try:
                render_shap_explanation(pipeline, input_df)
            except Exception as exc:
                st.warning(f"SHAP explanation unavailable: {exc}")


if __name__ == "__main__":
    main()
