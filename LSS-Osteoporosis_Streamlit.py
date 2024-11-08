import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
OPERATING_THRESHOLD = 0.30

FEATURES = [
    {
        "name": "Ca",
        "label": "Calcium (mmol/L)",
        "min": 0.45,
        "max": 2.90,
        "default": 2.390,
        "step": 0.001,
    },
    {
        "name": "Hematocrit",
        "label": "Hematocrit (%)",
        "min": 0.34,
        "max": 57.10,
        "default": 41.500,
        "step": 0.001,
    },
    {
        "name": "eGFR",
        "label": "eGFR (mL/min/1.73 m^2)",
        "min": 21.47,
        "max": 152.42,
        "default": 102.730,
        "step": 0.001,
    },
    {
        "name": "ALT",
        "label": "ALT (U/L)",
        "min": 3.20,
        "max": 316.38,
        "default": 35.600,
        "step": 0.001,
    },
    {
        "name": "Na",
        "label": "Sodium (mmol/L)",
        "min": 109.00,
        "max": 150.10,
        "default": 143.600,
        "step": 0.001,
    },
    {
        "name": "Creatinine",
        "label": "Creatinine (umol/L)",
        "min": 18.00,
        "max": 202.40,
        "default": 49.000,
        "step": 0.001,
    },
    {
        "name": "LDH",
        "label": "LDH (U/L)",
        "min": 102.73,
        "max": 634.40,
        "default": 216.000,
        "step": 0.001,
    },
    {
        "name": "HB",
        "label": "Hemoglobin (g/L)",
        "min": 41.00,
        "max": 193.00,
        "default": 136.000,
        "step": 0.001,
    },
]

feature_names = [feature["name"] for feature in FEATURES]

st.set_page_config(
    page_title="Osteoporosis Risk Predictor",
    layout="wide",
)

# Load the logistic regression model
try:
    model = joblib.load(APP_DIR / "LR_Tuned_LSS.pkl")
except FileNotFoundError:
    st.error("Model file 'LR_Tuned_LSS.pkl' not found. Please ensure it is in the correct directory.")
    st.stop()

# Load the explainer object from the file
try:
    with (APP_DIR / "explainer_LSS.pkl").open("rb") as f:
        explainer = pickle.load(f)
except FileNotFoundError:
    st.error("Explainer file 'explainer_LSS.pkl' not found. Please ensure it is in the correct directory.")
    st.stop()

# Streamlit user interface
st.title("Osteoporosis Risk Predictor for Lumbar Spinal Stenosis Patients")

# Create three columns
col1, col2, col3 = st.columns([1, 1, 1])

# Left column: Input section
with col1:
    st.header("Indicators")
    inputs = {}
    for feature in FEATURES:
        inputs[feature["name"]] = st.number_input(
            feature["label"],
            min_value=float(feature["min"]),
            max_value=float(feature["max"]),
            value=float(feature["default"]),
            step=float(feature["step"]),
            format="%.3f",
        )

# Process inputs and make predictions
feature_values = [inputs[feature] for feature in feature_names]
features_df = pd.DataFrame([feature_values], columns=feature_names)
if st.button("Predict"):
    predicted_proba = model.predict_proba(features_df)[0]
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    osteoporosis_probability = float(predicted_proba[positive_index])
    above_threshold = osteoporosis_probability >= OPERATING_THRESHOLD

    st.metric(
        "Model-estimated osteoporosis probability",
        f"{osteoporosis_probability * 100:.1f}%",
    )
    threshold_result = "Above" if above_threshold else "Below"
    st.write(
        f"**Screening result:** {threshold_result} the prespecified operating threshold "
        f"({OPERATING_THRESHOLD:.2f})"
    )
    st.caption(
        "Research prototype only. This model output does not establish a diagnosis "
        "and requires clinical evaluation."
    )

    shap_values = explainer.shap_values(features_df)
    combined_shap_values = np.vstack((shap_values, -shap_values))
    combined_shap_expected_value = [explainer.expected_value, -explainer.expected_value]
    class_name = ["Osteoporosis Class", "Normal Class"]
    # Middle and right columns: Explanation for both classes
    for which_class, col in enumerate([col2, col3]):
        with col:
            st.header(f"{class_name[which_class]} Explanation")
            st.subheader(f"SHAP Force Plot")
            fig = plt.figure()
            shap.force_plot(
                base_value=combined_shap_expected_value[which_class],
                shap_values=combined_shap_values[which_class],
                features=features_df,
                feature_names=feature_names,
                matplotlib=True,
                text_rotation=30
            )
            fig.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

            st.subheader(f"SHAP Waterfall Plot")
            fig = plt.figure()
            shap.waterfall_plot(
                shap.Explanation(base_values=combined_shap_expected_value[which_class],
                                 values=combined_shap_values[which_class],
                                 data=feature_values,
                                 feature_names=feature_names)
            )
            fig.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
