import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model/breast_cancer_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

st.set_page_config(page_title="Breast Cancer Prediction System")

st.title("🧬 Breast Cancer Prediction System")
st.write(
    "This system predicts whether a tumor is **Benign** or **Malignant** "
    "based on selected features. For educational purposes only."
)

# Feature inputs
radius = st.number_input("Radius Mean", min_value=0.0, step=0.1)
texture = st.number_input("Texture Mean", min_value=0.0, step=0.1)
perimeter = st.number_input("Perimeter Mean", min_value=0.0, step=0.1)
area = st.number_input("Area Mean", min_value=0.0, step=1.0)
concavity = st.number_input("Concavity Mean", min_value=0.0, step=0.01)

if st.button("Predict Diagnosis"):
    input_df = pd.DataFrame(
        [[radius, texture, perimeter, area, concavity]],
        columns=[
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean concavity"
        ]
    )

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    result = "Benign" if prediction == 1 else "Malignant"
    st.success(f"Prediction Result: **{result}**")
