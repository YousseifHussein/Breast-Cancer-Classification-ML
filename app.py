import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("breast_cancer_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Breast Cancer Prediction App")

st.write("Enter the tumor measurements below:")

mean_radius = st.number_input("Mean Radius", value=14.0)
mean_perimeter = st.number_input("Mean Perimeter", value=90.0)
mean_area = st.number_input("Mean Area", value=600.0)
mean_concavity = st.number_input("Mean Concavity", value=0.05)
mean_concave_points = st.number_input("Mean Concave Points", value=0.03)

worst_radius = st.number_input("Worst Radius", value=16.0)
worst_perimeter = st.number_input("Worst Perimeter", value=110.0)
worst_area = st.number_input("Worst Area", value=800.0)
worst_concave_points = st.number_input("Worst Concave Points", value=0.10)

tumor_size = st.selectbox(
    "Tumor Size",
    [0, 1, 2]
)

if st.button("Predict"):

    input_data = pd.DataFrame([[
        mean_radius,
        mean_perimeter,
        mean_area,
        mean_concavity,
        mean_concave_points,
        worst_radius,
        worst_perimeter,
        worst_area,
        worst_concave_points,
        tumor_size
    ]], columns=[
        'mean radius',
        'mean perimeter',
        'mean area',
        'mean concavity',
        'mean concave points',
        'worst radius',
        'worst perimeter',
        'worst area',
        'worst concave points',
        'tumor_size'
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.error("Prediction: Malignant Tumor")
    else:
        st.success("Prediction: Benign Tumor")