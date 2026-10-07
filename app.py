
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("diabetes_model.pkl")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient information to predict the diabetes category.")
st.divider()

# Binary features
binary_features = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
    'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
]

inputs = {}

for feature in binary_features:
    inputs[feature] = st.selectbox(
        feature,
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# Other features
inputs['BMI'] = st.number_input("BMI", 10.0, 100.0, 25.0)
inputs['GenHlth'] = st.selectbox("General Health", [1, 2, 3, 4, 5])
inputs['MentHlth'] = st.number_input("Mental Health Days", 0, 30, 0)
inputs['PhysHlth'] = st.number_input("Physical Health Days", 0, 30, 0)
inputs['Sex'] = st.selectbox("Sex", [0, 1])
inputs['Age'] = st.selectbox("Age Category", range(1, 14))
inputs['Education'] = st.selectbox("Education", range(1, 7))
inputs['Income'] = st.selectbox("Income", range(1, 9))

# Exact feature order used during training
feature_order = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex',
    'Age', 'Education', 'Income'
]

if st.button("🔍 Predict Diabetes"):

    data = pd.DataFrame([inputs])[feature_order]

    prediction = model.predict(data)[0]

    result = {
        0: "No Diabetes",
        1: "Prediabetes",
        2: "Diabetes"
    }

    st.success(f"### Prediction: {result[prediction]}")

st.caption(
    "Academic project only — this prediction should not be used as a medical diagnosis."
)
