import streamlit as st
import requests

API_URL = "https://autism-prediction-system-r3yx.onrender.com/predict"

st.title("🧠 Autism Prediction System")

age = st.number_input("Age", min_value=18, value=26)
gender = st.selectbox("Gender", ["m", "f"])

a1 = st.selectbox("A1 Score", [0, 1])
a2 = st.selectbox("A2 Score", [0, 1])
a3 = st.selectbox("A3 Score", [0, 1])
a4 = st.selectbox("A4 Score", [0, 1])
a5 = st.selectbox("A5 Score", [0, 1])
a6 = st.selectbox("A6 Score", [0, 1])
a7 = st.selectbox("A7 Score", [0, 1])
a8 = st.selectbox("A8 Score", [0, 1])
a9 = st.selectbox("A9 Score", [0, 1])
a10 = st.selectbox("A10 Score", [0, 1])

if st.button("Predict"):
    payload = {
        "age": age,
        "gender": gender,
        "ethnicity": "White-European",
        "jundice": "no",
        "austim": "no",
        "contry_of_res": "India",
        "relation": "Self",
        "A1_Score": a1,
        "A2_Score": a2,
        "A3_Score": a3,
        "A4_Score": a4,
        "A5_Score": a5,
        "A6_Score": a6,
        "A7_Score": a7,
        "A8_Score": a8,
        "A9_Score": a9,
        "A10_Score": a10
    }

    r = requests.post(API_URL, json=payload)

    if r.status_code == 200:
        st.success(r.json()["prediction"])
        st.write("Probability:", r.json()["probability_percent"])
    else:
        st.error(r.text)