import streamlit as st
import requests

API_URL = "https://autism-prediction-system-r3yx.onrender.com/predict"

st.set_page_config(page_title="Autism Prediction System")

st.title("🧠 Autism Prediction System")

age = st.number_input("Age", min_value=18, max_value=100, value=26)
gender = st.selectbox("Gender", ["m", "f"])

scores = {}
for i in range(1, 11):
    scores[f"A{i}_Score"] = st.selectbox(
        f"A{i}_Score",
        [0, 1],
        key=f"a{i}"
    )

if st.button("Predict"):
    payload = {
        "age": age,
        "gender": gender,
        "ethnicity": "White-European",
        "jundice": "no",
        "austim": "no",
        "contry_of_res": "India",
        "relation": "Self",
        **scores
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(result["prediction"])
            st.write("Probability:", result["probability_percent"])
        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Request failed: {e}")
