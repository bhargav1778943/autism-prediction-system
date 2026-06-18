import streamlit as st
import requests

API_URL = "https://autism-prediction-system-r3yx.onrender.com/predict"

st.set_page_config(
    page_title="Autism Prediction System",
    page_icon="🧠"
)

st.title("🧠 Autism Prediction System")

st.markdown("""
This tool uses a Machine Learning model to estimate the likelihood of Autism Spectrum Disorder (ASD).

**Note:** This is a screening tool only and not a medical diagnosis.
""")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=26
)

gender = st.selectbox(
    "Gender",
    ["m", "f"]
)

questions = {
    "A1_Score": "I often notice small sounds when others do not",
    "A2_Score": "I usually concentrate more on the whole picture than the small details",
    "A3_Score": "I find it easy to do more than one thing at once",
    "A4_Score": "If there is an interruption, I can switch back quickly to what I was doing",
    "A5_Score": "I find it easy to read between the lines when someone is talking to me",
    "A6_Score": "I know how to tell if someone listening to me is getting bored",
    "A7_Score": "When reading a story, I find it difficult to work out the characters' intentions",
    "A8_Score": "I like collecting information about categories of things",
    "A9_Score": "I find it easy to work out what someone is thinking or feeling from their face",
    "A10_Score": "I find it difficult to work out people's intentions"
}

st.subheader("AQ-10 Autism Screening Questions")
st.caption("Select Yes or No for each question.")

scores = {}

for key, question in questions.items():
    answer = st.radio(
        question,
        ["No", "Yes"],
        key=key,
        horizontal=True
    )

    scores[key] = 1 if answer == "Yes" else 0

if st.button("Predict", type="primary"):

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

            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Probability: {result['probability_percent']}")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Request failed: {e}")