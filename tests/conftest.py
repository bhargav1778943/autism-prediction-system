import pytest
import sys
import os

# Add the project directory to the python path so we can import app and train_and_save
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app import app, RAW_FEATURE_COLUMNS

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def valid_payload():
    return {
        "age": 30.0,
        "gender": "m",
        "ethnicity": "White-European",
        "jundice": "no",
        "austim": "yes",
        "contry_of_res": "United Kingdom",
        "relation": "Self",
        "A1_Score": 1,
        "A2_Score": 1,
        "A3_Score": 0,
        "A4_Score": 1,
        "A5_Score": 0,
        "A6_Score": 0,
        "A7_Score": 1,
        "A8_Score": 1,
        "A9_Score": 0,
        "A10_Score": 1
    }

@pytest.fixture
def child_payload(valid_payload):
    payload = valid_payload.copy()
    payload["age"] = 10.0  # Under 18
    return payload

@pytest.fixture
def missing_fields_payload():
    # Missing relation and A10_Score
    return {
        "age": 25.0,
        "gender": "f",
        "ethnicity": "Latino",
        "jundice": "no",
        "austim": "no",
        "contry_of_res": "Brazil",
        "A1_Score": 0,
        "A2_Score": 1,
        "A3_Score": 1,
        "A4_Score": 0,
        "A5_Score": 1,
        "A6_Score": 0,
        "A7_Score": 0,
        "A8_Score": 1,
        "A9_Score": 1
    }

@pytest.fixture
def invalid_categorical_payload(valid_payload):
    payload = valid_payload.copy()
    payload["jundice"] = "maybe"  # Invalid yes/no
    payload["gender"] = "unknown"  # Invalid gender
    return payload
