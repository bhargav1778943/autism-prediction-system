import pytest
import json

def test_api_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data["service"] == "Autism Detection API"
    assert "clinical_disclaimer" in data
    assert "This tool is for screening purposes only" in data["clinical_disclaimer"]
    assert "restrictions" in data
    assert "age >= 18" in data["restrictions"]["age"]
    
    # Ensure 'result' is NOT in required fields list
    assert "result" not in data["required_fields"]
    assert "age" in data["required_fields"]
    assert "gender" in data["required_fields"]

def test_api_predict_success(client, valid_payload):
    response = client.post(
        "/predict",
        data=json.dumps(valid_payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert "prediction" in data
    assert "prediction_code" in data
    assert data["prediction_code"] in [0, 1]
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "probability_percent" in data

def test_api_predict_missing_body(client):
    response = client.post(
        "/predict",
        data="",
        content_type="application/json"
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert "error" in data
    assert "Request body must be valid JSON" in data["error"]

def test_api_predict_missing_required_fields(client, missing_fields_payload):
    response = client.post(
        "/predict",
        data=json.dumps(missing_fields_payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert "error" in data
    assert "Missing required fields" in data["error"]
    assert "relation" in data["error"]
    assert "A10_Score" in data["error"]

def test_api_predict_age_restriction(client, child_payload):
    response = client.post(
        "/predict",
        data=json.dumps(child_payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert "error" in data
    assert "Prediction restricted to adults only" in data["error"]

def test_api_predict_invalid_age_type(client, valid_payload):
    payload = valid_payload.copy()
    payload["age"] = "not_a_number"
    
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert "error" in data
    assert "Age must be a valid number" in data["error"]
