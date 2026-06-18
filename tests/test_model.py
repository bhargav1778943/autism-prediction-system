import pytest
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "autism_pipeline.pkl"))

def test_model_loading_and_prediction(valid_payload):
    # Test 1: Load pipeline
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    assert pipeline is not None
    
    # Test 2: Predict on valid payload
    input_df = pd.DataFrame([valid_payload])
    preds = pipeline.predict(input_df)
    probas = pipeline.predict_proba(input_df)
    
    assert len(preds) == 1
    assert preds[0] in [0, 1]
    assert probas.shape == (1, 2)
    assert 0.0 <= probas[0][0] <= 1.0
    assert 0.0 <= probas[0][1] <= 1.0
    assert np.isclose(probas[0].sum(), 1.0)

def test_preprocessing_pipeline_structure(valid_payload):
    # Test 3: Load preprocessor
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Check numeric columns
    numeric_transformer = preprocessor.named_transformers_["num"]
    assert "age" in numeric_transformer.feature_names_in_
    assert "result" not in numeric_transformer.feature_names_in_
    
    # Check MinMaxScaler is working (fitted range check)
    assert hasattr(numeric_transformer, "data_min_")
    assert numeric_transformer.data_min_[0] == 18.0
    assert numeric_transformer.data_max_[0] == 64.0
    
    # Check Categorical columns
    cat_transformer = preprocessor.named_transformers_["cat"]
    assert "gender" in cat_transformer.feature_names_in_
    assert "ethnicity" in cat_transformer.feature_names_in_
    assert "jundice" in cat_transformer.feature_names_in_
    assert "austim" in cat_transformer.feature_names_in_
    assert "contry_of_res" in cat_transformer.feature_names_in_
    assert "relation" in cat_transformer.feature_names_in_

def test_one_hot_encoder_ignores_unknown_category(valid_payload):
    # Test 4: One-hot encoder handles unknown values correctly without raising errors
    pipeline = joblib.load(MODEL_PATH)
    
    payload = valid_payload.copy()
    payload["ethnicity"] = "Martian"  # Unknown ethnicity category
    payload["contry_of_res"] = "Atlantis"  # Unknown country category
    
    input_df = pd.DataFrame([payload])
    
    # Should transform and predict successfully (unknown categories ignored / all zeros)
    preds = pipeline.predict(input_df)
    assert len(preds) == 1

def test_train_and_save_functions():
    # Test 5: Verify standalone training data loading and pipeline creation
    from train_and_save import load_training_data, build_pipeline
    X, y = load_training_data()
    
    assert X is not None
    assert y is not None
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert "result" not in X.columns
    assert (X["age"] >= 18).all()
    assert (X["age"] <= 100).all()
    
    pipeline = build_pipeline()
    assert pipeline is not None

    # Test 6: Verify full training and serialization execution
    from train_and_save import main
    main()


