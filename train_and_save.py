"""Train the autism detection pipeline and save artifacts for Flask / Streamlit."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = ["gender", "ethnicity", "jundice", "austim", "contry_of_res", "relation"]
SCORE_FEATURES = [f"A{i}_Score" for i in range(1, 11)]
RAW_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + SCORE_FEATURES


def load_training_data(csv_path: str = "autism_data.csv") -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path).dropna()
    data = data[(data["age"] >= 18) & (data["age"] <= 100)]
    y = data["Class/ASD"].apply(lambda x: 1 if x == "YES" else 0)
    X = data[RAW_FEATURE_COLUMNS]
    return X, y


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("scores", "passthrough", SCORE_FEATURES),
        ],
        remainder="drop",
    )
    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=1,
        eval_metric="logloss",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    X, y = load_training_data()
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    joblib.dump(pipeline, "autism_pipeline.pkl")
    joblib.dump(pipeline.named_steps["preprocessor"], "preprocessor.pkl")
    joblib.dump(pipeline.named_steps["classifier"], "model.pkl")
    print("Saved autism_pipeline.pkl, preprocessor.pkl, model.pkl")


if __name__ == "__main__":
    main()
