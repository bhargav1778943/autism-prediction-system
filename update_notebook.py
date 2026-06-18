"""Append and patch autism_detection.ipynb with Prompts 1-6 enhancements."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "autism_detection.ipynb"


def cell_md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- Prompt 1: XGBoost (insert before Model Tuning) ---
    tuning_idx = next(
        i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "Model Tuning" in "".join(c.get("source", []))
    )

    xgb_cells = [
        cell_md("### 7. XGBoost"),
        cell_code(
            "from xgboost import XGBClassifier\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n"
            ")\n"
            "\n"
            "xgb_model = XGBClassifier(\n"
            "    n_estimators=100,\n"
            "    max_depth=4,\n"
            "    learning_rate=0.1,\n"
            "    random_state=1,\n"
            "    eval_metric='logloss',\n"
            "    use_label_encoder=False,\n"
            ")\n"
            "xgb_model.fit(X_train.values, y_train)\n"
            "xgb_pred = xgb_model.predict(X_test.values)\n"
            "xgb_proba = xgb_model.predict_proba(X_test.values)[:, 1]\n"
            "\n"
            "print('XGBoost — Accuracy:  {:.4f}'.format(accuracy_score(y_test, xgb_pred)))\n"
            "print('XGBoost — Precision: {:.4f}'.format(precision_score(y_test, xgb_pred)))\n"
            "print('XGBoost — Recall:    {:.4f}'.format(recall_score(y_test, xgb_pred)))\n"
            "print('XGBoost — F1 Score:  {:.4f}'.format(f1_score(y_test, xgb_pred)))\n"
            "print('XGBoost — ROC-AUC:   {:.4f}'.format(roc_auc_score(y_test, xgb_proba)))"
        ),
        cell_md("### Model Comparison (All Classifiers)"),
        cell_code(
            "def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):\n"
            "    model.fit(X_tr, y_tr)\n"
            "    preds = model.predict(X_te)\n"
            "    probas = (\n"
            "        model.predict_proba(X_te)[:, 1]\n"
            "        if hasattr(model, 'predict_proba')\n"
            "        else model.decision_function(X_te)\n"
            "    )\n"
            "    return {\n"
            "        'Model': name,\n"
            "        'Accuracy': accuracy_score(y_te, preds),\n"
            "        'Precision': precision_score(y_te, preds, zero_division=0),\n"
            "        'Recall': recall_score(y_te, preds, zero_division=0),\n"
            "        'F1 Score': f1_score(y_te, preds, zero_division=0),\n"
            "        'ROC-AUC': roc_auc_score(y_te, probas),\n"
            "    }\n"
            "\n"
            "comparison_models = {\n"
            "    'Decision Tree': DecisionTreeClassifier(random_state=1),\n"
            "    'Random Forest': RandomForestClassifier(n_estimators=5, random_state=1),\n"
            "    'SVM': svm.SVC(kernel='linear', C=1, gamma=2, probability=True, random_state=1),\n"
            "    'KNN': neighbors.KNeighborsClassifier(n_neighbors=10),\n"
            "    'Naive Bayes': MultinomialNB(),\n"
            "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=1),\n"
            "    'XGBoost': XGBClassifier(\n"
            "        n_estimators=100, max_depth=4, learning_rate=0.1,\n"
            "        random_state=1, eval_metric='logloss', use_label_encoder=False,\n"
            "    ),\n"
            "}\n"
            "\n"
            "comparison_results = [\n"
            "    evaluate_model(name, model, X_train.values, y_train, X_test.values, y_test)\n"
            "    for name, model in comparison_models.items()\n"
            "]\n"
            "comparison_df = pd.DataFrame(comparison_results).sort_values('F1 Score', ascending=False)\n"
            "comparison_df.reset_index(drop=True, inplace=True)\n"
            "display(comparison_df)\n"
            "best_model_name = comparison_df.iloc[0]['Model']\n"
            "print('Best model by F1 Score:', best_model_name)"
        ),
    ]
    cells[tuning_idx:tuning_idx] = xgb_cells

    # --- Prompt 4: Improve SVM GridSearchCV (patch existing tuning cells) ---
    tuning_idx = next(
        i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "Model Tuning" in "".join(c.get("source", []))
    )
    # cells shifted by len(xgb_cells)
    param_cell = tuning_idx + 2
    grid_cell = tuning_idx + 3
    eval_cell = tuning_idx + 5

    cells[param_cell]["source"] = [
        "from sklearn.model_selection import StratifiedKFold\n"
        "\n"
        "def f_beta_score(y_true, y_predict):\n"
        "    return fbeta_score(y_true, y_predict, beta=0.5)\n"
        "\n"
        "clf = SVC(random_state=1, probability=True)\n"
        "parameters = {\n"
        "    'C': [0.1, 1, 10, 100],\n"
        "    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],\n"
        "    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],\n"
        "    'degree': [2, 3, 4, 5],\n"
        "}\n"
        "scorer = make_scorer(f_beta_score)\n"
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)"
    ]

    cells[grid_cell]["source"] = [
        "grid_obj = GridSearchCV(\n"
        "    estimator=clf,\n"
        "    param_grid=parameters,\n"
        "    scoring=scorer,\n"
        "    cv=skf,\n"
        "    n_jobs=-1,\n"
        "    refit=True,\n"
        ")\n"
        "grid_fit = grid_obj.fit(X_train.values, y_train)\n"
        "best_clf = grid_fit.best_estimator_\n"
        "\n"
        "print('Best Parameters:', grid_fit.best_params_)\n"
        "print('Best Cross Validation Score: {:.4f}'.format(grid_fit.best_score_))"
    ]

    cells[eval_cell]["source"] = [
        "print('Unoptimized model\\n------')\n"
        "print('Accuracy score on testing data: {:.4f}'.format(accuracy_score(y_test, predictions)))\n"
        "print('F-score on testing data: {:.4f}'.format(fbeta_score(y_test, predictions, beta=0.5)))\n"
        "print('\\nOptimized Model (GridSearchCV + StratifiedKFold)\\n------')\n"
        "print('Best Parameters:', grid_fit.best_params_)\n"
        "print('Best Cross Validation Score: {:.4f}'.format(grid_fit.best_score_))\n"
        "print('Final Test Accuracy: {:.4f}'.format(accuracy_score(y_test, best_predictions)))\n"
        "print('Final F-score on testing data: {:.4f}'.format(fbeta_score(y_test, best_predictions, beta=0.5)))"
    ]

    # Remove trailing empty cell if present
    if cells and cells[-1]["cell_type"] == "code" and not "".join(cells[-1].get("source", [])).strip():
        cells.pop()

    # --- Prompt 2: SMOTE ---
    smote_cells = [
        cell_md("## Class Imbalance Analysis & SMOTE"),
        cell_code(
            "from collections import Counter\n"
            "from imblearn.over_sampling import SMOTE\n"
            "\n"
            "print('Class distribution (full dataset):')\n"
            "print(Counter(data_classes))\n"
            "print('\\nClass distribution (training set):')\n"
            "print(Counter(y_train))\n"
            "\n"
            "class_counts = Counter(y_train)\n"
            "imbalance_ratio = class_counts[0] / class_counts[1]\n"
            "print('\\nImbalance ratio (majority/minority): {:.2f}'.format(imbalance_ratio))\n"
            "has_imbalance = imbalance_ratio > 1.2 or imbalance_ratio < 0.8\n"
            "print('Class imbalance detected:' if has_imbalance else 'Classes are balanced:', has_imbalance)"
        ),
        cell_code(
            "smote = SMOTE(random_state=1)\n"
            "X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)\n"
            "\n"
            "print('Training set BEFORE SMOTE:', Counter(y_train))\n"
            "print('Training set AFTER SMOTE: ', Counter(y_train_smote))\n"
            "print('Test set (unchanged — no leakage):', Counter(y_test))"
        ),
        cell_code(
            "def smote_comparison(model_name, model):\n"
            "    before = evaluate_model(\n"
            "        model_name + ' (before SMOTE)', model,\n"
            "        X_train.values, y_train, X_test.values, y_test,\n"
            "    )\n"
            "    after = evaluate_model(\n"
            "        model_name + ' (after SMOTE)', model,\n"
            "        X_train_smote.values, y_train_smote, X_test.values, y_test,\n"
            "    )\n"
            "    return before, after\n"
            "\n"
            "smote_model = XGBClassifier(\n"
            "    n_estimators=100, max_depth=4, learning_rate=0.1,\n"
            "    random_state=1, eval_metric='logloss', use_label_encoder=False,\n"
            ")\n"
            "before_smote, after_smote = smote_comparison('XGBoost', smote_model)\n"
            "smote_comparison_df = pd.DataFrame([before_smote, after_smote])\n"
            "display(smote_comparison_df)"
        ),
    ]

    # --- Prompt 3: SHAP ---
    shap_cells = [
        cell_md("## Model Explainability with SHAP"),
        cell_code(
            "import shap\n"
            "\n"
            "best_comparison_model = comparison_models[best_model_name]\n"
            "best_comparison_model.fit(X_train.values, y_train)\n"
            "feature_names = list(features_final.columns)\n"
            "\n"
            "tree_models = ('Decision Tree', 'Random Forest', 'XGBoost')\n"
            "if best_model_name in tree_models:\n"
            "    explainer = shap.TreeExplainer(best_comparison_model)\n"
            "    shap_values = explainer.shap_values(X_test.values)\n"
            "    X_shap_display = X_test.values\n"
            "else:\n"
            "    background = shap.sample(X_train.values, 100, random_state=1)\n"
            "    explainer = shap.KernelExplainer(best_comparison_model.predict_proba, background)\n"
            "    X_shap_display = X_test.values[:50]\n"
            "    shap_values = explainer.shap_values(X_shap_display, nsamples=100)\n"
            "\n"
            "if isinstance(shap_values, list):\n"
            "    shap_values = shap_values[1]"
        ),
        cell_code(
            "shap.summary_plot(shap_values, X_shap_display, feature_names=feature_names, show=False)\n"
            "plt.title('SHAP Summary Plot — ' + best_model_name)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        cell_code(
            "mean_abs_shap = np.abs(shap_values).mean(axis=0)\n"
            "importance_df = pd.DataFrame({\n"
            "    'Feature': feature_names,\n"
            "    'Mean |SHAP|': mean_abs_shap,\n"
            "}).sort_values('Mean |SHAP|', ascending=False)\n"
            "\n"
            "top10 = importance_df.head(10)\n"
            "plt.figure(figsize=(10, 6))\n"
            "plt.barh(top10['Feature'][::-1], top10['Mean |SHAP|'][::-1], color='#4e6e8e')\n"
            "plt.xlabel('Mean |SHAP value|')\n"
            "plt.title('Top 10 Feature Importance (SHAP) — ' + best_model_name)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "display(top10.reset_index(drop=True))"
        ),
        cell_code(
            "sample_idx = 0\n"
            "shap.force_plot(\n"
            "    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,\n"
            "    shap_values[sample_idx],\n"
            "    X_shap_display[sample_idx],\n"
            "    feature_names=feature_names,\n"
            "    matplotlib=True,\n"
            "    show=False,\n"
            ")\n"
            "plt.title('SHAP Force Plot — Sample {} ({})'.format(sample_idx, best_model_name))\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print('\\nTop 10 Most Influential Features:')\n"
            "for i, row in top10.iterrows():\n"
            "    direction = 'increases' if shap_values[:, feature_names.index(row['Feature'])].mean() > 0 else 'decreases'\n"
            "    print(\"  {} — mean |SHAP| = {:.4f} (on average {} ASD probability)\".format(\n"
            "        row['Feature'], row['Mean |SHAP|'], direction))"
        ),
    ]

    # --- Prompt 6: sklearn Pipeline + joblib save ---
    pipeline_cells = [
        cell_md("## Save Preprocessing + Model Pipeline"),
        cell_code(
            "import joblib\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import OneHotEncoder\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "NUMERIC_FEATURES = ['age', 'result']\n"
            "CATEGORICAL_FEATURES = ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'relation']\n"
            "SCORE_FEATURES = [\n"
            "    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',\n"
            "    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',\n"
            "]\n"
            "RAW_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + SCORE_FEATURES\n"
            "\n"
            "preprocessor = ColumnTransformer(\n"
            "    transformers=[\n"
            "        ('num', MinMaxScaler(), NUMERIC_FEATURES),\n"
            "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),\n"
            "        ('scores', 'passthrough', SCORE_FEATURES),\n"
            "    ],\n"
            "    remainder='drop',\n"
            ")\n"
            "\n"
            "pipeline_classifier = comparison_models[best_model_name]\n"
            "autism_pipeline = Pipeline([\n"
            "    ('preprocessor', preprocessor),\n"
            "    ('classifier', pipeline_classifier),\n"
            "])\n"
            "\n"
            "X_raw = features_raw.reset_index(drop=True)\n"
            "y_raw = data_classes.reset_index(drop=True)\n"
            "autism_pipeline.fit(X_raw, y_raw)\n"
            "\n"
            "joblib.dump(autism_pipeline, 'autism_pipeline.pkl')\n"
            "joblib.dump(autism_pipeline.named_steps['preprocessor'], 'preprocessor.pkl')\n"
            "joblib.dump(autism_pipeline.named_steps['classifier'], 'model.pkl')\n"
            "print('Saved: autism_pipeline.pkl, preprocessor.pkl, model.pkl')"
        ),
    ]

    cells.extend(smote_cells)
    cells.extend(shap_cells)
    cells.extend(pipeline_cells)

    nb["cells"] = cells
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {NOTEBOOK} — total cells: {len(cells)}")


if __name__ == "__main__":
    main()
