# Autism Detection — Production Upgrade Roadmap

This roadmap upgrades the project from a university ML notebook into a production-ready system. Items are grouped by timeline and scored by **Impact** (H) / **Medium** (M) / **Low** (L) and **Difficulty** (Easy / Medium / Hard).

---

## 1. Immediate Improvements (Week 1–2)

| Priority | Item | Impact | Difficulty | Status |
|----------|------|--------|------------|--------|
| 1 | **XGBoost** — add gradient boosting to model comparison with Accuracy, Precision, Recall, F1, ROC-AUC | H | Easy | Done |
| 2 | **SMOTE** — analyze class imbalance; oversample training set only; compare before/after metrics | H | Easy | Done |
| 3 | **Unified model comparison table** — single DataFrame ranking all 7 classifiers | H | Easy | Done |
| 4 | **SVM tuning upgrade** — StratifiedKFold + search over `kernel`, `C`, `gamma`, `degree` | M | Easy | Done |
| 5 | **sklearn Pipeline** — ColumnTransformer (scale + encode + scores) + classifier; save with joblib | H | Medium | Done |
| 6 | **requirements.txt** — pin dependencies for reproducible installs | M | Easy | Done |
| 7 | **train_and_save.py** — standalone script to regenerate `model.pkl` / `preprocessor.pkl` without notebook | M | Easy | Done |

**Why first:** These deliver measurable model quality gains, fix data-leakage risk in preprocessing (pipeline fits on training data only in production script), and create deployable artifacts.

---

## 2. Intermediate Improvements (Week 3–6)

| Priority | Item | Impact | Difficulty | Status |
|----------|------|--------|------------|--------|
| 1 | **Flask REST API** (`app.py`) — `GET /`, `POST /predict` with JSON input and probability output | H | Medium | Done |
| 2 | **Streamlit UI** — input form for all 18 features, prediction + probability + SHAP | H | Medium | Done |
| 3 | **SHAP explainability** — summary plot, feature importance, force plot; top-10 feature narrative | H | Medium | Done |
| 4 | **Input validation layer** — schema validation (Pydantic / marshmallow) on API payloads | M | Easy | Pending |
| 5 | **Unit tests** — pytest for preprocessing, API endpoints, prediction shape | M | Medium | Pending |
| 6 | **CI pipeline** — GitHub Actions: lint, test, train smoke test on push | M | Medium | Pending |
| 7 | **Refactor notebook → modules** — move preprocessing, training, evaluation into `src/` package | M | Medium | Pending |
| 8 | **Fix preprocessing leakage in notebook** — fit scaler/encoder inside Pipeline on train split only | H | Medium | Partial (pipeline script) |

**Why next:** Expose the model to users (API + UI), add trust via SHAP, and establish engineering hygiene before scaling infrastructure.

---

## 3. Advanced Improvements (Month 2+)

| Priority | Item | Impact | Difficulty | Status |
|----------|------|--------|------------|--------|
| 1 | **Docker** — containerize Flask API + Streamlit; multi-stage build; health checks | H | Medium | Pending |
| 2 | **Model versioning** — MLflow or DVC to track experiments, params, metrics, artifacts | H | Hard | Pending |
| 3 | **Monitoring** — Prometheus/Grafana or Evidently AI for data drift, latency, error rates | H | Hard | Pending |
| 4 | **A/B testing** — shadow deployments comparing model versions in production | M | Hard | Pending |
| 5 | **Authentication** — API keys / OAuth2 for `/predict` in clinical settings | H | Medium | Pending |
| 6 | **Kubernetes deployment** — Helm charts, autoscaling, rolling updates | M | Hard | Pending |
| 7 | **FHIR / EHR integration** — standard healthcare data exchange | M | Hard | Pending |
| 8 | **Fairness audit** — evaluate performance across gender, ethnicity, age subgroups | H | Medium | Pending |
| 9 | **Automated retraining** — scheduled pipeline when new labeled data arrives | M | Hard | Pending |
| 10 | **Clinical disclaimer & governance** — IRB documentation, human-in-the-loop review workflow | H | Medium | Pending |

**Why last:** These require stable core ML + API foundations and organizational processes typical of regulated healthcare deployments.

---

## Impact vs. Difficulty Matrix

```
Impact
  ^
H |  XGBoost    SMOTE       Pipeline    Flask API
  |  SHAP       Streamlit   Docker      Monitoring
  |
M |  SVM tune    CI/tests    Auth        Fairness audit
  |
L |  requirements.txt
  +----------------------------------------> Difficulty
       Easy        Medium        Hard
```

---

## Recommended Execution Order

1. Run notebook end-to-end (or `python train_and_save.py`) to generate artifacts.
2. Start Flask: `python app.py` → test with `POST /predict`.
3. Start Streamlit: `streamlit run streamlit_app.py`.
4. Add Docker + MLflow (intermediate → advanced bridge).
5. Add monitoring before any real-user deployment.

---

## Quick Start (After Upgrades)

```bash
pip install -r requirements.txt
python train_and_save.py          # generates model.pkl, preprocessor.pkl, autism_pipeline.pkl
python app.py                     # Flask on http://localhost:5000
streamlit run streamlit_app.py    # Streamlit UI
```

---

## Medical & Ethical Note

This system is a **screening aid**, not a diagnostic tool. Production deployment requires clinical validation, regulatory review (where applicable), and explicit disclaimers that predictions must not replace professional assessment.
