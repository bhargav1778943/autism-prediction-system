# =====================================================================
# Base stage: Install common dependencies and system libraries
# =====================================================================
FROM python:3.11-slim AS base

WORKDIR /app

# Install libgomp1 (required for XGBoost OpenMP execution on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serialized model artifacts and python scripts
COPY autism_pipeline.pkl .
COPY preprocessor.pkl .
COPY model.pkl .
COPY app.py .
COPY streamlit_app.py .

# =====================================================================
# Target stage: Flask API
# =====================================================================
FROM base AS api
EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["python", "app.py"]

# =====================================================================
# Target stage: Streamlit UI
# =====================================================================
FROM base AS ui
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
