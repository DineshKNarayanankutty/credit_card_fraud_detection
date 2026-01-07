# ===============================
# Fraud Detection Inference API
# ===============================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default model locations (override in Azure ML / K8s)
ENV MODEL_PATH=/app/artifacts/model.pkl
ENV SCALER_PATH=/app/artifacts/scaler.pkl

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY artifacts/ ./artifacts/
COPY params.yaml .

RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
