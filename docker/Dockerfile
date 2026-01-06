# ===============================
# Fraud Detection Inference API
# ===============================

FROM python:3.9-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache-friendly)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ ./api/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY params.yaml .

# Non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

EXPOSE 8000

# Health check (FastAPI)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API properly
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
