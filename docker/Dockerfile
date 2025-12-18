# Production-grade Dockerfile for Fraud Detection API
# Base: Python 3.11 slim (minimal + secure)
# Layers: dependencies → code → runtime

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY monitoring/ ./monitoring/
COPY config/ ./config/

# Create non-root user for security
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app
USER mlops

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API
CMD ["python", "api/main.py"]
