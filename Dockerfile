FROM python:3.12-slim

LABEL maintainer="Zafran"
LABEL description="malaysian-manglish-nlp REST API + WebSocket — Full NLP toolkit for Malaysian Manglish"

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install package with API extras (includes websockets)
RUN pip install --no-cache-dir -e ".[api]"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API
CMD ["uvicorn", "malaysian_manglish_nlp.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]
