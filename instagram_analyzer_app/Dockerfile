FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /tmp/logs /tmp/uploads /tmp/processing logs

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LOG_DIR=/tmp/logs \
    UPLOAD_DIR=/tmp/uploads \
    PROCESSING_DIR=/tmp/processing

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status==200 else 1)" || exit 1

# gunicorn supervises 2 uvicorn workers: one blocked/crashed worker no longer
# stalls the whole site. Keep -w low — this host also runs the TF workers.
CMD ["gunicorn", "main:app", "-k", "uvicorn_worker.UvicornWorker", "-w", "2", "--bind", "0.0.0.0:8000", "--timeout", "120"]
