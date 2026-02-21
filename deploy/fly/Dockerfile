FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ ./code/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SEED_DATA=false
ENV ORT_LOGGING_LEVEL=3
ENV GRPC_VERBOSITY=ERROR
ENV GLOG_minloglevel=2
ENV TF_CPP_MIN_LOG_LEVEL=3

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "code.bot_runner"]
