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

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "code/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]
