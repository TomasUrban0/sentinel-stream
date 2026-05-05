FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "sentinel_stream.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
