FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl нужен только для healthcheck админки.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY btcproc ./btcproc

# Кэш скачанных дампов Binance переживает пересборку образа только через volume.
VOLUME ["/app/data"]

EXPOSE 8100

CMD ["python", "-m", "btcproc.cli", "admin", "--host", "0.0.0.0"]
