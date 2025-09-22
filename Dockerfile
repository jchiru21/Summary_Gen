FROM python:3.10-slim

WORKDIR /app

# minimal system deps
RUN apt-get update && apt-get install -y build-essential --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install CPU-only PyTorch then other requirements (adjust if you pinned versions)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu/ torch \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 8 --timeout 300 --bind 0.0.0.0:${PORT:-8080} app:app"]
