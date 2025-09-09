FROM python:3.10-slim

WORKDIR /app

# system deps for torch and git-lfs if you choose to use them in image build (optional)
RUN apt-get update && apt-get install -y git-lfs git build-essential && git lfs install && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

ENV FLASK_ENV=production
CMD ["python", "app.py"]
