<<<<<<< HEAD
FROM python:3.10-slim

WORKDIR /app

# This is for matplotlib to run in docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/evaluate.py .

=======
FROM python:3.10-slim

WORKDIR /app

# This is for matplotlib to run in docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/evaluate.py .

>>>>>>> 16f0156 (Initial commit - EfficientNet Pneumonia Classification with Docker)
CMD ["python", "evaluate.py"]