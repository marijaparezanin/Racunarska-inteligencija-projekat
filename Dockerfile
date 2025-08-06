# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 5000 
EXPOSE 8501

# No CMD here – Compose will override it
