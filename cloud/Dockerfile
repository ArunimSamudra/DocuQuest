FROM --platform=linux/amd64 python:3.10-slim

RUN pip install --no-cache-dir transformers google-cloud-storage flask
RUN pip install --no-cache-dir torch

COPY llama_model /app/llama_model
COPY app.py /app/app.py

WORKDIR /app
CMD ["python", "app.py"]
