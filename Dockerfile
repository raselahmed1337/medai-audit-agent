FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Render sets $PORT; HF Spaces uses app_port (default 8000 below)
EXPOSE 8000
CMD ["sh", "-c", "python -m medai.server --host 0.0.0.0 --port ${PORT:-8000}"]
