FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (ca-certificates for HTTPS, build basics not required for pinned wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py /app/app.py

EXPOSE 8080
CMD ["streamlit","run","app.py","--server.port=8080","--server.headless=true","--server.enableCORS=false","--server.enableXsrfProtection=false"]
