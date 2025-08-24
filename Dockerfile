FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mental_health_chatbot_STREAMlit.py ./app.py
ENV PORT=8080
EXPOSE 8080
CMD ["streamlit","run","app.py","--server.port=8080","--server.headless=true","--server.enableCORS=false","--server.enableXsrfProtection=false"]
