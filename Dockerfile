FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 8000

ENV AQI_DB_URL=postgresql://postgres@db:5432/india_air_quality

CMD ["streamlit", "run", "scripts/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
