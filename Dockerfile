FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential zlib1g-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 detoxify multilingual 模型
RUN python -c "from detoxify import Detoxify; Detoxify('multilingual')"

COPY . .

ENV GEMINI_API_KEY=""

EXPOSE 5000

CMD ["python", "-m", "api.main"]
