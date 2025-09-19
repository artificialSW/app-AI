# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY app/ ./app/
COPY scripts/ ./scripts/

# 모델 다운로드 스크립트
COPY download_model.py .

# 모델 다운로드 및 서버 시작
CMD ["sh", "-c", "python download_model.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

EXPOSE 8000