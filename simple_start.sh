#!/bin/bash

# simple_start.sh - 메모리 효율적인 서버 시작 스크립트

# 가상환경 활성화
source venv/bin/activate

# 로그 디렉토리 생성
mkdir -p logs

# 단일 워커로 uvicorn 직접 실행 (gunicorn보다 메모리 효율적)
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info