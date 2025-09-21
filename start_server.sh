#!/bin/bash

# start_server.sh
# FastAPI 서버를 Gunicorn으로 실행하는 스크립트

# 가상환경 활성화
source venv/bin/activate

# 로그 디렉토리 생성
mkdir -p logs

# Gunicorn으로 서버 시작
exec gunicorn app.main:app -c gunicorn.conf.py