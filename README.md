# app-AI
FAST API를 이용한 AI 서버

## 설정 방법

### 1. 환경변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에서 Hugging Face 토큰 설정
# HUGGINGFACE_TOKEN=your_actual_token_here
```

### 2. Hugging Face 토큰 발급
1. [Hugging Face](https://huggingface.co/)에 가입/로그인
2. [Settings > Access Tokens](https://huggingface.co/settings/tokens)에서 토큰 생성
3. `.env` 파일에 토큰 추가

### 3. 서버 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
gunicorn app.main:app -c gunicorn.conf.py
```
