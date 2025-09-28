# 🤖 Family Emotion Classifier API

한국어 가족 감정 분류를 위한 FastAPI 기반 AI 서버입니다. 사용자 입력 텍스트를 분석하여 6가지 감정 카테고리로 분류합니다.

## 📊 분류 카테고리

- **애정표현**: 사랑, 애정을 표현하는 텍스트
- **위로**: 위로, 격려의 내용
- **특별한 날**: 기념일, 특별한 순간
- **과거 회상**: 추억, 회상의 내용
- **기쁜일**: 기쁨, 행복감을 나타내는 텍스트
- **취미**: 취미, 관심사에 관한 내용

## 🏗️ 프로젝트 구조

```
app-AI/
├── app/
│   └── main.py                 # FastAPI 메인 애플리케이션
├── data/
│   ├── data.jsonl             # 원본 데이터
│   ├── train.jsonl            # 훈련 데이터
│   ├── val.jsonl              # 검증 데이터
│   └── split.py               # 데이터 분할 스크립트
├── scripts/
│   ├── train_lora.py          # LoRA 미세조정 스크립트
│   ├── merge_lora.py          # LoRA 모델 병합
│   └── eval_comfmat.py        # 평가 및 혼동행렬
├── models/                    # 모델 저장 디렉토리
├── output/                    # 훈련 결과 저장
├── .env                       # 환경변수 설정
├── requirements.txt           # Python 의존성
├── gunicorn.conf.py          # Gunicorn 설정
└── README.md
```

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/artificialSW/app-AI.git
cd app-AI
```

### 2. 가상환경 설정

```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
# HuggingFace Hub Token for model access
HUGGINGFACE_HUB_TOKEN=your_hf_token_here

# Model Configuration
MODEL_NAME=Pataegonia/korean-family-emotion-classifier
MODEL_CACHE_DIR=./models
```

### 5. HuggingFace 토큰 발급

1. [HuggingFace](https://huggingface.co/)에 가입/로그인
2. [Settings > Access Tokens](https://huggingface.co/settings/tokens)에서 **Read** 권한 토큰 생성
3. `.env` 파일에 토큰 추가

### 6. 서버 실행

**개발 환경:**
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**프로덕션 환경:**
```bash
gunicorn app.main:app -c gunicorn.conf.py
```

## 🔌 API 사용법

### 서버 실행 후 접속

- **API 문서**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### API 엔드포인트

#### 1. 감정 분류 예측

**POST** `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "오늘 정말 즐거운 하루였어요!"}'
```

**응답 예시:**
```json
{
  "label": "기쁜일",
  "confidence": 0.8456,
  "probs": {
    "애정표현": 0.0234,
    "위로": 0.0456,
    "특별한 날": 0.0789,
    "과거 회상": 0.0123,
    "기쁜일": 0.8456,
    "취미": 0.0942
  }
}
```

#### 2. 헬스 체크

**GET** `/health`

```bash
curl -X GET "http://localhost:8000/health"
```

**응답 예시:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🤖 모델 정보

- **모델명**: Pataegonia/korean-family-emotion-classifier
- **기반 모델**: BERT 기반 한국어 사전훈련 모델
- **훈련 방법**: LoRA (Low-Rank Adaptation) 미세조정
- **지원 언어**: 한국어

## 🔧 개발 및 훈련

### 데이터 준비

```bash
# 데이터 분할
python data/split.py
```

### 모델 훈련

```bash
# LoRA 미세조정
python scripts/train_lora.py

# LoRA 모델 병합
python scripts/merge_lora.py

# 모델 평가
python scripts/eval_comfmat.py
```

## 🚢 배포

### AWS EC2 배포

1. **EC2 인스턴스 설정**
```bash
# 패키지 업데이트
sudo yum update -y

# Python 3.9 설치
sudo yum install python3.9 python3.9-pip -y

# Git 설치
sudo yum install git -y
```

2. **프로젝트 배포**
```bash
# 프로젝트 클론
git clone https://github.com/artificialSW/app-AI.git
cd app-AI

# 가상환경 설정
python3.9 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" > .env

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. **백그라운드 실행**
```bash
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

### Docker 배포

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Docker 이미지 빌드
docker build -t family-emotion-api .

# Docker 컨테이너 실행
docker run -d -p 8000:8000 --env-file .env family-emotion-api
```

## 🛠️ 기술 스택

- **웹 프레임워크**: FastAPI
- **ML 라이브러리**: PyTorch, Transformers, PEFT
- **서버**: Uvicorn, Gunicorn
- **모델**: HuggingFace Hub
- **언어**: Python 3.9+

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 GitHub Issues를 통해 연락해주세요.

---

⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!
