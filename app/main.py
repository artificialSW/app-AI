# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import json

# .env 파일 로드
load_dotenv()

LABELS = ["애정표현","위로","특별한 날","과거 회상","기쁜일","취미"]

# 감정 카테고리별 꽃 매핑
EMOTION_FLOWERS = {
    "애정표현": ["동백꽃", "장미"],
    "위로": ["아카시아꽃", "수국"],
    "특별한 날": ["매화꽃", "튤립"],
    "과거 회상": ["팥배꽃", "재비꽃"],
    "기쁜일": ["벚꽃", "코스모스"],
    "취미": ["목련", "해바라기"]
}

import random

class InputPayload(BaseModel):
    text: str

class OutputPayload(BaseModel):
    label: str
    confidence: float
    flower: str
    probs: dict

app = FastAPI(title="Family Mood Classifier")

# 글로벌 변수로 모델 저장 (멀티라벨 모델을 메인으로 사용)
model = None
tokenizer = None

def load_model():
    """멀티라벨 모델을 메인 모델로 로딩합니다."""
    global model, tokenizer
    
    if model is None:
        try:
            # 먼저 로컬 멀티라벨 모델을 시도
            MULTILABEL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "multilabel_export")
            
            if os.path.exists(MULTILABEL_MODEL_PATH):
                print(f"Loading multilabel model from: {MULTILABEL_MODEL_PATH}")
                tokenizer = AutoTokenizer.from_pretrained(MULTILABEL_MODEL_PATH)
                model = AutoModelForSequenceClassification.from_pretrained(
                    MULTILABEL_MODEL_PATH,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model.eval()
                print("Multilabel model loaded successfully!")
                return
            
            # 멀티라벨 모델이 없으면 기존 단일라벨 모델 로드
            MODEL_PATH = os.getenv("MODEL_PATH", "outputs/export_model")
            HF_MODEL_NAME = "Pataegonia/korean-family-emotion-classifier"
            
            # 로컬 모델이 없거나 비어있으면 HuggingFace에서 다운로드
            model_files_exist = False
            if os.path.exists(MODEL_PATH):
                model_files = ['model.safetensors', 'pytorch_model.bin', 'config.json']
                model_files_exist = any(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files)
            
            if not os.path.exists(MODEL_PATH) or not model_files_exist:
                print("Model not found locally. Downloading from HuggingFace...")
                from huggingface_hub import snapshot_download, login
                
                hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
                if hf_token and hf_token.strip():
                    try:
                        login(token=hf_token.strip())
                        print("Logged in to HuggingFace Hub")
                    except Exception as e:
                        print(f"Token login failed: {e}, trying without token...")
                
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                snapshot_download(
                    repo_id=HF_MODEL_NAME,
                    local_dir=MODEL_PATH,
                    repo_type="model",
                    token=hf_token if hf_token else None
                )
                print("Model downloaded successfully!")
            
            print(f"Loading single-label model from: {MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model.eval()
            print("Single-label model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_model()

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None
    }

def is_multilabel_model(model_path):
    """모델이 멀티라벨 모델인지 확인"""
    try:
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("problem_type") == "multi_label_classification"
    except:
        pass
    return False

@app.post("/predict", response_model=OutputPayload)
async def predict(p: InputPayload):
    """텍스트 분류 예측 (멀티라벨 모델을 사용하되 최고 확률의 단일 라벨 반환)"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 토큰화
        enc = tokenizer(
            p.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # 예측
        with torch.no_grad():
            logits = model(**enc).logits
            
            # 멀티라벨 모델인지 확인하여 적절한 활성화 함수 사용
            if hasattr(model.config, 'problem_type') and model.config.problem_type == "multi_label_classification":
                # 멀티라벨 모델: 시그모이드 사용
                probs = torch.sigmoid(logits)[0].cpu().tolist()
            else:
                # 단일라벨 모델: 소프트맥스 사용
                probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
        
        # 가장 높은 확률의 라벨 선택 (단일 라벨 출력)
        best_idx = probs.index(max(probs))
        predicted_label = LABELS[best_idx] if best_idx < len(LABELS) else "기타"
        
        # 예측된 감정에 따라 랜덤 꽃 선택
        selected_flower = random.choice(EMOTION_FLOWERS.get(predicted_label, ["꽃"]))
        
        return {
            "label": predicted_label,
            "confidence": float(probs[best_idx]),
            "flower": selected_flower,
            "probs": {LABELS[i]: float(probs[i]) for i in range(min(len(LABELS), len(probs)))}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
