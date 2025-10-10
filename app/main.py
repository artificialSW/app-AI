# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

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

# 글로벌 변수로 모델 저장
model = None
tokenizer = None

def load_model():
    """모델을 지연 로딩합니다."""
    global model, tokenizer
    
    if model is None:
        try:
            # 환경변수에서 모델 경로 읽기
            MODEL_PATH = os.getenv("MODEL_PATH", "output/export_model")
            HF_MODEL_NAME = "Pataegonia/korean-family-emotion-classifier"
            
            # 로컬 모델이 없거나 비어있으면 HuggingFace에서 다운로드
            model_files_exist = False
            if os.path.exists(MODEL_PATH):
                # 모델 파일이 실제로 있는지 확인
                model_files = ['model.safetensors', 'pytorch_model.bin', 'config.json']
                model_files_exist = any(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files)
            
            if not os.path.exists(MODEL_PATH) or not model_files_exist:
                print("Model not found locally. Downloading from HuggingFace...")
                from huggingface_hub import snapshot_download, login
                
                # HuggingFace 토큰으로 로그인 (선택적)
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
            
            print(f"Loading model from: {MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model.eval()
            print("Model loaded successfully!")
            
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
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=OutputPayload)
async def predict(p: InputPayload):
    """텍스트 분류 예측"""
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
            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
        
        # 결과 처리
        best_idx = int(torch.tensor(probs).argmax())
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
