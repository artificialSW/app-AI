# app/main_simple.py - 메모리 효율적인 버전
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

LABELS = ["애정표현", "위로", "특별한 날", "과거 회상", "기쁜일", "취미"]

class InputPayload(BaseModel):
    text: str

class OutputPayload(BaseModel):
    label: str
    confidence: float
    probs: dict

app = FastAPI(title="Family Mood Classifier - Simple Version")

# 글로벌 변수로 모델 저장
model = None
tokenizer = None

def load_model():
    """모델을 지연 로딩합니다."""
    global model, tokenizer
    
    if model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # 메모리 효율적인 설정
            torch.set_num_threads(1)  # CPU 스레드 제한
            
            MODEL_PATH = "output/export_model"
            
            # 로컬 모델이 없으면 간단한 fallback 모델 사용
            if not os.path.exists(MODEL_PATH):
                MODEL_PATH = "klue/bert-base"  # 더 작은 한국어 모델
            
            print(f"Loading model from: {MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,  # float16 대신 float32 (호환성)
                low_cpu_mem_usage=True      # 메모리 효율적 로딩
            )
            model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")

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
        import torch
        import torch.nn.functional as F
        
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
        
        return {
            "label": LABELS[best_idx] if best_idx < len(LABELS) else "기타",
            "confidence": float(probs[best_idx]),
            "probs": {LABELS[i]: float(probs[i]) for i in range(min(len(LABELS), len(probs)))}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)