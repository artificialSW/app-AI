# app/main_multilabel.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import json
from typing import List, Dict
import numpy as np

# .env 파일 로드
load_dotenv()

class InputPayload(BaseModel):
    text: str
    threshold: float = 0.5  # 다중 레이블 임계값

class OutputPayload(BaseModel):
    label: str  # 예측된 레이블 (가장 높은 확률)
    confidence: float  # 가장 높은 확률
    probs: Dict[str, float]  # 각 레이블별 확률

app = FastAPI(title="Multi-Label Family Emotion Classifier")

# 글로벌 변수
model = None
tokenizer = None
label_names = []

def load_model():
    """다중 레이블 모델을 지연 로딩합니다."""
    global model, tokenizer, label_names
    
    if model is None:
        try:
            # 환경변수에서 모델 경로 읽기
            MODEL_PATH = os.getenv("MODEL_PATH", "multilabel_model")
            
            # 모델이 없으면 HuggingFace에서 다운로드
            if not os.path.exists(MODEL_PATH):
                print("Model not found locally. Please train the model first.")
                raise HTTPException(status_code=500, detail="Model not found")
            
            # 레이블 정보 로드
            label_info_path = os.path.join(MODEL_PATH, "label_info.json")
            if os.path.exists(label_info_path):
                with open(label_info_path, 'r', encoding='utf-8') as f:
                    label_info = json.load(f)
                    label_names = label_info['labels']
            else:
                # 기본 레이블
                label_names = ["애정표현", "위로", "특별한 날", "과거 회상", "기쁜일", "취미"]
            
            print(f"Loading model from: {MODEL_PATH}")
            print(f"Labels: {label_names}")
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model.eval()
            print("Multi-label model loaded successfully!")
            
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
        "model_loaded": model is not None,
        "model_type": "multi-label",
        "labels": label_names
    }

@app.post("/predict", response_model=OutputPayload)
async def predict(p: InputPayload):
    """텍스트 분류 예측 - 단일 레이블 반환"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 텍스트 토크나이징
        inputs = tokenizer(
            p.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 시그모이드로 확률 변환
            probabilities = torch.sigmoid(logits).squeeze().tolist()
            
            # 레이블별 확률 딕셔너리 생성
            prob_dict = {}
            for label, prob in zip(label_names, probabilities):
                prob_dict[label] = float(prob)
            
            # 가장 높은 확률의 레이블과 신뢰도
            max_idx = np.argmax(probabilities)
            top_label = label_names[max_idx]
            confidence = float(probabilities[max_idx])
            
            return OutputPayload(
                label=top_label,
                confidence=confidence,
                probs=prob_dict
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/multi")
async def predict_multi_label(p: InputPayload):
    """다중 레이블 결과 반환 (임계값 기반)"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 텍스트 토크나이징
        inputs = tokenizer(
            p.text,
            return_tensors="pt", 
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 시그모이드로 확률 변환
            probabilities = torch.sigmoid(logits).squeeze().tolist()
            
            # 레이블별 확률 딕셔너리 생성
            prob_dict = {}
            predicted_labels = []
            
            for label, prob in zip(label_names, probabilities):
                prob_dict[label] = float(prob)
                
                # 임계값보다 높은 레이블들을 선택
                if prob >= p.threshold:
                    predicted_labels.append(label)
            
            # 예측된 레이블이 없으면 가장 높은 확률의 레이블 선택
            if not predicted_labels:
                max_idx = np.argmax(probabilities)
                predicted_labels = [label_names[max_idx]]
            
            return {
                "labels": predicted_labels,
                "probabilities": prob_dict,
                "threshold": p.threshold
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-label prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)