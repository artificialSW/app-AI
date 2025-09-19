# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["애정표현","위로","특별한 날","과거 회상","기쁜일","취미"]

class InputPayload(BaseModel):
    text: str

class OutputPayload(BaseModel):
    label: str
    confidence: float
    probs: dict

app = FastAPI(title="Family Mood Classifier")

# export_model 폴더(merge된 모델)에서 로드
import os
from huggingface_hub import snapshot_download

# 모델 경로 설정
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "export_model")
HF_MODEL_NAME = "Pataegonia/korean-family-emotion-classifier"

# 모델이 로컬에 없으면 HuggingFace에서 다운로드
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from HuggingFace...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    snapshot_download(
        repo_id=HF_MODEL_NAME,
        local_dir=MODEL_PATH,
        repo_type="model"
    )
    print("Model downloaded successfully!")

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cpu")  # GPU 사용 원하면 "cuda"

@app.post("/predict", response_model=OutputPayload)
def predict(p: InputPayload):
    enc = tok(p.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**{k:v.to(device) for k,v in enc.items()}).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
    best_idx = int(torch.tensor(probs).argmax())
    return {
        "label": LABELS[best_idx],
        "confidence": float(probs[best_idx]),
        "probs": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    }
