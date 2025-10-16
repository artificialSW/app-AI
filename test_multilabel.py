import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# 라벨 정보 로드
with open("outputs/multilabel_label_map.json", "r", encoding="utf-8") as f:
    label_info = json.load(f)

LABELS = [label_info["id2label"][str(i)] for i in range(len(label_info["id2label"]))]
print("라벨들:", LABELS)

# 모델과 토크나이저 로드
MODEL_PATH = "outputs/multilabel_export"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_multilabel(text, threshold=0.5):
    """멀티라벨 예측"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[0]  # 시그모이드로 확률 변환
    
    # threshold 이상인 라벨들 선택
    predicted_labels = []
    label_probs = {}
    
    for i, prob in enumerate(probs):
        label_name = LABELS[i]
        label_probs[label_name] = float(prob)
        if prob > threshold:
            predicted_labels.append(label_name)
    
    return {
        "predicted_labels": predicted_labels,
        "all_probabilities": label_probs
    }

# 테스트 예시들
test_texts = [
    "오늘 정말 기뻤고, 가족과 함께한 시간이 너무 소중했어",
    "힘들 때 위로해줘서 고마워, 옛날 생각도 나고",
    "우리 결혼기념일에 뭐 하고 싶어?",
    "요즘 무슨 취미활동 하고 있어?",
]

print("\n=== 멀티라벨 예측 테스트 ===")
for text in test_texts:
    result = predict_multilabel(text, threshold=0.3)
    print(f"\n텍스트: {text}")
    print(f"예측 라벨: {result['predicted_labels']}")
    print("확률들:")
    for label, prob in result['all_probabilities'].items():
        print(f"  {label}: {prob:.3f}")