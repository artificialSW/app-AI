import json
from sklearn.model_selection import train_test_split

# 데이터 로드
with open("./data/data.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

# stratified split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# 저장 함수
def save_jsonl(filename, texts, labels):
    with open(filename, "w", encoding="utf-8") as f:
        for t, l in zip(texts, labels):
            f.write(json.dumps({"text": t, "label": l}, ensure_ascii=False) + "\n")

save_jsonl("train.jsonl", train_texts, train_labels)
save_jsonl("val.jsonl", val_texts, val_labels)
