# scripts/eval_confmat.py
import json, numpy as np, matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LABELS = ["애정표현","위로","특별한 날","과거 회상","기쁜일","취미"]
label2id = {l:i for i,l in enumerate(LABELS)}

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append({"text": o["text"], "label": label2id[o["label"]]})
    return Dataset.from_list(rows)

tok = AutoTokenizer.from_pretrained("outputs/lora_ckpt")
model = AutoModelForSequenceClassification.from_pretrained("outputs/lora_ckpt")
model.eval()

test_ds = load_jsonl("data/test.jsonl")

def predict(texts):
    enc = tok(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
    return preds

y_true = np.array(test_ds["label"])
y_pred = predict(test_ds["text"])

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=160)
print("saved -> outputs/confusion_matrix.png")
