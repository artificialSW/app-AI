import os, json, random, numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
MODEL_NAME = "klue/bert-base"
LABELS = ["애정표현","위로","특별한 날","과거 회상","기쁜일","취미"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_multilabel_jsonl(path):
    """멀티라벨 데이터 로드"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            # 멀티라벨을 바이너리 벡터로 변환
            label_vector = [0] * len(LABELS)
            for label_name in o["labels"]:
                if label_name in label2id:
                    label_vector[label2id[label_name]] = 1
            rows.append({"text": o["text"], "labels": label_vector})
    return Dataset.from_list(rows)

def tokenize(batch, tok):
    return tok(batch["text"], truncation=True, max_length=128)

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_multilabel_metrics(eval_pred):
    logits, labels = eval_pred
    # 시그모이드 적용 후 0.5로 임계값
    predictions = torch.sigmoid(torch.tensor(logits)) > 0.5
    predictions = predictions.int().numpy()
    labels = labels.astype(int)
    
    # 전체 정확도 (모든 라벨이 맞아야 함)
    exact_match = np.all(predictions == labels, axis=1).mean()
    
    # F1 스코어들
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        "exact_match": exact_match,
        "f1_micro": f1_micro, 
        "f1_macro": f1_macro
    }

def main():
    set_seed()
    os.makedirs("outputs", exist_ok=True)

    # 1) load data
    train_ds = load_multilabel_jsonl("data/train_multilabel.jsonl")
    valid_ds = load_multilabel_jsonl("data/val_multilabel.jsonl")

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(valid_ds)}")

    # 2) tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_tok = train_ds.map(lambda b: tokenize(b, tok), batched=True)
    valid_tok = valid_ds.map(lambda b: tokenize(b, tok), batched=True)

    cols = ["input_ids","attention_mask","labels"]
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in cols])
    valid_tok = valid_tok.remove_columns([c for c in valid_tok.column_names if c not in cols])

    # 3) multilabel model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABELS), 
        id2label=id2label, 
        label2id=label2id,
        problem_type="multi_label_classification"
    )

    # 4) trainer
    args = TrainingArguments(
        output_dir="outputs/multilabel_ckpt",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED
    )

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    trainer = MultilabelTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=valid_tok,
        data_collator=data_collator,
        compute_metrics=compute_multilabel_metrics,
    )

    trainer.train()
    print("Multilabel training completed!")

    # 5) save model
    model.save_pretrained("outputs/multilabel_export")
    tok.save_pretrained("outputs/multilabel_export")

    # 6) save label map
    with open("outputs/multilabel_label_map.json","w",encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()