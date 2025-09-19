import os, json, random, numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

SEED = 42
MODEL_NAME = "klue/bert-base"
LABELS = ["애정표현","위로","특별한 날","과거 회상","기쁜일","취미"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append({"text": o["text"], "label": label2id[o["label"]]})
    return Dataset.from_list(rows)

def compute_class_weights(train_ds):
    counts = [0]*len(LABELS)
    for y in train_ds["label"]:
        counts[y]+=1
    total = sum(counts)
    # inverse frequency, normalized
    weights = np.array([total/c for c in counts], dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def tokenize(batch, tok):
    return tok(batch["text"], truncation=True, max_length=128)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f, "macro_precision": p, "macro_recall": r}

def main():
    set_seed()
    os.makedirs("outputs", exist_ok=True)

    # 1) load data
    train_ds = load_jsonl("data/train.jsonl")
    valid_ds = load_jsonl("data/val.jsonl")

    # 2) tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_tok = train_ds.map(lambda b: tokenize(b, tok), batched=True)
    valid_tok = valid_ds.map(lambda b: tokenize(b, tok), batched=True)

    cols = ["input_ids","attention_mask","label"]
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in cols])
    valid_tok = valid_tok.remove_columns([c for c in valid_tok.column_names if c not in cols])

    # 3) base + LoRA
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["query","key","value","dense"], # BERT 계열 일반적
        bias="none", task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(base, lora_cfg)

    # 4) class weights
    class_weights = compute_class_weights(train_ds)

    # 5) trainer
    args = TrainingArguments(
        output_dir="outputs/ckpt",
        learning_rate=2e-4,             # LoRA라 비교적 큼
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=SEED
    )

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=valid_tok,
        tokenizer=tok, data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()
    print("Training completed!")

    # 7) save lora adapter
    model.save_pretrained("outputs/lora_ckpt")
    tok.save_pretrained("outputs/lora_ckpt")

    # 8) save label map
    with open("outputs/label_map.json","w",encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
