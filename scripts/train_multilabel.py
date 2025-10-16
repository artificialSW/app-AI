#!/usr/bin/env python3
"""
다중 레이블 분류 모델 훈련 스크립트
"""

import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset
from typing import List, Dict, Any
import os

class MultiLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 레이블을 바이너리 형태로 변환
        self.mlb = MultiLabelBinarizer()
        self.label_matrix = self.mlb.fit_transform(labels)
        self.num_labels = len(self.mlb.classes_)
        
        print(f"레이블 클래스: {list(self.mlb.classes_)}")
        print(f"레이블 수: {self.num_labels}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.label_matrix[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # 다중 레이블을 위한 BCE Loss 사용
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def load_multilabel_data(file_path: str):
    """다중 레이블 데이터 로드"""
    texts = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                texts.append(data['text'])
                labels.append(data['labels'])
    
    return texts, labels

def compute_metrics(eval_pred):
    """다중 레이블 평가 메트릭"""
    predictions, labels = eval_pred
    
    # 시그모이드 활용해서 확률로 변환
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    # 0.5 임계값으로 예측
    y_pred = (probs > 0.5).int().numpy()
    y_true = labels.astype(int)
    
    # 메트릭 계산
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }

def train_multilabel_model(
    train_file: str,
    val_file: str,
    model_name: str = "klue/bert-base",
    output_dir: str = "outputs/multilabel_model",
    max_length: int = 128,
    batch_size: int = 16,
    num_epochs: int = 5,
    learning_rate: float = 2e-5
):
    """다중 레이블 모델 훈련"""
    
    # 데이터 로드
    print("데이터 로딩 중...")
    train_texts, train_labels = load_multilabel_data(train_file)
    val_texts, val_labels = load_multilabel_data(val_file)
    
    print(f"훈련 데이터: {len(train_texts)}개")
    print(f"검증 데이터: {len(val_texts)}개")
    
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 레이블 수 확인을 위해 임시로 dataset 생성
    temp_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer, max_length)
    num_labels = temp_dataset.num_labels
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # 데이터셋 생성
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer, max_length)
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        save_total_limit=3,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )
    
    # 트레이너 생성
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 훈련 시작
    print("모델 훈련 시작...")
    trainer.train()
    
    # 최고 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 레이블 정보 저장
    label_info = {
        "labels": list(train_dataset.mlb.classes_),
        "num_labels": num_labels
    }
    
    with open(f"{output_dir}/label_info.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    
    print(f"모델 저장 완료: {output_dir}")
    return trainer

if __name__ == "__main__":
    train_file = "data/train_multilabel.jsonl"
    val_file = "data/val_multilabel.jsonl"
    
    if not os.path.exists(train_file):
        print(f"훈련 데이터가 없습니다: {train_file}")
        print("먼저 convert_to_multilabel.py를 실행해서 데이터를 변환하세요.")
        exit(1)
    
    trainer = train_multilabel_model(
        train_file=train_file,
        val_file=val_file,
        output_dir="outputs/multilabel_model",
        num_epochs=5,
        batch_size=16
    )