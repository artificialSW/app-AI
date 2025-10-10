#!/usr/bin/env python3
"""
간단한 다중 레이블 분류 모델 훈련 스크립트
"""

import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset
from typing import List, Dict, Any
import os
from collections import Counter

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
        
        # 레이블 분포 확인
        label_counts = Counter()
        for label_list in labels:
            for label in label_list:
                label_counts[label] += 1
        
        print("\n레이블 분포:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}개")
    
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
    
    print(f"데이터 로딩: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    texts.append(data['text'])
                    
                    # labels 필드 확인
                    if 'labels' in data:
                        labels.append(data['labels'])
                    elif 'label' in data:
                        # 단일 레이블을 리스트로 변환
                        single_label = data['label']
                        if isinstance(single_label, str):
                            labels.append([single_label])
                        else:
                            labels.append(single_label)
                    else:
                        print(f"Warning: Line {line_num} has no label field")
                        continue
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    
    print(f"로드된 데이터: {len(texts)}개")
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
    f1_samples = f1_score(y_true, y_pred, average='samples')
    
    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples
    }

def main():
    # 데이터 파일 확인
    train_file = "data/train_multilabel.jsonl"
    val_file = "data/val_multilabel.jsonl"
    
    if not os.path.exists(train_file):
        print(f"훈련 데이터 없음: {train_file}")
        print("먼저 data/data.jsonl을 train/val로 분할해야 합니다.")
        
        # 자동 분할
        print("자동으로 데이터를 분할합니다...")
        
        if not os.path.exists("data/data.jsonl"):
            print("Error: data/data.jsonl 파일이 없습니다!")
            return
        
        import random
        
        # 전체 데이터 로드
        all_texts, all_labels = load_multilabel_data("data/data.jsonl")
        
        # 데이터와 레이블을 함께 셔플
        combined = list(zip(all_texts, all_labels))
        random.seed(42)
        random.shuffle(combined)
        
        # 80:20 분할
        split_idx = int(len(combined) * 0.8)
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        # 훈련 데이터 저장
        with open(train_file, 'w', encoding='utf-8') as f:
            for text, labels in train_data:
                f.write(json.dumps({"text": text, "labels": labels}, ensure_ascii=False) + '\n')
        
        # 검증 데이터 저장
        with open(val_file, 'w', encoding='utf-8') as f:
            for text, labels in val_data:
                f.write(json.dumps({"text": text, "labels": labels}, ensure_ascii=False) + '\n')
        
        print(f"훈련 데이터: {len(train_data)}개")
        print(f"검증 데이터: {len(val_data)}개")
    
    # 데이터 로드
    print("\n=== 데이터 로딩 ===")
    train_texts, train_labels = load_multilabel_data(train_file)
    val_texts, val_labels = load_multilabel_data(val_file)
    
    # 토크나이저와 모델 로드
    model_name = "klue/bert-base"
    print(f"\n=== 모델 로딩: {model_name} ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 레이블 수 확인을 위해 임시로 dataset 생성
    temp_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer, 128)
    num_labels = temp_dataset.num_labels
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # 데이터셋 생성
    print(f"\n=== 데이터셋 생성 ===")
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer, 128)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer, 128)
    
    # 훈련 설정
    output_dir = "./multilabel_model"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # 적은 데이터이므로 3 에폭
        per_device_train_batch_size=8,  # 메모리 고려해서 작게
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        save_total_limit=2,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=None,  # wandb 등 비활성화
    )
    
    # 트레이너 생성
    print(f"\n=== 훈련 시작 ===")
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 훈련 시작
    trainer.train()
    
    # 최고 모델 저장
    print(f"\n=== 모델 저장: {output_dir} ===")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 레이블 정보 저장
    label_info = {
        "labels": list(train_dataset.mlb.classes_),
        "num_labels": num_labels
    }
    
    with open(f"{output_dir}/label_info.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    
    print(f"훈련 완료!")
    print(f"모델 저장 위치: {output_dir}")
    print(f"레이블: {list(train_dataset.mlb.classes_)}")

if __name__ == "__main__":
    main()