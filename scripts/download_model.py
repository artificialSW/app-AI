#!/usr/bin/env python3
"""
모델을 로컬에 다운로드하는 스크립트
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def download_model():
    model_name = "klue/bert-base"
    save_path = "./klue-bert-base"
    
    print(f"모델 다운로드 중: {model_name}")
    
    # 토크나이저 다운로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    
    # 모델 다운로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6  # 기본 레이블 수
    )
    model.save_pretrained(save_path)
    
    print(f"모델 저장 완료: {save_path}")

if __name__ == "__main__":
    download_model()