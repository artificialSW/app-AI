#!/usr/bin/env python3
"""
단일 레이블 데이터를 다중 레이블 형식으로 변환하는 스크립트
"""

import json
import os
from typing import List, Dict, Any

def convert_single_to_multi_label(input_file: str, output_file: str):
    """
    단일 레이블 JSONL을 다중 레이블 형식으로 변환
    """
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                # 단일 레이블을 리스트로 변환
                converted_data.append({
                    "text": data["text"],
                    "labels": [data["label"]]  # 단일 레이블을 리스트로 감쌈
                })
    
    # 결과를 새 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"변환 완료: {len(converted_data)}개 데이터")
    print(f"출력 파일: {output_file}")

def suggest_multi_labels(text: str, current_label: str) -> List[str]:
    """
    텍스트 내용을 분석해서 추가 레이블을 제안하는 함수
    """
    labels = [current_label]
    text_lower = text.lower()
    
    # 키워드 기반 다중 레이블 제안
    label_keywords = {
        "애정표현": ["사랑", "고마", "소중", "가족", "함께", "우리", "감사", "표현", "마음"],
        "위로": ["힘들", "아프", "슬프", "고민", "두렵", "걱정", "위로", "괜찮", "힘내"],
        "특별한 날": ["꿈", "목표", "계획", "버킷리스트", "하고싶", "도전", "새해", "미래"],
        "과거 회상": ["어렸을때", "예전", "추억", "기억", "그때", "옛날", "돌아보", "회상"],
        "기쁜일": ["행복", "기쁜", "즐거", "웃음", "좋았", "감사", "자랑", "뿌듯"],
        "취미": ["좋아하는", "취미", "관심", "즐기", "여행", "음식", "영화", "책", "노래"]
    }
    
    for label, keywords in label_keywords.items():
        if label != current_label:  # 현재 레이블이 아닌 경우에만
            if any(keyword in text_lower for keyword in keywords):
                if label not in labels:
                    labels.append(label)
    
    return labels

def interactive_labeling(input_file: str, output_file: str):
    """
    대화형으로 다중 레이블을 지정하는 함수
    """
    all_labels = ["애정표현", "위로", "특별한 날", "과거 회상", "기쁜일", "취미"]
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n=== 다중 레이블 변환 시작 ===")
    print(f"총 {len(lines)}개 데이터를 처리합니다.")
    print("\n가능한 레이블:", ", ".join(all_labels))
    print("\n사용법:")
    print("- Enter: 현재 레이블만 유지")
    print("- 숫자: 추가 레이블 선택 (예: 1,3,5)")
    print("- 'auto': 자동 제안 사용")
    print("- 'skip': 건너뛰기")
    print("- 'quit': 종료\n")
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        data = json.loads(line.strip())
        text = data["text"]
        current_label = data["label"]
        
        print(f"\n[{i+1}/{len(lines)}]")
        print(f"텍스트: {text}")
        print(f"현재 레이블: {current_label}")
        
        # 자동 제안
        suggested_labels = suggest_multi_labels(text, current_label)
        if len(suggested_labels) > 1:
            print(f"제안 레이블: {', '.join(suggested_labels)}")
        
        # 사용자 입력
        for idx, label in enumerate(all_labels):
            marker = "★" if label == current_label else " "
            print(f"  {idx+1}. {marker} {label}")
        
        user_input = input(f"추가 레이블 선택 (현재: {current_label}): ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'skip':
            continue
        elif user_input.lower() == 'auto':
            final_labels = suggested_labels
        elif user_input == '':
            final_labels = [current_label]
        else:
            try:
                # 숫자로 레이블 선택
                selected_indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                selected_labels = [all_labels[idx] for idx in selected_indices if 0 <= idx < len(all_labels)]
                
                # 기존 레이블과 합치기
                final_labels = list(set([current_label] + selected_labels))
            except (ValueError, IndexError):
                print("잘못된 입력입니다. 기존 레이블만 유지합니다.")
                final_labels = [current_label]
        
        converted_data.append({
            "text": text,
            "labels": final_labels
        })
        
        print(f"최종 레이블: {', '.join(final_labels)}")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n변환 완료: {len(converted_data)}개 데이터")
    print(f"출력 파일: {output_file}")

def analyze_potential_multilabels(input_file: str):
    """
    다중 레이블 가능성을 분석하는 함수
    """
    potential_multi = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                data = json.loads(line.strip())
                text = data["text"]
                current_label = data["label"]
                
                suggested_labels = suggest_multi_labels(text, current_label)
                if len(suggested_labels) > 1:
                    potential_multi.append({
                        "line": line_num,
                        "text": text,
                        "current": current_label,
                        "suggested": suggested_labels
                    })
    
    print(f"\n=== 다중 레이블 가능성 분석 ===")
    print(f"다중 레이블 가능 데이터: {len(potential_multi)}개")
    
    for item in potential_multi[:10]:  # 처음 10개만 표시
        print(f"\n라인 {item['line']}: {item['text']}")
        print(f"현재: {item['current']} → 제안: {', '.join(item['suggested'])}")
    
    if len(potential_multi) > 10:
        print(f"\n... 총 {len(potential_multi)}개 중 10개만 표시")
    
    return potential_multi

if __name__ == "__main__":
    input_file = "../data/data.jsonl"
    output_file = "../data/data_multilabel.jsonl"
    
    print("다중 레이블 변환 도구")
    print("1. 자동 변환 (기존 레이블만 유지)")
    print("2. 대화형 변환 (수동으로 추가 레이블 선택)")
    print("3. 다중 레이블 가능성 분석")
    
    choice = input("선택하세요 (1-3): ").strip()
    
    if choice == "1":
        convert_single_to_multi_label(input_file, output_file)
    elif choice == "2":
        interactive_labeling(input_file, output_file)
    elif choice == "3":
        analyze_potential_multilabels(input_file)
    else:
        print("잘못된 선택입니다.")