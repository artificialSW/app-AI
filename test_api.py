import requests
import json

# API 테스트
BASE_URL = "http://localhost:8000"

def test_health():
    """헬스 체크 테스트"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("=== Health Check ===")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_single_label(text):
    """단일 라벨 예측 테스트"""
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        print(f"\n=== Single Label: {text} ===")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"Single label prediction failed: {e}")
        return False

def test_multilabel(text, threshold=0.3):
    """멀티라벨 예측 테스트"""
    try:
        response = requests.post(
            f"{BASE_URL}/predict/multilabel?threshold={threshold}",
            json={"text": text}
        )
        print(f"\n=== Multilabel (threshold={threshold}): {text} ===")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"Multilabel prediction failed: {e}")
        return False

if __name__ == "__main__":
    # 테스트 케이스
    test_texts = [
        "오늘 가족과 함께 행복한 시간을 보냈고, 어릴 때 추억도 생각났어",
        "힘들 때 위로해줘서 정말 고마워",
        "우리 결혼기념일에 뭐 하고 싶어?",
        "요즘 취미로 책 읽는 게 재밌어"
    ]
    
    print("🚀 API 테스트 시작")
    
    # 헬스 체크
    if not test_health():
        print("❌ 서버가 준비되지 않았습니다.")
        exit(1)
    
    # 각 텍스트에 대해 단일라벨과 멀티라벨 테스트
    for text in test_texts:
        test_single_label(text)
        test_multilabel(text, threshold=0.3)
    
    print("\n✅ 테스트 완료!")