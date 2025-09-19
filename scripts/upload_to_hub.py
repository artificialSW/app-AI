# scripts/upload_to_hub.py
from huggingface_hub import HfApi, create_repo
import os

# 설정
MODEL_NAME = "Pataegonia/korean-family-emotion-classifier"
MODEL_PATH = "outputs/export_model"

def upload_model():
    api = HfApi()
    
    # 1. 저장소 생성 (이미 존재하면 무시됨)
    try:
        create_repo(MODEL_NAME, exist_ok=True)
        print(f"Repository {MODEL_NAME} created/verified")
    except Exception as e:
        print(f"Repository creation error: {e}")
    
    # 2. 모델 파일들 업로드
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=MODEL_NAME,
        repo_type="model"
    )
    print(f"Model uploaded to https://huggingface.co/{MODEL_NAME}")

if __name__ == "__main__":
    upload_model()