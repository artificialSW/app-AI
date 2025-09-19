# download_model.py
import os
from huggingface_hub import snapshot_download

MODEL_NAME = "Pataegonia/korean-family-emotion-classifier"
LOCAL_PATH = "outputs/export_model"

def download_model():
    if not os.path.exists(LOCAL_PATH):
        print("Downloading model from HuggingFace...")
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_PATH,
            repo_type="model"
        )
        print("Model downloaded successfully!")
    else:
        print("Model already exists locally.")

if __name__ == "__main__":
    download_model()