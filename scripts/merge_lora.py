# scripts/merge_lora.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

BASE = "klue/bert-base"
LORA_DIR = "outputs/lora_ckpt"
EXPORT = "outputs/export_model"

base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=6)
tok = AutoTokenizer.from_pretrained(BASE)
lora = PeftModel.from_pretrained(base, LORA_DIR)
merged = lora.merge_and_unload()

merged.save_pretrained(EXPORT)
tok.save_pretrained(EXPORT)
print("merged model saved to", EXPORT)
