import random
import os
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
import nlpaug.augmenter.word as naw
from gensim.models import KeyedVectors
from huggingface_hub import login

# 환경변수에서 토큰 읽기
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(hf_token)
else:
    print("Warning: HUGGINGFACE_TOKEN environment variable not set")

# ----------------------
# 1. Back Translation (ko<->en)
# ----------------------
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


def translate(text, src_lang="ko", tgt_lang="en"):
    m2m_tokenizer.src_lang = src_lang
    encoded = m2m_tokenizer(text, return_tensors="pt")
    generated = m2m_model.generate(
        **encoded,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(tgt_lang)
    )
    return m2m_tokenizer.decode(generated[0], skip_special_tokens=True)

def back_translate(text):
    en_text = translate(text, src_lang="ko", tgt_lang="en")
    ko_text = translate(en_text, src_lang="en", tgt_lang="ko")
    return ko_text

# ----------------------
# 2. Paraphrasing (KoBART Paraphrase)
# ----------------------
paraphraser = pipeline("text2text-generation", model="paust/pko-t5-base")

def paraphrase(text, num=2):
    results = paraphraser(
        f"paraphrase: {text}",   # T5는 task prefix 필요
        num_return_sequences=num,
        max_length=64
    )
    return [r["generated_text"] for r in results]

# ----------------------
# 3. Synonym Replacement (Word2Vec 기반)
# ----------------------
aug = naw.ContextualWordEmbsAug(
    model_path="bert-base-multilingual-cased",  # 한국어 지원
    action="substitute"
)

def synonym_aug(text):
    return aug.augment(text)

# ----------------------
# 전체 파이프라인
# ----------------------
def augment_dataset(sentences):
    augmented = []
    for idx, s in enumerate(sentences, 1):
        try:
            bt1 = back_translate(s)
            bt2 = back_translate(s)
            pps = paraphrase(s, num=2)
            syn = synonym_aug(s)

            augmented.append(s)        # 원문
            augmented.append(bt1)
            augmented.append(bt2)
            augmented.extend(pps)
            augmented.append(syn)

            print(f"[{idx}/{len(sentences)}] 완료")
        except Exception as e:
            print("Error on:", s, e)
    return augmented

# ----------------------
# 실행부
# ----------------------
if __name__ == "__main__":
    with open("data.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    augmented = augment_dataset(sentences)

    print("총 데이터 개수:", len(augmented))
    with open("augmented_data.txt", "w", encoding="utf-8") as f:
        for line in augmented:
            if isinstance(line, list):        # augment 결과가 list면
                for sub in line:
                    f.write(str(sub) + "\n")
            else:
                f.write(str(line) + "\n")
