# model.py
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import random
import os

# ============================================================
# 1. Konfigurasi DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. Path model & file data
#    -> SESUAIKAN dengan nama folder Anda
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_INTENT = os.path.join(BASE_DIR, "data/intent-model")
MODEL_PATH_PRODUCT = os.path.join(BASE_DIR, "data/product-model")

KB_TEMPLATE_PATH = os.path.join(BASE_DIR, "data/kb-templates.json")
KB_VIDEO_PATH = os.path.join(BASE_DIR, "data/kb-video.json")

# ============================================================
# 3. Load Intent Classifier
# ============================================================
intent_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_INTENT)
intent_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH_INTENT
).to(DEVICE)

# ============================================================
# 4. Load Product Classifier
# ============================================================
product_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_PRODUCT)
product_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH_PRODUCT
).to(DEVICE)

# ============================================================
# 5. Load KB Templates
# ============================================================
with open(KB_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    KB = json.load(f)

# ============================================================
# 6. Load SBERT + Video DB
# ============================================================
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open(KB_VIDEO_PATH, "r", encoding="utf-8") as f:
    VIDEO_DB = json.load(f)


# ============================================================
# 7. Prediction Functions
# ============================================================
def predict_intent(text: str) -> str:
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = intent_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    pred_id = probs.argmax().item()
    intent = intent_model.config.id2label[pred_id]
    return intent


def predict_product(text: str) -> str:
    inputs = product_tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = product_model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    pred_id = probs.argmax().item()
    product = product_model.config.id2label[pred_id]
    return product


# ============================================================
# 8. Video Retrieval (SBERT similarity)
# ============================================================
def find_best_video(product: str, user_query: str):
    # Filter video sesuai produk
    videos = [v for v in VIDEO_DB if v["product"] == product]
    if len(videos) == 0:
        return None

    query_emb = sbert.encode(user_query)

    best_score = -1
    best_video = None

    for v in videos:
        title_emb = sbert.encode(v["title"])
        score = float(util.cos_sim(query_emb, title_emb))
        if score > best_score:
            best_score = score
            best_video = v

    return best_video


# ============================================================
# 9. Intent yang membutuhkan video tutorial
# ============================================================
TUTORIAL_INTENTS = ["tanya_cara_pakai", "tanya_tutorial_fitur", "tanya_demo_fitur"]


# ============================================================
# 10. generate_answer() — fungsi utama chatbot
# ============================================================
def generate_answer(user_query: str) -> str:

    # Step 1: Prediksi intent & produk
    intent = predict_intent(user_query)
    product = predict_product(user_query)

    # Step 2: Tutorial → pakai SBERT
    if intent in TUTORIAL_INTENTS:
        video = find_best_video(product, user_query)
        if video:
            return f"Berikut tutorial yang relevan:\n{video['title']}\n{video['link']}"
        else:
            return "Maaf, belum ada video tutorial untuk produk ini."

    # Step 3: Template-based answer (jika tersedia)
    if product in KB and intent in KB[product]:
        return random.choice(KB[product][intent])

    # Step 4: Fallback
    return "Maaf, saya belum bisa memahami pertanyaan Anda."
