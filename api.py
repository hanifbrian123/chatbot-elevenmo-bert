# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from model import generate_answer   # fungsi utama chatbot
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# Setup FastAPI App
# ============================================================
app = FastAPI(
    title="Chatbot Distributed System API",
    description="API untuk chatbot intent classification + video retrieval",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # untuk dev, buka semua origin
    allow_credentials=True,
    allow_methods=["*"],   # penting agar OPTIONS diizinkan
    allow_headers=["*"],
)


# ============================================================
# Logger
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("chatbot_api")


# ============================================================
# Pydantic Models (Request Body)
# ============================================================
class ChatRequest(BaseModel):
    text: str


# ============================================================
# Root health check
# ============================================================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Chatbot API berjalan"}


# ============================================================
# Chat endpoint
# ============================================================
@app.post("/chat")
def chat(req: ChatRequest):
    """
    Endpoint utama untuk chatbot.
    Menerima input user → mengembalikan jawaban dari generate_answer().
    """
    start = time.time()

    # Validasi input
    user_text = req.text.strip()
    if len(user_text) == 0:
        raise HTTPException(status_code=400, detail="Teks input tidak boleh kosong")

    try:
        response = generate_answer(user_text)
    except Exception as e:
        logger.exception("Error saat generate_answer")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada model")

    latency = round((time.time() - start) * 1000, 2)

    # Logging
    logger.info(f"UserInput='{user_text[:50]}' | Latency={latency}ms")

    return {
        "user_query": user_text,
        "response": response,
        "latency_ms": latency
    }
