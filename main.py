"""
main.py — FastAPI Sentiment Analysis API (GRU / PyTorch backend)

Endpoints:
  GET  /          → health check
  POST /predict   → sentiment + category prediction

Run with:
  uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from utils import (
    clean_text,
    load_artifacts,
    load_config,
    text_to_sequence,
    pad_sequence,
    adjust_category,
    get_vader_sentiment,
    MAX_LEN,
)

# ─── App state (loaded once at startup) ────────────────────────────────────────
_tfidf             = None
_gru_model         = None
_sentiment_encoder = None
_category_encoder  = None
_vocab             = None
_device            = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy artifacts once when the server starts."""
    global _tfidf, _gru_model, _sentiment_encoder, _category_encoder, _vocab, _device

    config = load_config()  # noqa: F841
    _tfidf, _gru_model, _sentiment_encoder, _category_encoder, _vocab = load_artifacts()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _gru_model.to(_device)
    print(f"✅  Models loaded successfully (device: {_device}).")
    yield
    print("🛑  Server shutting down.")


# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Sentiment Analysis API",
    description=(
        "Predict the **sentiment** (Positive / Negative / Neutral) and "
        "**category** (Product / Billing / Shipping) of customer reviews "
        "using a Bidirectional GRU model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text field must not be empty or whitespace only.")
        return v


class PredictResponse(BaseModel):
    input_text:            str
    clean_text:            str
    sentiment:             str
    category:              str
    raw_model_category:    str
    sentiment_confidence:  Optional[float] = None
    category_confidence:   Optional[float] = None


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health-check endpoint."""
    return {"status": "ok", "message": "Sentiment Analysis API is running 🚀"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict sentiment and category for a customer review.

    - **text**: raw customer review string
    """
    if _gru_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet. Try again shortly.")

    # 1. Clean text
    cleaned = clean_text(request.text)
    if not cleaned.strip():
        raise HTTPException(
            status_code=422,
            detail="After cleaning, the input text is empty. Please provide a meaningful review.",
        )

    # 2. Convert to sequence and pad
    seq    = text_to_sequence(cleaned, _vocab)
    seq    = pad_sequence(seq, MAX_LEN)
    tensor = torch.tensor([seq], dtype=torch.long).to(_device)

    # 3. GRU inference
    with torch.no_grad():
        sent_out, cat_out = _gru_model(tensor)

    # 4. Probabilities
    sent_probs = F.softmax(sent_out, dim=1)
    cat_probs  = F.softmax(cat_out,  dim=1)

    sent_idx = int(torch.argmax(sent_probs, dim=1).item())
    cat_idx  = int(torch.argmax(cat_probs,  dim=1).item())

    sentiment            = _sentiment_encoder.inverse_transform([sent_idx])[0]
    raw_model_category   = _category_encoder.inverse_transform([cat_idx])[0]
    category             = adjust_category(request.text, raw_model_category)

    sent_conf = round(float(torch.max(sent_probs).item()), 4)
    cat_conf  = round(float(torch.max(cat_probs).item()),  4)

    # Override sentiment when keywords / negation indicate clear signal
    sentiment = get_vader_sentiment(request.text)

    return PredictResponse(
        input_text=request.text,
        clean_text=cleaned,
        sentiment=sentiment,
        category=category,
        raw_model_category=raw_model_category,
        sentiment_confidence=sent_conf,
        category_confidence=cat_conf,
    )
