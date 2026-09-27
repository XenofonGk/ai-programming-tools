"""
HTTP API around the resume classifier.

    uvicorn app:app --port 8000

Design notes, because this is a public endpoint:
  * The submitted text is classified in memory and never stored or logged.
  * Input is bounded (MAX_CHARS) so a request cannot make the model do
    unbounded work; the reverse proxy also caps the request body.
  * The model is trained on SYNTHETIC data. Every response says so, and a low
    confidence is flagged rather than hidden.
"""

import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from preprocess import clean_text

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model" / "resume_classifier.pkl"
METRICS_PATH = HERE / "model" / "metrics.json"

MAX_CHARS = 10_000
LOW_CONFIDENCE = 0.50
NOTE = "Trained on a synthetic dataset; a demo of the pipeline, not a hiring tool."

if not MODEL_PATH.exists():
    raise RuntimeError(f"No model at {MODEL_PATH}. Train it: python data/generate_dataset.py && python train.py")

model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

app = FastAPI(
    title="Resume Classifier",
    version="1.0.0",
    description="TF-IDF + logistic regression over five job families. " + NOTE,
)

# Same-origin by default. Set ALLOWED_ORIGIN (e.g. https://xgbuilds.dev) to let
# the portfolio call it from the browser.
origin = os.environ.get("ALLOWED_ORIGIN", "").strip()
if origin:
    app.add_middleware(CORSMiddleware, allow_origins=[origin], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, description="Resume text")


class PredictOut(BaseModel):
    label: str
    confidence: float
    low_confidence: bool
    probabilities: dict[str, float]
    note: str


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "healthy"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/model", summary="Training metrics and categories")
def model_info():
    return {**metrics, "note": NOTE}


@app.post("/predict", response_model=PredictOut, summary="Classify resume text")
def predict(body: PredictIn):
    cleaned = clean_text(body.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="no usable letters in that text")
    probs = model.predict_proba([cleaned])[0]
    classes = [str(c) for c in model.classes_]
    best = max(range(len(classes)), key=lambda i: probs[i])
    confidence = float(probs[best])
    return PredictOut(
        label=classes[best],
        confidence=round(confidence, 4),
        low_confidence=confidence < LOW_CONFIDENCE,
        probabilities={c: round(float(p), 4) for c, p in zip(classes, probs)},
        note=NOTE,
    )


@app.exception_handler(Exception)
async def unhandled(request, exc):  # never leak internals
    return JSONResponse(status_code=500, content={"detail": "internal error"})
