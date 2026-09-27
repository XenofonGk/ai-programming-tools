"""Contract tests for the HTTP API. Run: python data/generate_dataset.py && python train.py && pytest"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import MAX_CHARS, app  # noqa: E402

client = TestClient(app)

SW = "Built REST APIs in Python and Java, wrote unit tests, reviewed pull requests and shipped features with CI/CD."


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "healthy"}


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200 and "Resume Classifier" in r.text


def test_predict_shape():
    r = client.post("/predict", json={"text": SW})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in body["probabilities"]
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
    assert 0.0 <= body["confidence"] <= 1.0
    assert "synthetic" in body["note"].lower()


def test_gibberish_is_flagged_low_confidence_or_still_valid():
    r = client.post("/predict", json={"text": "zzzz qqqq xxxx"})
    assert r.status_code == 200
    body = r.json()
    assert body["low_confidence"] is (body["confidence"] < 0.5)


def test_rejects_empty_and_oversized():
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "a" * (MAX_CHARS + 1)}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_rejects_text_with_no_letters():
    assert client.post("/predict", json={"text": "12345 !!! 678"}).status_code == 422


def test_model_info_and_headers():
    r = client.get("/model")
    assert r.status_code == 200 and "categories" in r.json()
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_docs_available():
    assert client.get("/docs").status_code == 200
