"""FastAPI application that serves the saved Titanic survival model."""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import ml.features  # noqa: F401  (the saved pipeline refers to ml.features.add_features)
from app.schemas import BatchIn, BatchOut, PassengerIn, PredictionOut

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "titanic_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # the model is loaded ONCE when the server starts, not on every request
    app.state.model, app.state.meta = None, {}
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        app.state.meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    yield


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="End-to-end machine learning project: a scikit-learn pipeline served with FastAPI. "
                "For learning purposes only.",
    version="1.0.0", lifespan=lifespan)


def _get_model():
    if app.state.model is None:
        raise HTTPException(503, "Model file not found. Run `python -m ml.train` first.")
    return app.state.model


def _to_frame(items: list[PassengerIn]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Pclass": p.pclass, "Sex": p.sex, "Age": np.nan if p.age is None else p.age, "SibSp": p.sibsp,
        "Parch": p.parch, "Fare": p.fare, "Embarked": p.embarked, "HasCabin": int(p.has_cabin)} for p in items])


def _to_output(prob: float) -> PredictionOut:
    survived = prob >= 0.5
    margin = abs(prob - 0.5)
    return PredictionOut(
        survived=bool(survived), survival_probability=round(float(prob), 4),
        prediction="Survived" if survived else "Did not survive",
        confidence="high" if margin >= 0.3 else "medium" if margin >= 0.15 else "low",
        model=app.state.meta.get("model_name", "unknown"))


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.get("/model-info", tags=["system"])
def model_info():
    _get_model()
    m = app.state.meta
    return {k: m.get(k) for k in ("model_name", "trained_at", "sklearn_version", "features", "n_train", "n_test", "test_metrics")}


@app.post("/predict", response_model=PredictionOut, tags=["prediction"])
def predict(passenger: PassengerIn):
    prob = _get_model().predict_proba(_to_frame([passenger]))[0, 1]
    return _to_output(prob)


@app.post("/predict/batch", response_model=BatchOut, tags=["prediction"])
def predict_batch(batch: BatchIn):
    probs = _get_model().predict_proba(_to_frame(batch.passengers))[:, 1]
    return BatchOut(count=len(probs), predictions=[_to_output(p) for p in probs])
