"""
main.py

FastAPI inference service for the manufactured-home PRICE model
produced by the DVC pipeline (src/model_training.py -> models/model.pkl).

The model was trained on one-hot encoded categorical columns
(see src/feature_engg.py). Rather than depending on
data/features/train.csv at runtime, the exact training column order
is hardcoded below (FEATURE_COLUMNS) so the served Docker image only
needs the model artifact - not the training data - to run.

Endpoints:
    GET  /            - liveness check
    GET  /health       - readiness check (model loaded?)
    POST /predict      - predict PRICE for a single home
"""

import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from logger import get_logger  # noqa: E402
from exception import PipelineException  # noqa: E402

logger = get_logger("inference_api")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
TARGET_COLUMN = "PRICE"

# Categorical columns that get one-hot encoded, mirroring src/feature_engg.py
CATEGORICAL_COLUMNS = [
    "STATUS", "FINALDEST", "FOOTINGS", "LEASE", "LOCATION",
    "REGION", "PIERS", "SECURED", "TITLED", "SECTIONS", "BEDROOMS",
]

# Exact training feature order from data/features/train.csv (minus PRICE),
# hardcoded so the served image doesn't need the training data on disk.
FEATURE_COLUMNS: List[str] = [
    "SQFT", "SHIP_MONTH", "STATUS_2", "FINALDEST_2", "FOOTINGS_2",
    "FOOTINGS_3", "FOOTINGS_4", "FOOTINGS_5", "FOOTINGS_9", "LEASE_9",
    "LOCATION_2", "LOCATION_3", "LOCATION_4", "LOCATION_9", "REGION_2",
    "REGION_3", "REGION_4", "REGION_5", "PIERS_1", "PIERS_2", "PIERS_3",
    "PIERS_4", "PIERS_9", "SECURED_1", "SECURED_2", "SECURED_3",
    "SECURED_9", "TITLED_2", "TITLED_3", "TITLED_9", "SECTIONS_2",
    "SECTIONS_3", "BEDROOMS_3",
]

app = FastAPI(
    title="Housing Price Prediction API",
    description="Serves predictions from the RandomForestRegressor trained by the DVC pipeline.",
    version="1.0.0",
)

model = None


class PredictRequest(BaseModel):
    """
    Raw feature payload. Categorical fields are passed as their
    original Census survey codes (e.g. "REGION": 2) and are one-hot
    encoded internally to match the training schema. SQFT and
    SHIP_MONTH are numeric passthrough fields.
    """

    SQFT: float = Field(..., description="Square footage of the home")
    SHIP_MONTH: int = Field(..., ge=1, le=12, description="Shipment month, 1-12")
    STATUS: Optional[int] = None
    FINALDEST: Optional[int] = None
    FOOTINGS: Optional[int] = None
    LEASE: Optional[int] = None
    LOCATION: Optional[int] = None
    REGION: Optional[int] = None
    PIERS: Optional[int] = None
    SECURED: Optional[int] = None
    TITLED: Optional[int] = None
    SECTIONS: Optional[int] = None
    BEDROOMS: Optional[int] = None

    class Config:
        extra = "allow"  # also allow already-encoded dummy columns, e.g. "REGION_2": 1


class PredictResponse(BaseModel):
    predicted_price: float


@app.on_event("startup")
def load_model():
    """Load the pickled model once at process startup."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        logger.info("Loaded model from %s", MODEL_PATH)
    except Exception as e:
        err = PipelineException(e, sys)
        logger.error(str(err))
        raise err


def build_feature_row(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert a raw request payload into a single-row DataFrame aligned
    to the model's training columns (one-hot encoding categoricals,
    filling any unseen dummy columns with 0)."""
    try:
        row = {k: v for k, v in payload.items() if v is not None}
        df = pd.DataFrame([row])

        cat_cols_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        if cat_cols_present:
            df = pd.get_dummies(df, columns=cat_cols_present)

        # Align to training schema: add any missing dummy columns as 0,
        # drop anything the model wasn't trained on, enforce column order.
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        return df
    except Exception as e:
        raise PipelineException(e, sys)


@app.get("/")
def root():
    return {"status": "ok", "message": "Housing Price Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "unhealthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        payload = request.dict()
        X = build_feature_row(payload)
        prediction = model.predict(X)[0]
        logger.info("Prediction request: %s -> %.2f", payload, prediction)
        return PredictResponse(predicted_price=float(prediction))
    except PipelineException as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)