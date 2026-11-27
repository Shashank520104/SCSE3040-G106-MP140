import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.joblib")
model = joblib.load(MODEL_PATH)

logger = logging.getLogger("mp140_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

app = FastAPI(title="MP140 Used Two-Wheeler Price API")


class BikeFeatures(BaseModel):
    age_years: float = Field(..., ge=0, le=30, description="Age of the bike in years (max 30)")
    km_driven: float = Field(..., ge=0, description="Odometer reading in km")
    engine_cc: float = Field(..., gt=0, description="Engine displacement in cc")
    owners: int = Field(..., ge=1, description="Number of previous owners")
    service_records: int = Field(..., ge=0, description="Service visits on record (must be >= 0)")
    accidents: int = Field(..., ge=0, description="Reported accidents")
    brand_tier: Literal["budget", "mid", "premium"]
    city_zone: Literal["North", "South", "Central"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: BikeFeatures):
    input_df = pd.DataFrame([features.model_dump()])
    pred = float(model.predict(input_df)[0])

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/predict",
        "brand_tier": features.brand_tier,
        "city_zone": features.city_zone,
        "prediction": round(pred, 4),
    }
    logger.info(json.dumps(log_entry))

    return {"prediction": round(pred, 4)}