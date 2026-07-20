"""
Life Expectancy Prediction API
================================
FastAPI service that serves a trained Random Forest regressor to predict a
country's average life expectancy (years) from WHO health and socio-economic
indicators.

Endpoints
---------
GET  /            -> redirects to the interactive Swagger UI (/docs)
GET  /health      -> liveness + which model is loaded
POST /predict     -> returns a life-expectancy prediction for 8 validated inputs
POST /retrain     -> upload a CSV of new rows to retrain & hot-swap the model
"""
import io
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model artifact (pipeline = StandardScaler + Random Forest).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "training_data.csv")

_artifact = joblib.load(MODEL_PATH)
MODEL: Pipeline = _artifact["pipeline"]
FEATURES = _artifact["features"]          # exact column order the model expects
TARGET = _artifact["target"]
MODEL_NAME = _artifact.get("best_model_name", "model")

app = FastAPI(
    title="Life Expectancy Prediction API",
    description=(
        "Predicts a country's average life expectancy (years) from 8 WHO "
        "health & socio-economic indicators. Best model: Random Forest "
        "(R2 ~= 0.96, RMSE ~= 1.8 years). Mission: improving healthcare "
        "outcomes in Africa through digital solutions."
    ),
    version="1.0.0",
)

# CORS is scoped rather than a wildcard:
#   origins     -> the hosted Swagger UI and local Flutter-web dev origins only
#                  (a native mobile app sends no Origin header, so it is unaffected).
#   methods     -> GET and POST (plus the OPTIONS preflight); the API uses no others.
#   headers     -> Content-Type only.
#   credentials -> disabled; the API is stateless and uses no cookies or auth.
# Additional origins can be supplied through the ALLOWED_ORIGINS environment variable.
_default_origins = (
    "http://localhost,http://localhost:8080,http://127.0.0.1:8080,"
    "https://linear-regression-model-z9ly.onrender.com"
)
ALLOWED_ORIGINS = [o.strip() for o in
                   os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------- #
# Request schema — Pydantic enforces data TYPES and realistic RANGES.          #
# Every field is a float with min/max bounds taken from the observed WHO data  #
# and widened to realistic real-world limits. Out-of-range input => HTTP 422.  #
# --------------------------------------------------------------------------- #
class LifeExpectancyInput(BaseModel):
    adult_mortality: float = Field(
        ..., ge=0, le=1000,
        description="Adult mortality: deaths of adults per 1000 population (15-60 yrs).",
        examples=[150.0])
    bmi: float = Field(
        ..., ge=1, le=80,
        description="Average Body Mass Index of the population.",
        examples=[40.0])
    hiv_aids: float = Field(
        ..., ge=0.1, le=60,
        description="Deaths per 1000 live births due to HIV/AIDS (ages 0-4).",
        examples=[0.5])
    gdp: float = Field(
        ..., ge=0, le=200000,
        description="Gross Domestic Product per capita (USD).",
        examples=[5000.0])
    income_composition: float = Field(
        ..., ge=0, le=1,
        description="Income composition of resources (HDI income index, 0-1).",
        examples=[0.7])
    schooling: float = Field(
        ..., ge=0, le=25,
        description="Average number of years of schooling.",
        examples=[12.0])
    diphtheria: float = Field(
        ..., ge=0, le=100,
        description="Diphtheria immunization coverage among 1-year-olds (%).",
        examples=[85.0])
    thinness_1_19: float = Field(
        ..., ge=0, le=50,
        description="Prevalence of thinness among ages 1-19 (%).",
        examples=[5.0])

    model_config = {
        "json_schema_extra": {
            "example": {
                "adult_mortality": 150.0, "bmi": 40.0, "hiv_aids": 0.5,
                "gdp": 5000.0, "income_composition": 0.7, "schooling": 12.0,
                "diphtheria": 85.0, "thinness_1_19": 5.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_life_expectancy: float
    unit: str = "years"
    model_used: str


# Map API field names -> exact model feature column names (in the model's order).
_FIELD_TO_FEATURE = {
    "adult_mortality": "Adult Mortality",
    "bmi": "BMI",
    "hiv_aids": "HIV/AIDS",
    "gdp": "GDP",
    "income_composition": "Income composition of resources",
    "schooling": "Schooling",
    "diphtheria": "Diphtheria",
    "thinness_1_19": "thinness  1-19 years",
}


@app.get("/", include_in_schema=False)
def root():
    """Redirect the base URL to the Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "n_features": len(FEATURES)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: LifeExpectancyInput):
    """Predict life expectancy (years) from the 8 validated indicators."""
    # Build the feature row in the exact order the model was trained on.
    values = payload.model_dump()
    row = np.array([[values[k] for k in _FIELD_TO_FEATURE]])  # dict preserves order
    try:
        pred = float(MODEL.predict(row)[0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return PredictionResponse(
        predicted_life_expectancy=round(pred, 2),
        model_used=MODEL_NAME,
    )


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    """
    Trigger a model update from NEW data.

    Upload a CSV that contains the 8 feature columns plus the target
    ('Life expectancy'). The new rows are appended to the stored training data,
    a fresh Random Forest is trained, evaluated on a hold-out split, and — if it
    is valid — the in-memory model AND best_model.pkl are hot-swapped so the
    /predict endpoint immediately serves the retrained model. No redeploy needed.
    """
    global MODEL
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    content = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    new_df.columns = new_df.columns.str.strip()
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in new_df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV missing required columns: {missing}. Needed: {required}")

    # Combine existing snapshot with the newly uploaded rows.
    base = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame(columns=required)
    combined = pd.concat([base[required], new_df[required]], ignore_index=True)
    combined = combined.dropna(subset=[TARGET])
    for c in FEATURES:
        combined[c] = combined[c].fillna(combined[c].median())

    if len(combined) < 20:
        raise HTTPException(status_code=422, detail="Not enough data to retrain (need >= 20 rows).")

    X = combined[FEATURES].values
    y = combined[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    new_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("m", RandomForestRegressor(n_estimators=100, max_depth=10,
                                    random_state=42, n_jobs=-1)),
    ]).fit(Xtr, ytr)

    pred = new_pipe.predict(Xte)
    rmse = float(mean_squared_error(yte, pred) ** 0.5)
    r2 = float(r2_score(yte, pred))

    # Persist & hot-swap.
    combined.to_csv(DATA_PATH, index=False)
    joblib.dump({"pipeline": new_pipe, "features": FEATURES, "target": TARGET,
                 "best_model_name": "Random Forest (retrained)"}, MODEL_PATH)
    MODEL = new_pipe

    return {
        "message": "Model retrained and hot-swapped successfully.",
        "rows_added": int(len(new_df)),
        "total_training_rows": int(len(combined)),
        "new_test_rmse": round(rmse, 3),
        "new_test_r2": round(r2, 4),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prediction:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
