# server.py — FastAPI version for the Money Laundering model (robust CSV endpoint)
import os
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf  # noqa: F401
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from tensorflow import keras
import uvicorn

# ==============================
# Config & Artifacts
# ==============================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"

MODEL_PATH = Path(
    os.getenv("ML_MODEL_PATH", ARTIFACTS_DIR / "money_laundering_model.h5")
)
PREPROCESSOR_PATH = Path(
    os.getenv("ML_PREPROC_PATH", ARTIFACTS_DIR / "preprocessor.pkl")
)
FEATURE_NAMES_PATH = Path(
    os.getenv("ML_FEATURES_PATH", ARTIFACTS_DIR / "feature_names.pkl")
)

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# ==============================
# Load Artifacts Once
# ==============================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(f"Preprocessor not found at: {PREPROCESSOR_PATH}")
if not FEATURE_NAMES_PATH.exists():
    raise FileNotFoundError(f"Feature names not found at: {FEATURE_NAMES_PATH}")

MODEL = keras.models.load_model(MODEL_PATH)
PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
FEATURE_NAMES: List[str] = list(joblib.load(FEATURE_NAMES_PATH))

# ==============================
# Helpers
# ==============================
# Flexible key normalization for JSON payloads (single/batch)
ALIAS_MAP = {
    # Amount
    "amount paid": "Amount Paid",
    "amount_paid": "Amount Paid",
    "amountpaid": "Amount Paid",
    "amount": "Amount Paid",
    # Payment Currency
    "payment currency": "Payment Currency",
    "payment_currency": "Payment Currency",
    "paymentcurrency": "Payment Currency",
    "currency": "Payment Currency",
    # Optional/example
    "country": "Country",
    "country_code": "Country",
    "countrycode": "Country",
}

REQUIRED_MIN_COLUMNS = ["Amount Paid", "Payment Currency"]  # FE needs only these


def _clean_header(s: str) -> str:
    # remove BOM/NBSP, trim, collapse whitespace
    s = str(s).replace("\ufeff", "").replace("\u00a0", " ")
    s = " ".join(s.strip().split())
    return s


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_header(c) for c in df.columns]
    return df


def normalize_keys(record: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in record.items():
        k_clean = _clean_header(k)
        lk = k_clean.lower().replace(" ", "")
        out[ALIAS_MAP.get(lk, k_clean)] = v
    return out


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if "Amount Paid" not in df.columns:
        raise KeyError("Missing 'Amount Paid'")
    if "Payment Currency" not in df.columns:
        raise KeyError("Missing 'Payment Currency'")
    df = df.copy()
    df["Amount_Paid_Log"] = np.log1p(df["Amount Paid"])
    df["Payment_Currency_Unique"] = df["Payment Currency"].apply(
        lambda x: 1 if str(x).strip().lower() == "bitcoin" else 0
    )
    return df


def align_to_feature_names(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_names]


def preprocess(df: pd.DataFrame):
    """
    Keep it identical to your original Flask pipeline:
      - feature_engineering(df)
      - drop 'Is Laundering' if it exists
      - pass the *raw engineered* dataframe to preprocessor.transform
    """
    df_fe = feature_engineering(df)
    X = df_fe.drop(columns=["Is Laundering"], errors="ignore")
    X_processed = PREPROCESSOR.transform(X)
    return X_processed, df_fe


def predict_proba(X_processed: np.ndarray) -> np.ndarray:
    return MODEL.predict(X_processed, verbose=0).flatten()


def estimate_feature_contributions(
    model, X_row: np.ndarray, feature_names: List[str]
) -> Dict[str, float]:
    try:
        first_w = model.layers[0].get_weights()[0]  # (n_features, units)
        contribs = X_row * first_w.T.mean(axis=0)
        contribution_dict = dict(zip(feature_names, contribs))
    except Exception:
        contribution_dict = {fn: 0.0 for fn in feature_names}
    top_items = sorted(
        contribution_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:3]
    return dict(top_items)


def generate_natural_language_explanation(
    row: Dict[str, Any], pred: int, prob: float, top_features: Dict[str, float]
) -> str:
    explanation = []
    explanation.append(
        f"Amount Paid: {row.get('Amount Paid', 'N/A')}, "
        f"Payment Currency: {row.get('Payment Currency', 'N/A')}\n"
        f"Laundering Probability: {prob:.4f}\n"
        f"Predicted Label: {pred}\n"
        f"Top Feature Contributions:"
    )
    for feature, shap_value in top_features.items():
        explanation.append(f"  - {feature}: {shap_value:+.3f}")

    explanation.append("\nNatural Language Explanation:\n")
    if pred == 1:
        explanation.append(
            f"This transaction was classified as laundering with a high probability ({prob:.2%}).\n"
            "The model identified several risk factors:"
        )
    else:
        explanation.append(
            f"This transaction was classified as legitimate with a low probability of laundering ({prob:.2%}).\n"
            "The model found no strong indicators of suspicious activity:"
        )

    for feature, shap_value in top_features.items():
        if "Amount" in feature:
            explanation.append(
                "• The transaction amount contributed "
                + (
                    "to an increase in risk."
                    if shap_value > 0
                    else "to a decrease in risk."
                )
            )
        elif "Currency" in feature:
            explanation.append(
                "• The currency used "
                + (
                    "raised concerns due to traceability."
                    if shap_value > 0
                    else "is well-regulated/traceable, lowering risk."
                )
            )
        elif "Country" in feature:
            explanation.append(
                "• The country associated with the transaction "
                + ("raised red flags." if shap_value > 0 else "suggested legitimacy.")
            )
        elif shap_value > 0:
            explanation.append(f"• {feature} slightly increased the risk.")
        else:
            explanation.append(f"• {feature} helped reduce the risk.")
    return "\n".join(explanation)


def to_bool_label(prob: float, threshold: float = 0.5) -> int:
    return int(prob > threshold)


# ==============================
# Schemas
# ==============================
class TransactionInput(BaseModel):
    amount_paid: Optional[float] = Field(None, description="Alias for 'Amount Paid'")
    payment_currency: Optional[str] = Field(
        None, description="Alias for 'Payment Currency'"
    )
    Amount_Paid: Optional[float] = None
    Amount_Paid_Log: Optional[float] = None
    Amount_Paid__legacy: Optional[float] = None
    Amount_Paid__camel: Optional[float] = Field(None, alias="amountPaid")
    Payment_Currency: Optional[str] = None
    Payment_Currency__camel: Optional[str] = Field(None, alias="paymentCurrency")

    class Config:
        extra = "allow"
        populate_by_name = True


class PredictResponse(BaseModel):
    label: int
    probability: float


class FeatureContribution(BaseModel):
    feature: str
    value: Union[int, float, bool, str, None] = None
    contribution: float


class ExplainResponse(BaseModel):
    label: int
    probability: float
    explanation: str
    top_contributions: List[FeatureContribution]


class BatchInput(BaseModel):
    records: List[Dict[str, Any]]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class BatchExplainResponse(BaseModel):
    results: List[ExplainResponse]


# ==============================
# App
# ==============================
app = FastAPI(title="Money Laundering Detection API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "preprocessor": PREPROCESSOR_PATH.name,
        "features_count": len(FEATURE_NAMES),
        "cors_origins": CORS_ORIGINS,
    }


def _records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    normed = [normalize_keys(r) for r in records]
    df = pd.DataFrame(normed)
    df = normalize_dataframe_columns(df)
    # Ensure required fields
    for col in REQUIRED_MIN_COLUMNS:
        if col not in df.columns:
            raise HTTPException(
                status_code=400, detail=f"Missing required field '{col}'"
            )
    return df


@app.post("/api/predict", response_model=PredictResponse)
def predict_single(payload: Dict[str, Any]):
    try:
        df = _records_to_dataframe([payload])
        X_processed, df_fe = preprocess(df)
        y_prob = float(predict_proba(X_processed)[0])
        y_label = to_bool_label(y_prob)
        return PredictResponse(label=y_label, probability=y_prob)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/api/predict-batch", response_model=BatchPredictResponse)
def predict_batch(batch: BatchInput):
    try:
        df = _records_to_dataframe(batch.records)
        X_processed, df_fe = preprocess(df)
        y_probs = predict_proba(X_processed)
        results = [
            PredictResponse(label=to_bool_label(p), probability=float(p))
            for p in y_probs
        ]
        return BatchPredictResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@app.post("/api/explain", response_model=ExplainResponse)
def explain_single(payload: Dict[str, Any]):
    try:
        df = _records_to_dataframe([payload])
        X_processed, df_fe = preprocess(df)
        prob = float(predict_proba(X_processed)[0])
        label = to_bool_label(prob)

        x_row = X_processed[0]
        top_contribs = estimate_feature_contributions(MODEL, x_row, FEATURE_NAMES)

        top_list = []
        for feat, contrib in top_contribs.items():
            val = df_fe.get(feat).iloc[0] if feat in df_fe.columns else None
            top_list.append(
                FeatureContribution(
                    feature=feat,
                    value=None if (val is None or pd.isna(val)) else val,
                    contribution=float(contrib),
                )
            )
        explanation = generate_natural_language_explanation(
            df_fe.iloc[0].to_dict(), label, prob, top_contribs
        )
        return ExplainResponse(
            label=label,
            probability=prob,
            explanation=explanation,
            top_contributions=top_list,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")


@app.post("/api/explain-batch", response_model=BatchExplainResponse)
def explain_batch(batch: BatchInput):
    try:
        df = _records_to_dataframe(batch.records)
        X_processed, df_fe = preprocess(df)
        probs = predict_proba(X_processed)

        results: List[ExplainResponse] = []
        for i, p in enumerate(probs):
            lbl = to_bool_label(float(p))
            x_row = X_processed[i]
            top_contribs = estimate_feature_contributions(MODEL, x_row, FEATURE_NAMES)

            top_list = []
            for feat, contrib in top_contribs.items():
                val = df_fe.iloc[i].get(feat) if feat in df_fe.columns else None
                top_list.append(
                    FeatureContribution(
                        feature=feat,
                        value=None if (val is None or pd.isna(val)) else val,
                        contribution=float(contrib),
                    )
                )
            explanation = generate_natural_language_explanation(
                df_fe.iloc[i].to_dict(), lbl, float(p), top_contribs
            )

            results.append(
                ExplainResponse(
                    label=lbl,
                    probability=float(p),
                    explanation=explanation,
                    top_contributions=top_list,
                )
            )
        return BatchExplainResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch explanation failed: {e}")


# ==============================
# CSV upload endpoint (robust)
# ==============================
def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# Comprehensive alias map for CSV headers (normalized)
CSV_ALIAS = {
    # ---- Amount Paid ----
    "amount paid": "Amount Paid",
    "amount_paid": "Amount Paid",
    "amount": "Amount Paid",
    "amountpaid": "Amount Paid",
    # ---- Payment Currency ----
    "payment currency": "Payment Currency",
    "payment_currency": "Payment Currency",
    "paymentcurrency": "Payment Currency",
    "currency": "Payment Currency",
    # ---- Payment Format (optional) ----
    "payment format": "Payment Format",
    "payment_format": "Payment Format",
    "paymentmethod": "Payment Format",
    "payment method": "Payment Format",
    "method": "Payment Format",
    "paymethod": "Payment Format",
    # Some users export “Receiving Currency” only — not required; here just kept original if present
    "receiving currency": "Receiving Currency",
    "receiving_currency": "Receiving Currency",
}


def _normalize_and_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataframe_columns(df)
    new_cols: List[str] = []
    for c in df.columns:
        key = _clean_header(c).lower().replace(" ", "")
        new_cols.append(CSV_ALIAS.get(key, _clean_header(c)))
    df.columns = new_cols
    return df


def _ensure_minimum_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_MIN_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"columns are missing: {set(missing)}; received={list(df.columns)}",
        )
    # Optional Payment Format — if absent, create a neutral placeholder
    if "Payment Format" not in df.columns:
        df["Payment Format"] = "Unknown"
    return df


def process_dataframe_for_api(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    X_processed, df_fe = preprocess(df)
    y_probs = predict_proba(X_processed)
    y_labels = (y_probs > 0.5).astype(int)

    explanations: List[str] = []
    for i in range(len(df_fe)):
        x_row = X_processed[i]
        contribs = estimate_feature_contributions(MODEL, x_row, FEATURE_NAMES)
        row_dict = df_fe.iloc[i].to_dict()
        exp = generate_natural_language_explanation(
            row=row_dict,
            pred=int(y_labels[i]),
            prob=float(y_probs[i]),
            top_features=contribs,
        )
        explanations.append(exp)

    out = df.copy()
    if "Amount_Paid_Log" in df_fe.columns:
        out["Amount_Paid_Log"] = df_fe["Amount_Paid_Log"].values
    if "Payment_Currency_Unique" in df_fe.columns:
        out["Payment_Currency_Unique"] = df_fe["Payment_Currency_Unique"].values

    out["Laundering Probability"] = y_probs
    out["Predicted Label"] = y_labels
    out["Explanation"] = explanations
    return out


@app.post("/api/predict-csv")
async def predict_from_csv(
    file: UploadFile = File(..., description="CSV file"),
    as_csv: bool = Query(False, description="Return CSV file instead of JSON"),
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # ---- Robust header normalization ----
        # strip BOM + whitespace
        def _clean(s: str) -> str:
            return str(s).replace("\ufeff", "").strip()

        df.columns = [_clean(c) for c in df.columns]
        # build a case-insensitive lookup of present columns
        lower_to_orig = {c.lower(): c for c in df.columns}

        # Helper to copy/create a canonical column from a list of aliases
        def ensure_col(target: str, aliases: List[str]):
            if target in df.columns:
                return

            # try case-insensitive direct match
            if target.lower() in lower_to_orig:
                df[target] = df[lower_to_orig[target.lower()]]
                return

            # try any alias variants (case-insensitive)
            for a in aliases:
                if a in df.columns:
                    df[target] = df[a]
                    return
                if a.lower() in lower_to_orig:
                    df[target] = df[lower_to_orig[a.lower()]]
                    return

            # if still missing, raise a *clear* error with what we actually saw
            raise HTTPException(
                status_code=400,
                detail=(
                    f"columns are missing: {{{target!r}}}; "
                    f"seen_columns={list(df.columns)}"
                ),
            )

        # ---- Map your CSV to what the preprocessor expects ----
        # 'Amount Paid' (you may also accept 'Amount' from some exports)
        ensure_col("Amount Paid", ["Amount_Paid", "amount_paid", "Amount", "amount"])

        # 'Payment Currency' (fall back to 'Receiving Currency' if needed)
        ensure_col(
            "Payment Currency",
            [
                "Payment_Currency",
                "payment_currency",
                "Currency",
                "currency",
                "Receiving Currency",
                "receiving currency",
            ],
        )

        # 'Payment Format' (lots of real-world variants)
        ensure_col(
            "Payment Format",
            [
                "Payment_Format",
                "payment_format",
                "Payment Type",
                "payment type",
                "Payment Method",
                "payment method",
            ],
        )

        # ---- Process like your Flask app ----
        result_df = process_dataframe_for_api(df)

        if as_csv:
            csv_bytes = _df_to_csv_bytes(result_df)
            return StreamingResponse(
                io.BytesIO(csv_bytes),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="aml_results_{file.filename}"'
                },
            )
        else:
            return JSONResponse(content=result_df.to_dict(orient="records"))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {e}")


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
