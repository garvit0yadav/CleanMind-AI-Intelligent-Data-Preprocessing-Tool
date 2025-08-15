
import pandas as pd
import numpy as np
import re
from typing import Dict, Any

def coerce_datetime(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True, utc=False)
                # Only convert if a reasonable fraction are valid dates
                valid_ratio = parsed.notna().mean()
                if valid_ratio > 0.8:
                    df[col] = parsed
            except Exception:
                pass
    return df

def text_standardize_series(s: pd.Series) -> pd.Series:
    # Lower, strip, collapse spaces, remove leading/trailing punctuation
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"^[^\w]+|[^\w]+$", "", regex=True)
    return s

def summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "null_counts": df.isna().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records")
    }
    return info

def iqr_outlier_mask(series: pd.Series, k: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

