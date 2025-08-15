
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from utils import text_standardize_series, coerce_datetime, iqr_outlier_mask

class CleaningReport:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.rows_removed = 0
        self.cols_encoded: List[str] = []
        self.cols_scaled: List[str] = []

    def add(self, action: str, detail: Dict[str, Any] = None):
        self.steps.append({"action": action, "detail": detail or {}})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "rows_removed": self.rows_removed,
            "cols_encoded": self.cols_encoded,
            "cols_scaled": self.cols_scaled,
        }

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = CleaningReport()

    # --------- Basic Cleaning ---------
    def drop_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        self.report.rows_removed += removed
        self.report.add("drop_duplicates", {"removed": removed})

    def standardize_text_cols(self):
        obj_cols = self.df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            self.df[c] = text_standardize_series(self.df[c])
        self.report.add("standardize_text", {"columns": list(obj_cols)})

    def coerce_types(self):
        before_dtypes = self.df.dtypes.astype(str).to_dict()
        self.df = coerce_datetime(self.df)
        after_dtypes = self.df.dtypes.astype(str).to_dict()
        self.report.add("coerce_types", {"before": before_dtypes, "after": after_dtypes})

    # --------- Missing Values ---------
    def impute_missing(self, num_strategy="median", cat_strategy="most_frequent"):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(include=["object"]).columns

        if len(num_cols) > 0:
            if num_strategy == "median":
                self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
            elif num_strategy == "mean":
                self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].mean())
            else:
                self.df[num_cols] = self.df[num_cols].fillna(0)

        if len(cat_cols) > 0:
            if cat_strategy == "most_frequent":
                mode_vals = {c: self.df[c].mode(dropna=True)[0] if not self.df[c].mode(dropna=True).empty else "" for c in cat_cols}
                self.df[cat_cols] = self.df[cat_cols].fillna(mode_vals)
            else:
                self.df[cat_cols] = self.df[cat_cols].fillna("missing")

        self.report.add("impute_missing", {"num_strategy": num_strategy, "cat_strategy": cat_strategy})

    # --------- Outliers ---------
    def remove_outliers_iqr(self, cols: Optional[List[str]] = None, k: float = 1.5):
        if cols is None:
            cols = list(self.df.select_dtypes(include=[np.number]).columns)
        mask = pd.Series(False, index=self.df.index)
        for c in cols:
            series = self.df[c].dropna()
            if series.empty:
                continue
            out = iqr_outlier_mask(series, k=k)
            # align to df index
            out_full = pd.Series(False, index=self.df.index)
            out_full.loc[series.index] = out
            mask = mask | out_full
        removed = int(mask.sum())
        self.df = self.df[~mask]
        self.report.rows_removed += removed
        self.report.add("remove_outliers_iqr", {"cols": cols, "k": k, "removed": removed})

    # --------- Encoding & Scaling ---------
    def one_hot_encode(self, drop_first=True, max_unique=50):
        cat_cols = [c for c in self.df.select_dtypes(include=["object"]).columns if self.df[c].nunique() <= max_unique]
        before_cols = set(self.df.columns)
        self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=drop_first)
        added = list(set(self.df.columns) - before_cols)
        self.report.cols_encoded.extend(cat_cols)
        self.report.add("one_hot_encode", {"encoded_cols": cat_cols, "new_cols": added})

    def scale_numeric(self):
        num_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            return
        scaler = StandardScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        self.report.cols_scaled.extend(num_cols)
        self.report.add("scale_numeric", {"scaled_cols": num_cols})

