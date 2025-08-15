
import streamlit as st
import pandas as pd
import numpy as np
import json
from cleaner import DataCleaner
from utils import summarize_df
from ai_helper import suggest_schema_and_steps, ai_available

st.set_page_config(page_title="AI-Assisted Data Cleaner", layout="wide")

st.title("🧹 AI-Assisted Data Cleaning Tool")
st.caption("Upload a CSV, preview AI suggestions, and download a cleaned dataset + JSON report.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

with st.expander("⚙️ Options", expanded=False):
    remove_dupes = st.checkbox("Remove Duplicates", value=True)
    standardize_text = st.checkbox("Standardize Text Columns", value=True)
    coerce_types = st.checkbox("Auto-parse Dates & Fix Types", value=True)
    impute_num = st.selectbox("Numeric Imputation", ["median", "mean", "zero"], index=0)
    impute_cat = st.selectbox("Categorical Imputation", ["most_frequent", "missing"], index=0)
    remove_outliers = st.checkbox("Remove Outliers (IQR)", value=False)
    encode = st.checkbox("One-Hot Encode Categorical", value=True)
    scale = st.checkbox("Scale Numeric", value=False)

if uploaded is not None:
    df = pd.read_csv(uploaded,encoding="ISO-8859-1")
    st.subheader("Preview")
    st.dataframe(df.head(20))

    # AI suggestion
    st.subheader("🤖 AI Suggestions")
    summary = summarize_df(df)
    suggestions = suggest_schema_and_steps(summary)
    st.json(suggestions)

    cleaner = DataCleaner(df)
    if remove_dupes: cleaner.drop_duplicates()
    if standardize_text: cleaner.standardize_text_cols()
    if coerce_types: cleaner.coerce_types()
    cleaner.impute_missing(num_strategy=impute_num, cat_strategy=impute_cat)
    if remove_outliers: cleaner.remove_outliers_iqr()
    if encode: cleaner.one_hot_encode()
    if scale: cleaner.scale_numeric()

    st.subheader("Cleaned Data (first 1,000 rows)")
    st.dataframe(cleaner.df.head(1000))

    # Downloads
    cleaned_csv = cleaner.df.to_csv(index=False).encode("utf-8")
    report_json = json.dumps(cleaner.report.to_dict(), indent=2).encode("utf-8")

    st.download_button("⬇️ Download Cleaned CSV", cleaned_csv, "cleaned_data.csv", mime="text/csv")
    st.download_button("⬇️ Download Cleaning Report (JSON)", report_json, "cleaning_report.json", mime="application/json")

else:
    st.info("Upload a CSV to begin. You can try the sample dataset below.")

# Sample data download
with st.expander("📁 Sample Dataset"):
    import io
    import numpy as np
    rng = np.random.default_rng(42)
    n = 500
    sample = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "age": rng.integers(18, 65, size=n).astype(float),
        "income": rng.normal(60000, 15000, size=n),
        "city": rng.choice(["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Pune"], size=n),
        "plan": rng.choice(["free", "basic", "pro"], size=n),
        "notes": rng.choice(["  Hello ", "WORLD!!", None, "TeSt  "], size=n)
    })
    # inject some missingness/outliers
    sample.loc[rng.choice(n, size=30, replace=False), "income"] = None
    sample.loc[rng.choice(n, size=15, replace=False), "age"] = None
    sample.loc[rng.choice(n, size=5, replace=False), "income"] *= 5

    st.dataframe(sample.head(20))
    csv_bytes = sample.to_csv(index=False).encode("utf-8")
    st.download_button("Download Sample CSV", csv_bytes, "sample_data.csv", mime="text/csv")
