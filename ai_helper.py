
import os, json
from typing import Dict, Any, List

def ai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

def suggest_schema_and_steps(df_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses OpenAI if available. Otherwise, returns a reasonable heuristic suggestion.
    """
    if not ai_available():
        # Heuristic fallback
        dtypes = df_summary.get("dtypes", {})
        suggestion = {"columns": {}, "global": {"notes": "Heuristic suggestion (no API key)."}}
        for col, dt in dtypes.items():
            rec = {"type": None, "impute": None, "encode": None, "scale": None}
            if "float" in dt or "int" in dt:
                rec["type"] = "numeric"
                rec["impute"] = "median"
                rec["scale"] = "standard"
            elif "datetime" in dt:
                rec["type"] = "datetime"
                rec["impute"] = "drop-row-if-critical"
            else:
                rec["type"] = "categorical"
                rec["impute"] = "most_frequent"
                rec["encode"] = "onehot"
            suggestion["columns"][col] = rec
        return suggestion

    # If API available
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = (
            "You are a data preprocessing assistant. Given this dataset summary, "
            "recommend types and cleaning steps for each column. Use json with keys: "
            "{columns: {col: {type, impute, encode, scale}}, global: {notes}}.\n\n"
            f"DATA SUMMARY:\n{json.dumps(df_summary, ensure_ascii=False)}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Be concise and return strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        # Try to parse JSON; if the model added text, extract the JSON portion
        start = content.find("{")
        end = content.rfind("}")
        json_str = content[start:end+1] if start != -1 and end != -1 else "{}"
        data = json.loads(json_str)
        return data
    except Exception as e:
        # Fallback if API fails
        return {"error": str(e), "fallback": True}
