# compare_3_excels_accuracy.py
# pip install pandas openpyxl

import math
import re
from typing import Dict, List, Tuple
import pandas as pd


# =========================
# PATHS
# =========================
PAIRS = [
    ("trade",
     r"C:\Users\quang\Desktop\trade.xlsx",
     r"C:\AIRC-Invoice\broker-extraction-api\outputs_cpu\trade.xlsx"),
    ("other",
     r"C:\Users\quang\Desktop\other.xlsx",
     r"C:\AIRC-Invoice\broker-extraction-api\outputs_cpu\other.xlsx"),
    ("fx_tf",
     r"C:\Users\quang\Desktop\fx_tf.xlsx",
     r"C:\AIRC-Invoice\broker-extraction-api\outputs_cpu\fx_tf.xlsx"),
]

# =========================
# KEY CONFIG (your keys)
# =========================
KEY_COLS: Dict[str, List[str]] = {
    "trade": ["Name/ Security", "Securities ID", "Quantity"],
    "other": ["Description", "Foreign Gross Amount/Interest", "Foreign Net Amount"],
    "fx_tf": ["Rate", "Account no. Buy", "Account no. Sell"],
}

# numeric tolerance
ABS_TOL = 1e-6
REL_TOL = 1e-6


# =========================
# Helpers
# =========================
def is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def normalize_value(x):
    """Normalize for stable comparisons (no fuzzy matching)."""
    if is_empty(x):
        return None

    if isinstance(x, pd.Timestamp):
        if pd.isna(x):
            return None
        return x.isoformat()

    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)

    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)

    # parse datetime
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if not pd.isna(dt):
        return dt.isoformat()

    # parse numeric (keep '-')
    s_num = s.replace(",", "")
    # (123.45) => -123.45
    m_paren = re.fullmatch(r"\(([-+]?\d+(\.\d+)?)\)", s_num)
    if m_paren:
        try:
            return -float(m_paren.group(1))
        except Exception:
            pass

    try:
        return float(s_num)
    except Exception:
        return s


def values_equal(a, b) -> bool:
    a_n = normalize_value(a)
    b_n = normalize_value(b)

    if a_n is None and b_n is None:
        return True
    if a_n is None or b_n is None:
        return False

    if isinstance(a_n, float) and isinstance(b_n, float):
        diff = abs(a_n - b_n)
        if diff <= ABS_TOL:
            return True
        denom = max(abs(a_n), abs(b_n), 1.0)
        return (diff / denom) <= REL_TOL

    return a_n == b_n


def read_first_sheet(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, dtype=object, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_key(df: pd.DataFrame, key_cols: List[str]) -> pd.Series:
    for c in key_cols:
        if c not in df.columns:
            raise ValueError(f"Key column '{c}' not found. Columns: {df.columns.tolist()}")
    key_df = df[key_cols].astype(str).applymap(lambda x: x.strip())
    return key_df.apply(lambda r: "||".join(r.values), axis=1)


def align_union_by_key(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    key_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Align by UNION keys:
      - missing key in pred => pred row becomes NaN => accuracy penalized
      - extra key in pred   => gold row becomes NaN => extra info penalized
    """
    g = gold.copy()
    p = pred.copy()

    g["_KEY_"] = build_key(g, key_cols)
    p["_KEY_"] = build_key(p, key_cols)

    # Handle duplicate keys: keep first occurrence per key
    g = g.drop_duplicates(subset=["_KEY_"], keep="first").set_index("_KEY_").sort_index()
    p = p.drop_duplicates(subset=["_KEY_"], keep="first").set_index("_KEY_").sort_index()

    gold_keys = set(g.index.tolist())
    pred_keys = set(p.index.tolist())
    union_keys = sorted(list(gold_keys.union(pred_keys)))

    stats = {
        "gold_rows": len(gold),
        "pred_rows": len(pred),
        "gold_unique_keys": len(gold_keys),
        "pred_unique_keys": len(pred_keys),
        "missing_in_pred_keys": len(gold_keys - pred_keys),
        "extra_in_pred_keys": len(pred_keys - gold_keys),
        "union_keys": len(union_keys),
    }

    g2 = g.reindex(union_keys)
    p2 = p.reindex(union_keys)

    # unify columns
    all_cols = sorted(set(g2.columns).union(set(p2.columns)))
    for c in all_cols:
        if c not in g2.columns:
            g2[c] = None
        if c not in p2.columns:
            p2[c] = None

    g2 = g2[all_cols]
    p2 = p2[all_cols]

    return g2, p2, stats


def compute_accuracy_table(gold_u: pd.DataFrame, pred_u: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accuracy computed on ALL union keys × ALL columns.
    Rules:
      - gold empty & pred empty => correct
      - gold empty & pred filled => incorrect (extra info)
      - gold filled & pred empty => incorrect (missing info)
      - both filled => compare value => correct/incorrect
    """
    cols = list(gold_u.columns)
    n_rows = len(gold_u)
    n_cols = len(cols)
    total_cells = n_rows * n_cols if n_rows and n_cols else 0

    per_col = []
    total_correct = 0

    for c in cols:
        g = gold_u[c]
        p = pred_u[c]
        correct = sum(values_equal(g.iloc[i], p.iloc[i]) for i in range(n_rows))
        acc = (correct / n_rows * 100.0) if n_rows else 0.0
        per_col.append({"Field": c, "Accuracy_%": round(acc, 2), "Correct": int(correct), "Total_rows_union": int(n_rows)})
        total_correct += correct

    overall_acc = (total_correct / total_cells * 100.0) if total_cells else 0.0

    # Summary row with many accuracy columns (as requested)
    summary = {"overall_accuracy_%": round(overall_acc, 2)}
    for row in per_col:
        summary[f"{row['Field']}_acc_%"] = row["Accuracy_%"]

    summary_df = pd.DataFrame([summary])
    per_col_df = pd.DataFrame(per_col).sort_values("Field")

    return summary_df, per_col_df


def main():
    for name, gold_path, pred_path in PAIRS:
        key_cols = KEY_COLS[name]

        gold = read_first_sheet(gold_path)
        pred = read_first_sheet(pred_path)

        gold_u, pred_u, stats = align_union_by_key(gold, pred, key_cols)
        summary_df, per_col_df = compute_accuracy_table(gold_u, pred_u)

        # Add key stats into summary
        summary_df["file"] = name
        summary_df["key_cols"] = ", ".join(key_cols)
        for k, v in stats.items():
            summary_df[k] = v

        out_path = f"{name}_accuracy_report.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            summary_df.to_excel(w, index=False, sheet_name="summary")
            per_col_df.to_excel(w, index=False, sheet_name="per_column")

        print(
            f"✅ {out_path} | overall_accuracy={summary_df.loc[0,'overall_accuracy_%']}% | "
            f"missing_keys={stats['missing_in_pred_keys']} | extra_keys={stats['extra_in_pred_keys']}"
        )


if __name__ == "__main__":
    main()
