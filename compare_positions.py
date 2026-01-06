import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# CONFIG: update your paths
# =========================
GOLD_PATH = r"C:\Users\quang\Desktop\positions.xlsx"
FILE_A_PATH = r"C:\Users\quang\Downloads\untitled_128\postion.xlsx"
FILE_B_PATH = r"C:\AIRC-Invoice\broker-extraction-api\outputs_cpu\position.xlsx"

# Nếu bạn chạy trong môi trường có file upload (như trong chat này) thì có thể dùng:
# GOLD_PATH = r"/mnt/data/positions.xlsx"
# FILE_A_PATH = r"/mnt/data/postion.xlsx"
# FILE_B_PATH = r"/mnt/data/position.xlsx"

OUT_REPORT = "compare_report.xlsx"

# Key logic: OR-key (ưu tiên Security ID, fallback Account)
SECURITY_ID_COL = "Security ID"   # nếu tên cột khác, thêm alias bên dưới
ACCOUNT_COL = "Account"           # hoặc "Account No" tuỳ file

# Ngưỡng so text (0..100). 100 = y hệt.
TEXT_MATCH_THRESHOLD = 92

# Sai số tương đối khi so số (0.2%).
NUM_REL_TOL = 0.002


# =========================
# Column aliases (rename)
# =========================
COLUMN_ALIASES: Dict[str, List[str]] = {
    "Security ID": ["Security ID", "SecurityID", "Instrument ID", "ID", "Security Id"],
    "Account": ["Account", "Account No", "Acct", "Account number", "AccountNumber", "Account No."],
    "ISIN": ["ISIN", "Isin"],
    "Portfolio No.": ["Portfolio No.", "Portfolio", "Portfolio Number", "Portfolio No"],
    "Currency": ["Currency", "CCY", "Curr"],
    "Quantity/ Amount": ["Quantity/ Amount", "Quantity", "Qty", "Amount", "Units", "Nominal"],
    "MarketPrice": ["MarketPrice", "Market Price", "Price", "MktPrice"],
    "MarketValue": ["MarketValue", "Market Value", "Value", "MktValue", "MV"],
    "Cost price": ["Cost price", "Cost Price", "Cost"],
    "Accrued interest": ["Accrued interest", "Accrued Interest", "Accrued"],
    "Valuation date": ["Valuation date", "Valuation Date", "Date", "Val Date"],
    "Type": ["Type", "Asset Type"],
    "SecurityName": ["Security Name", "Security", "Name", "Instrument", "Description"],
}


# =========================
# Helpers: normalize + parse
# =========================
def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_number(x) -> Optional[float]:
    if pd.isna(x) or str(x).strip() == "":
        return None

    s = str(x).strip()

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # keep digits ., , and -
    s = re.sub(r"[^\d\.,\-]", "", s)
    if s in ("", "-", ".", ","):
        return None

    # both '.' and ',' exist: decide decimal sep by last occurrence
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            # ',' is decimal: remove thousand '.' then replace ',' with '.'
            s = s.replace(".", "").replace(",", ".")
        else:
            # '.' is decimal: remove thousand ','
            s = s.replace(",", "")
    elif "," in s and "." not in s:
        # ambiguous: treat ',' as decimal
        s = s.replace(",", ".")
    elif "." in s and "," not in s:
        # possible thousand separators like 1.234.567
        parts = s.split(".")
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "")

    try:
        v = float(s)
        if neg:
            v = -v
        return v
    except ValueError:
        return None

def is_numeric_series(ser: pd.Series) -> bool:
    vals = ser.dropna().astype(str).head(200)
    if len(vals) == 0:
        return False
    ok = 0
    for v in vals:
        if normalize_number(v) is not None:
            ok += 1
    return ok / len(vals) >= 0.7

def text_similarity(a: str, b: str) -> int:
    """
    Simple token-based similarity without extra libs.
    0..100
    """
    a = normalize_text(a)
    b = normalize_text(b)
    if a == "" and b == "":
        return 100
    if a == "" or b == "":
        return 0
    # token overlap heuristic
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 100 if a == b else 0
    inter = len(sa & sb)
    union = len(sa | sb) if (sa | sb) else 1
    return int(round(inter / union * 100))

def build_column_renamer(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    ren = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in cols:
            continue
        found = None
        for a in aliases:
            if a.lower() in lower_map:
                found = lower_map[a.lower()]
                break
        if found is not None:
            ren[found] = canonical

    return ren

def load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=object)
    df.columns = [_norm_col(c) for c in df.columns]
    ren = build_column_renamer(df)
    if ren:
        df = df.rename(columns=ren)
    # trim strings
    for c in df.columns:
        df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df


# =========================
# Matching logic (OR-key)
# =========================
def build_index_map(df: pd.DataFrame, col: str) -> Dict[str, List[int]]:
    """
    Map key_value -> list of row indices (allow duplicates).
    """
    mp: Dict[str, List[int]] = {}
    if col not in df.columns:
        return mp
    for idx, v in df[col].items():
        key = normalize_text(v)
        if key == "":
            continue
        mp.setdefault(key, []).append(idx)
    return mp

def pick_best_candidate(
    gold_row: pd.Series,
    pred_df: pd.DataFrame,
    candidates: List[int],
    common_cols: List[str],
    numeric_cols: Dict[str, bool],
) -> Optional[int]:
    """
    If duplicates exist for the same key, pick the candidate row that best matches gold_row
    using a simple score across common columns (only where gold has value).
    """
    best_idx = None
    best_score = -1.0

    for j in candidates:
        score = 0.0
        weight = 0.0
        for c in common_cols:
            gv = gold_row.get(c, None)
            if normalize_text(gv) == "":
                continue  # only judge where gold has value
            pv = pred_df.at[j, c] if c in pred_df.columns else ""
            if numeric_cols.get(c, False):
                gn = normalize_number(gv)
                pn = normalize_number(pv)
                if gn is None or pn is None:
                    s = 0.0
                else:
                    if gn == 0:
                        s = 1.0 if abs(pn - gn) < 1e-9 else 0.0
                    else:
                        s = 1.0 if abs(pn - gn) / abs(gn) <= NUM_REL_TOL else 0.0
            else:
                sim = text_similarity(str(gv), str(pv))
                s = 1.0 if sim >= TEXT_MATCH_THRESHOLD else sim / 100.0
            score += s
            weight += 1.0

        final = score / weight if weight > 0 else 0.0
        if final > best_score:
            best_score = final
            best_idx = j

    return best_idx

def match_rows_or_key(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    sec_col: str,
    acc_col: str,
    common_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      - matched_pairs: columns [gold_idx, pred_idx, key_type, key_value]
      - missing_gold: gold rows that were not found in pred
    """
    sec_map = build_index_map(pred, sec_col)
    acc_map = build_index_map(pred, acc_col)

    numeric_cols = {c: is_numeric_series(gold[c]) for c in common_cols if c in gold.columns}

    used_pred = set()
    pairs = []
    missing = []

    for gi in gold.index:
        g_row = gold.loc[gi]

        sec_val = normalize_text(g_row.get(sec_col, ""))
        acc_val = normalize_text(g_row.get(acc_col, ""))

        matched_pred_idx = None
        key_type = None
        key_value = None

        # 1) prefer Security ID
        if sec_val and sec_val in sec_map:
            cand = [j for j in sec_map[sec_val] if j not in used_pred]
            if cand:
                if len(cand) == 1:
                    matched_pred_idx = cand[0]
                else:
                    matched_pred_idx = pick_best_candidate(g_row, pred, cand, common_cols, numeric_cols)
                key_type = "Security ID"
                key_value = sec_val

        # 2) fallback Account
        if matched_pred_idx is None and acc_val and acc_val in acc_map:
            cand = [j for j in acc_map[acc_val] if j not in used_pred]
            if cand:
                if len(cand) == 1:
                    matched_pred_idx = cand[0]
                else:
                    matched_pred_idx = pick_best_candidate(g_row, pred, cand, common_cols, numeric_cols)
                key_type = "Account"
                key_value = acc_val

        if matched_pred_idx is None:
            missing.append({"gold_idx": gi, "Security ID": sec_val, "Account": acc_val})
        else:
            used_pred.add(matched_pred_idx)
            pairs.append({
                "gold_idx": gi,
                "pred_idx": matched_pred_idx,
                "key_type": key_type,
                "key_value": key_value
            })

    matched_pairs = pd.DataFrame(pairs)
    missing_gold = pd.DataFrame(missing)
    return matched_pairs, missing_gold


# =========================
# Evaluation
# =========================
def compare_cell(gv, pv, numeric: bool) -> bool:
    if numeric:
        gn = normalize_number(gv)
        pn = normalize_number(pv)
        if gn is None and pn is None:
            return True
        if gn is None or pn is None:
            return False
        if gn == 0:
            return abs(pn - gn) < 1e-9
        return abs(pn - gn) / abs(gn) <= NUM_REL_TOL
    else:
        gs = normalize_text(gv)
        ps = normalize_text(pv)
        if gs == "" and ps == "":
            return True
        if gs == "" or ps == "":
            return False
        return text_similarity(gs, ps) >= TEXT_MATCH_THRESHOLD

def evaluate_against_gold(gold: pd.DataFrame, pred: pd.DataFrame, label: str) -> Dict[str, pd.DataFrame]:
    common_cols = sorted(set(gold.columns).intersection(set(pred.columns)))

    if SECURITY_ID_COL not in common_cols and SECURITY_ID_COL not in gold.columns:
        # still allow matching even if gold lacks security id; matching will fallback to account
        pass

    if not common_cols:
        raise RuntimeError(f"[{label}] No common columns between gold and pred (after alias rename).")

    # Match by OR-key
    pairs, missing_gold = match_rows_or_key(gold, pred, SECURITY_ID_COL, ACCOUNT_COL, common_cols)

    total_gold = len(gold)
    matched_gold = len(pairs)
    completeness = (matched_gold / total_gold * 100) if total_gold else 0.0

    # Determine numeric columns from gold
    numeric_cols = {c: is_numeric_series(gold[c]) for c in common_cols}

    # Field accuracy on matched rows, only where gold has value
    field_rows = []
    mismatches = []

    total_compared = 0
    total_correct = 0

    for c in common_cols:
        compared = 0
        correct = 0
        is_num = numeric_cols[c]

        for _, pr in pairs.iterrows():
            gi = pr["gold_idx"]
            pj = pr["pred_idx"]

            gv = gold.at[gi, c]
            if normalize_text(gv) == "":
                continue  # only judge where gold has value

            pv = pred.at[pj, c] if c in pred.columns else ""

            compared += 1
            ok = compare_cell(gv, pv, is_num)
            correct += int(ok)

            if not ok:
                mismatches.append({
                    "Field": c,
                    "gold_idx": gi,
                    "pred_idx": pj,
                    "key_type": pr["key_type"],
                    "key_value": pr["key_value"],
                    "Gold_value": gv,
                    "Pred_value": pv
                })

        acc = (correct / compared * 100) if compared else 0.0
        field_rows.append({
            "Field": c,
            "Compared_cells": compared,
            "Correct_cells": correct,
            "Accuracy_%": round(acc, 2),
            "IsNumeric": is_num
        })

        total_compared += compared
        total_correct += correct

    overall_accuracy = (total_correct / total_compared * 100) if total_compared else 0.0

    summary = pd.DataFrame([{
        "Label": label,
        "Gold_rows": total_gold,
        "Pred_rows": len(pred),
        "Matched_gold_rows": matched_gold,
        "Completeness_%": round(completeness, 2),
        "Overall_accuracy_%": round(overall_accuracy, 2),
        "Text_match_threshold": TEXT_MATCH_THRESHOLD,
        "Num_rel_tol": NUM_REL_TOL,
        "Key_rule": "Prefer Security ID; fallback Account"
    }])

    field_acc = pd.DataFrame(field_rows).sort_values(by="Accuracy_%", ascending=True)
    mism_df = pd.DataFrame(mismatches)

    return {
        "summary": summary,
        "pairs": pairs,
        "missing_gold": missing_gold,
        "field_accuracy": field_acc,
        "mismatches": mism_df
    }


def main():
    gold = load_excel(GOLD_PATH)
    a = load_excel(FILE_A_PATH)
    b = load_excel(FILE_B_PATH)

    # Ensure key columns exist (after alias rename). If not, try to proceed with what's available.
    for df_name, df in [("GOLD", gold), ("A", a), ("B", b)]:
        # just informative prints
        pass

    res_a = evaluate_against_gold(gold, a, "A")
    res_b = evaluate_against_gold(gold, b, "B")

    # Compare A vs B overall
    a_comp = float(res_a["summary"].iloc[0]["Completeness_%"])
    b_comp = float(res_b["summary"].iloc[0]["Completeness_%"])
    a_acc = float(res_a["summary"].iloc[0]["Overall_accuracy_%"])
    b_acc = float(res_b["summary"].iloc[0]["Overall_accuracy_%"])

    compare_overall = pd.DataFrame([{
        "Metric": "Completeness_% (matched gold rows / gold rows)",
        "A": a_comp,
        "B": b_comp,
        "B - A (pp)": round(b_comp - a_comp, 2),
        "Winner": "B" if b_comp > a_comp else ("A" if a_comp > b_comp else "Tie")
    }, {
        "Metric": "Overall_accuracy_% (cells correct where gold non-empty, on matched rows)",
        "A": a_acc,
        "B": b_acc,
        "B - A (pp)": round(b_acc - a_acc, 2),
        "Winner": "B" if b_acc > a_acc else ("A" if a_acc > b_acc else "Tie")
    }])

    # Field-level winner
    fa = res_a["field_accuracy"][["Field", "Accuracy_%"]].rename(columns={"Accuracy_%": "A_Accuracy_%"}).copy()
    fb = res_b["field_accuracy"][["Field", "Accuracy_%"]].rename(columns={"Accuracy_%": "B_Accuracy_%"}).copy()
    field_winner = fa.merge(fb, on="Field", how="outer").fillna(0.0)
    field_winner["Winner"] = field_winner.apply(
        lambda r: "B" if r["B_Accuracy_%"] > r["A_Accuracy_%"] else ("A" if r["A_Accuracy_%"] > r["B_Accuracy_%"] else "Tie"),
        axis=1
    )
    field_winner["B - A (pp)"] = (field_winner["B_Accuracy_%"] - field_winner["A_Accuracy_%"]).round(2)
    field_winner = field_winner.sort_values(by="B - A (pp)", ascending=False)

    # Print quick summary
    print("=== OVERALL COMPARISON ===")
    print(compare_overall.to_string(index=False))
    print("\n=== A SUMMARY ===")
    print(res_a["summary"].to_string(index=False))
    print("\n=== B SUMMARY ===")
    print(res_b["summary"].to_string(index=False))

    # Save report
    with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as w:
        compare_overall.to_excel(w, index=False, sheet_name="overall_compare")
        field_winner.to_excel(w, index=False, sheet_name="field_winner")

        res_a["summary"].to_excel(w, index=False, sheet_name="A_summary")
        res_a["field_accuracy"].to_excel(w, index=False, sheet_name="A_field_accuracy")
        res_a["missing_gold"].to_excel(w, index=False, sheet_name="A_missing_gold")
        res_a["pairs"].to_excel(w, index=False, sheet_name="A_pairs")
        res_a["mismatches"].head(1000).to_excel(w, index=False, sheet_name="A_mismatches_1000")

        res_b["summary"].to_excel(w, index=False, sheet_name="B_summary")
        res_b["field_accuracy"].to_excel(w, index=False, sheet_name="B_field_accuracy")
        res_b["missing_gold"].to_excel(w, index=False, sheet_name="B_missing_gold")
        res_b["pairs"].to_excel(w, index=False, sheet_name="B_pairs")
        res_b["mismatches"].head(1000).to_excel(w, index=False, sheet_name="B_mismatches_1000")

    print(f"\n✅ Saved report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
