"""
Build and save the final submission file.

Output format (UTF-8-sig):
  user_id, action_typ, prod_typ, prod_sub_typ, rsk_lvl, index
  index = 1 (most likely) to 20 (least likely)
  Empty string fields are written as '' (not NaN) in the CSV.
"""

import pandas as pd
from src.data_processing import combo_id_to_fields


SUBMISSION_COLS = ["user_id", "action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl", "index"]


def build_submission(
    ranked_combos_per_user: dict[str, list[str]],
    top_k: int = 20,
) -> pd.DataFrame:
    """
    ranked_combos_per_user : {user_id: [combo_id_rank1, combo_id_rank2, ...]}
    Returns a DataFrame with exactly top_k rows per user.
    """
    rows = []
    for uid, combos in ranked_combos_per_user.items():
        for rank, cid in enumerate(combos[:top_k], start=1):
            fields = combo_id_to_fields(cid)
            rows.append({
                "user_id":      uid,
                "action_typ":   fields["action_typ"],
                "prod_typ":     fields["prod_typ"],
                "prod_sub_typ": fields["prod_sub_typ"],
                "rsk_lvl":      fields["rsk_lvl"],
                "index":        rank,
            })
    df = pd.DataFrame(rows, columns=SUBMISSION_COLS)
    return df


def save_submission(df: pd.DataFrame, path: str = "prediction.csv"):
    """Save as UTF-8-sig encoded CSV (required by competition)."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {path}  ({df['user_id'].nunique()} users)")
