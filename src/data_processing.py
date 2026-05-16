"""
Data loading, cleaning, and preprocessing.

The four-tuple (action_typ, prod_typ, prod_sub_typ, rsk_lvl) is the atomic
prediction unit; we call it a "combo" throughout the codebase.

Missing-value policy (chosen 2026-05-16):
    Drop any row whose required fields are missing. This removes "参与活动 / 非财富"
    rows where prod_sub_typ and rsk_lvl are structurally NaN. The simpler invariant
    downstream (no empty strings in combo_id) outweighs losing that signal.
    If we ever want to keep them, switch DROP_MISSING -> False and rows with
    NaN combo fields will be filled with SENTINEL instead.
"""

import pandas as pd


COMBO_COLS    = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
REQUIRED_COLS = ["user_id", "acs_tm"] + COMBO_COLS
SENTINEL      = "NA"
DROP_MISSING  = True


def load_raw(path: str) -> pd.DataFrame:
    # utf-8-sig strips the BOM that real exports sometimes carry on the first header.
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame with:
      - all required columns non-null and stripped
      - acs_tm parsed to datetime64 (unparseable rows dropped)
      - exact duplicates removed
      - combo_id column "<action>|<prod>|<sub>|<rsk>"
      - sorted by (user_id, acs_tm)
    """
    df = df.copy()

    for col in ["user_id"] + COMBO_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "None", "NaN"]), col] = pd.NA

    df["acs_tm"] = pd.to_datetime(df["acs_tm"], errors="coerce")

    before = len(df)
    if DROP_MISSING:
        df = df.dropna(subset=REQUIRED_COLS)
    else:
        df = df.dropna(subset=["user_id", "acs_tm", "action_typ", "prod_typ"])
        for col in ["prod_sub_typ", "rsk_lvl"]:
            df[col] = df[col].fillna(SENTINEL)
    dropped = before - len(df)
    if dropped:
        print(f"  [clean] dropped {dropped} rows with missing required fields")

    df = df.drop_duplicates(subset=["user_id", "acs_tm"] + COMBO_COLS)

    df["combo_id"] = (
        df["action_typ"] + "|" +
        df["prod_typ"]   + "|" +
        df["prod_sub_typ"] + "|" +
        df["rsk_lvl"]
    )

    df = df.sort_values(["user_id", "acs_tm"]).reset_index(drop=True)
    return df


def split_train_val(df: pd.DataFrame, val_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: hold out each user's last `val_days` of actions as ground truth."""
    cutoff = df["acs_tm"].max() - pd.Timedelta(days=val_days)
    train_part = df[df["acs_tm"] <= cutoff].copy()
    val_part   = df[df["acs_tm"] >  cutoff].copy()
    return train_part, val_part


def build_combo_vocab(df: pd.DataFrame) -> dict:
    """Map each unique combo_id to a 1-based integer index (0 reserved for padding)."""
    combos = df["combo_id"].unique().tolist()
    return {c: i + 1 for i, c in enumerate(sorted(combos))}


def combo_id_to_fields(combo_id: str) -> dict:
    parts = combo_id.split("|")
    return {
        "action_typ":   parts[0],
        "prod_typ":     parts[1],
        "prod_sub_typ": parts[2],
        "rsk_lvl":      parts[3],
    }
