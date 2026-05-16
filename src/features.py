"""
Feature engineering for the LightGBM reranker.

Features fall into three groups:
  A. User-level      — who is this user?
  B. Combo-level     — how popular / recent is this combo globally?
  C. User×Combo      — how does this user relate to this specific combo?

All features are computed from the "context window" (train or val split),
never from future data, to prevent leakage.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ──────────────────────────────────────────────
# A. User-level features
# ──────────────────────────────────────────────

def user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per user_id.
    Computed over the entire context window for that user.
    """
    now = df["acs_tm"].max()

    records = []
    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values("acs_tm")
        n = len(grp)
        span_days = (grp["acs_tm"].max() - grp["acs_tm"].min()).total_seconds() / 86400 + 1
        recency_days = (now - grp["acs_tm"].max()).total_seconds() / 86400

        action_counts = grp["action_typ"].value_counts()
        prod_counts   = grp["prod_typ"].value_counts()
        sub_counts    = grp["prod_sub_typ"].value_counts()
        rsk_counts    = grp["rsk_lvl"].value_counts()

        # dominant risk level (numeric: R1→1, R5→5, ''→0)
        rsk_map = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5, "": 0}
        rsk_num = grp["rsk_lvl"].map(rsk_map).fillna(0)
        avg_rsk = rsk_num.mean()

        # purchase / redeem activity rate
        buy_rate    = action_counts.get("购买", 0) / n
        redeem_rate = action_counts.get("赎回", 0) / n
        browse_rate = action_counts.get("浏览详情", 0) / n

        records.append({
            "user_id": uid,
            "uf_n_actions":         n,
            "uf_n_unique_combos":   grp["combo_id"].nunique(),
            "uf_n_unique_prod_typ": grp["prod_typ"].nunique(),
            "uf_n_unique_sub_typ":  grp["prod_sub_typ"].nunique(),
            "uf_span_days":         span_days,
            "uf_recency_days":      recency_days,
            "uf_avg_rsk":           avg_rsk,
            "uf_buy_rate":          buy_rate,
            "uf_redeem_rate":       redeem_rate,
            "uf_browse_rate":       browse_rate,
            "uf_top_prod_typ":      prod_counts.index[0] if len(prod_counts) else "",
            "uf_top_action":        action_counts.index[0] if len(action_counts) else "",
            "uf_top_rsk":           rsk_counts.index[0] if len(rsk_counts) else "",
        })

    return pd.DataFrame(records).set_index("user_id")


# ──────────────────────────────────────────────
# B. Combo-level features  (global statistics)
# ──────────────────────────────────────────────

def combo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per combo_id.
    Captures global popularity and recency of each four-tuple.
    """
    now = df["acs_tm"].max()
    n_users = df["user_id"].nunique()

    records = []
    for cid, grp in df.groupby("combo_id"):
        last_seen_days = (now - grp["acs_tm"].max()).total_seconds() / 86400
        records.append({
            "combo_id":              cid,
            "cf_global_freq":        len(grp),
            "cf_global_user_cnt":    grp["user_id"].nunique(),
            "cf_global_user_ratio":  grp["user_id"].nunique() / n_users,
            "cf_last_seen_days":     last_seen_days,
            # exponential-decay global score: recent occurrences weigh more
            "cf_decay_score":        _decay_score(grp["acs_tm"], now, half_life=7),
        })

    return pd.DataFrame(records).set_index("combo_id")


def _decay_score(timestamps: pd.Series, now: pd.Timestamp, half_life: float) -> float:
    """Sum of exp(-lambda * days_ago) for each occurrence."""
    days_ago = (now - timestamps).dt.total_seconds() / 86400
    lam = np.log(2) / half_life
    return float(np.exp(-lam * days_ago).sum())


# ──────────────────────────────────────────────
# C. User × Combo interaction features
# ──────────────────────────────────────────────

def user_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (user_id, combo_id).
    Captures how much the user has historically interacted with this combo.
    """
    now = df["acs_tm"].max()
    records = []

    for (uid, cid), grp in df.groupby(["user_id", "combo_id"]):
        last_days = (now - grp["acs_tm"].max()).total_seconds() / 86400
        user_total = df[df["user_id"] == uid].shape[0]
        records.append({
            "user_id":           uid,
            "combo_id":          cid,
            "ucf_freq":          len(grp),
            "ucf_freq_ratio":    len(grp) / user_total,
            "ucf_last_days":     last_days,
            "ucf_decay":         _decay_score(grp["acs_tm"], now, half_life=7),
            "ucf_ever_bought":   int("购买" in grp["action_typ"].values),
            "ucf_ever_redeemed": int("赎回" in grp["action_typ"].values),
        })

    return pd.DataFrame(records).set_index(["user_id", "combo_id"])


# ──────────────────────────────────────────────
# D. Transition / sequential features (Markov)
# ──────────────────────────────────────────────

def build_transition_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    First-order Markov transition probabilities between combo_ids.
    Returns a DataFrame with columns: from_combo, to_combo, prob.
    Built only from training data.
    """
    df_sorted = df.sort_values(["user_id", "acs_tm"])
    df_sorted["next_combo"] = df_sorted.groupby("user_id")["combo_id"].shift(-1)
    transitions = (
        df_sorted.dropna(subset=["next_combo"])
        .groupby(["combo_id", "next_combo"])
        .size()
        .reset_index(name="cnt")
    )
    total_per_combo = transitions.groupby("combo_id")["cnt"].transform("sum")
    transitions["prob"] = transitions["cnt"] / total_per_combo
    return transitions.rename(columns={"combo_id": "from_combo", "next_combo": "to_combo"})


def last_combo_transition_score(
    user_last_combo: str,
    candidate_combo: str,
    transition_table: pd.DataFrame,
) -> float:
    """
    P(candidate_combo | user's last combo), smoothed with 0.
    """
    row = transition_table[
        (transition_table["from_combo"] == user_last_combo) &
        (transition_table["to_combo"]   == candidate_combo)
    ]
    return float(row["prob"].values[0]) if len(row) else 0.0


def get_user_last_combo(df: pd.DataFrame) -> pd.Series:
    """Returns the most recent combo_id for each user."""
    return (
        df.sort_values("acs_tm")
        .groupby("user_id")["combo_id"]
        .last()
    )
