"""
Baseline model: frequency + recency-weighted scoring, no ML reranker.

Scoring formula for each (user, combo) pair:
  score = alpha * personal_freq_norm
        + beta  * personal_decay
        + gamma * global_freq_norm

Personal history combos always rank above unseen combos (via a priority flag).
"""

import numpy as np
import pandas as pd


ALPHA = 0.5   # personal frequency weight
BETA  = 0.3   # personal recency-decay weight
GAMMA = 0.2   # global popularity weight


def _decay(timestamps: pd.Series, now: pd.Timestamp, half_life_days: float = 7.0) -> float:
    if len(timestamps) == 0:
        return 0.0
    days_ago = (now - timestamps).dt.total_seconds() / 86400
    lam = np.log(2) / half_life_days
    return float(np.exp(-lam * days_ago).sum())


def score_candidates(
    df_context: pd.DataFrame,
    user_id: str,
    candidates: list[str],
    global_freq: pd.Series,
) -> pd.Series:
    """
    Return a Series mapping combo_id -> score, for the given candidate list.

    Parameters
    ----------
    df_context  : full context DataFrame (cleaned)
    user_id     : target user
    candidates  : ordered list of combo_ids to score
    global_freq : Series (index=combo_id, values=global occurrence count)
    """
    now = df_context["acs_tm"].max()
    user_df = df_context[df_context["user_id"] == user_id]

    # Personal stats
    personal_freq: dict[str, int] = {}
    personal_decay: dict[str, float] = {}
    if len(user_df) > 0:
        for cid, grp in user_df.groupby("combo_id"):
            personal_freq[cid]  = len(grp)
            personal_decay[cid] = _decay(grp["acs_tm"], now)

    # Normalizers
    max_pfreq  = max(personal_freq.values(), default=1)
    max_pdecay = max(personal_decay.values(), default=1)
    max_gfreq  = global_freq.max() if len(global_freq) > 0 else 1

    scores: dict[str, float] = {}
    for cid in candidates:
        pf  = personal_freq.get(cid, 0) / max_pfreq
        pd_ = personal_decay.get(cid, 0) / max_pdecay
        gf  = global_freq.get(cid, 0) / max_gfreq

        # Combos seen by user always above unseen (priority bonus)
        seen_bonus = 1.0 if cid in personal_freq else 0.0

        scores[cid] = seen_bonus + ALPHA * pf + BETA * pd_ + GAMMA * gf

    return pd.Series(scores).sort_values(ascending=False)


def predict_user(
    df_context: pd.DataFrame,
    user_id: str,
    candidates: list[str],
    global_freq: pd.Series,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      user_id, action_typ, prod_typ, prod_sub_typ, rsk_lvl, index
    with exactly top_k rows (index 1 … top_k).
    """
    from src.data_processing import combo_id_to_fields

    scores = score_candidates(df_context, user_id, candidates, global_freq)
    top_combos = scores.head(top_k).index.tolist()

    rows = []
    for rank, cid in enumerate(top_combos, start=1):
        fields = combo_id_to_fields(cid)
        rows.append({
            "user_id":      user_id,
            "action_typ":   fields["action_typ"],
            "prod_typ":     fields["prod_typ"],
            "prod_sub_typ": fields["prod_sub_typ"],
            "rsk_lvl":      fields["rsk_lvl"],
            "index":        rank,
        })

    return pd.DataFrame(rows)
