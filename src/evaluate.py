"""
NDCG@K evaluation following the exact competition definition.

A prediction is relevant (rel=1) only if ALL FOUR fields match exactly:
  action_typ, prod_typ, prod_sub_typ, rsk_lvl

Empty string '' is treated as a valid value (matches structural NaN filled rows).
"""

import numpy as np
import pandas as pd


def ndcg_at_k(
    predictions: list[dict],
    ground_truth: list[dict],
    k: int = 20,
) -> float:
    """
    predictions : list of dicts with keys action_typ/prod_typ/prod_sub_typ/rsk_lvl,
                  ordered from index=1 (highest confidence) to index=k.
    ground_truth: list of dicts with the same keys (actual future actions).

    Returns NDCG@k for a single user.
    """
    COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]

    gt_set = set(
        tuple(g[c] for c in COLS) for g in ground_truth
    )

    dcg  = 0.0
    idcg = 0.0

    n_relevant = len(gt_set)
    for i in range(1, k + 1):
        # IDCG: assume top-min(n_relevant,k) positions are all relevant
        if i <= n_relevant:
            idcg += 1.0 / np.log2(i + 1)

        # DCG: check if prediction at rank i is relevant
        if i <= len(predictions):
            pred_tuple = tuple(predictions[i - 1][c] for c in COLS)
            if pred_tuple in gt_set:
                dcg += 1.0 / np.log2(i + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_all_users(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    k: int = 20,
) -> dict:
    """
    pred_df  : columns [user_id, action_typ, prod_typ, prod_sub_typ, rsk_lvl, index]
    truth_df : columns [user_id, action_typ, prod_typ, prod_sub_typ, rsk_lvl]

    Returns dict with per-user scores and overall mean NDCG@k.
    """
    COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]

    scores = {}
    for uid in truth_df["user_id"].unique():
        user_preds = (
            pred_df[pred_df["user_id"] == uid]
            .sort_values("index")
            .head(k)
            [COLS]
            .to_dict("records")
        )
        user_truth = (
            truth_df[truth_df["user_id"] == uid]
            [COLS]
            .to_dict("records")
        )
        scores[uid] = ndcg_at_k(user_preds, user_truth, k)

    mean_score = float(np.mean(list(scores.values()))) if scores else 0.0
    return {"per_user": scores, "ndcg@20": mean_score}
