"""
LightGBM reranker: second-stage ranking over recalled candidates.

Training data construction (for validation):
  - For each training user, hold out their last val_days actions as labels
  - For each (user, candidate_combo) pair, build a feature vector
  - Positive label = 1 if combo appears in user's held-out future actions
  - Negative label = 0 otherwise
  - Train with LightGBM LambdaRank (or binary classification as fallback)

At inference:
  - Score every (test_user, candidate_combo) pair
  - Return top-20 combos sorted by predicted score
"""

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[reranker] lightgbm not installed — reranker will be disabled.")


FEATURE_COLS: list[str] = []   # filled by build_feature_matrix


def build_feature_matrix(
    candidates_per_user: dict[str, list[str]],
    user_feat: pd.DataFrame,
    combo_feat: pd.DataFrame,
    user_combo_feat: pd.DataFrame,
    transition_table: pd.DataFrame,
    user_last_combo: pd.Series,
    sasrec_scores: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    For each (user_id, combo_id) pair, assemble all features into one row.

    Parameters
    ----------
    candidates_per_user : {user_id: [combo_id, ...]}
    user_feat           : output of features.user_features()
    combo_feat          : output of features.combo_features()
    user_combo_feat     : output of features.user_combo_features()
    transition_table    : output of features.build_transition_table()
    user_last_combo     : output of features.get_user_last_combo()
    sasrec_scores       : {user_id: {combo_id: score}} — optional SASRec signal

    Returns a DataFrame with columns [user_id, combo_id, <feature_cols>].
    """
    global FEATURE_COLS

    rows = []
    for uid, combos in candidates_per_user.items():
        u_feat   = user_feat.loc[uid] if uid in user_feat.index else pd.Series(dtype=float)
        last_c   = user_last_combo.get(uid, "")
        sasrec_u = (sasrec_scores or {}).get(uid, {})

        for cid in combos:
            c_feat  = combo_feat.loc[cid] if cid in combo_feat.index else pd.Series(dtype=float)
            uc_idx  = (uid, cid)
            uc_feat = (
                user_combo_feat.loc[uc_idx]
                if uc_idx in user_combo_feat.index
                else pd.Series(dtype=float)
            )

            # Transition score from last combo
            trans_rows = transition_table[
                (transition_table["from_combo"] == last_c) &
                (transition_table["to_combo"]   == cid)
            ]
            trans_prob = float(trans_rows["prob"].values[0]) if len(trans_rows) else 0.0

            # SASRec score (normalised to [0,1] within user)
            raw_sasrec = sasrec_u.get(cid, 0.0)
            if sasrec_u:
                max_s = max(sasrec_u.values())
                min_s = min(sasrec_u.values())
                norm_sasrec = (raw_sasrec - min_s) / (max_s - min_s + 1e-9)
            else:
                norm_sasrec = 0.0

            row = {
                "user_id":  uid,
                "combo_id": cid,
                # user features
                **{f"u_{k}": v for k, v in u_feat.items()
                   if isinstance(v, (int, float, np.integer, np.floating))},
                # combo features
                **{f"c_{k}": v for k, v in c_feat.items()
                   if isinstance(v, (int, float, np.integer, np.floating))},
                # user-combo features
                **{f"uc_{k}": v for k, v in uc_feat.items()
                   if isinstance(v, (int, float, np.integer, np.floating))},
                # sequential features
                "f_trans_prob":  trans_prob,
                "f_sasrec":      norm_sasrec,
            }
            rows.append(row)

    df = pd.DataFrame(rows).fillna(0.0)
    FEATURE_COLS = [c for c in df.columns if c not in ("user_id", "combo_id")]
    return df


def _add_labels(
    feat_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach binary label: 1 if the combo appears in the user's ground-truth future actions.
    """
    truth_df = truth_df.copy()
    truth_df["label"] = 1
    truth_keys = truth_df[["user_id", "combo_id", "label"]].drop_duplicates()
    merged = feat_df.merge(truth_keys, on=["user_id", "combo_id"], how="left")
    merged["label"] = merged["label"].fillna(0).astype(int)
    return merged


def train_reranker(
    feat_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    use_lambdarank: bool = True,
) -> "lgb.Booster | None":
    """
    Train a LightGBM ranking model.

    truth_df : DataFrame with [user_id, combo_id] of the ground-truth future combos.
    """
    if not LGB_AVAILABLE:
        return None

    df = _add_labels(feat_df, truth_df)
    df = df.sort_values("user_id")     # LambdaRank requires group-contiguous rows

    X = df[FEATURE_COLS].values
    y = df["label"].values

    if use_lambdarank:
        group = df.groupby("user_id", sort=False).size().values
        train_set = lgb.Dataset(X, label=y, group=group)
        params = {
            "objective":     "lambdarank",
            "metric":        "ndcg",
            "ndcg_eval_at":  [20],
            "num_leaves":    63,
            "learning_rate": 0.05,
            "n_estimators":  200,
            "verbose":       -1,
        }
    else:
        # Fallback: binary classification (simpler, no group needed)
        train_set = lgb.Dataset(X, label=y)
        params = {
            "objective":     "binary",
            "metric":        "auc",
            "num_leaves":    63,
            "learning_rate": 0.05,
            "n_estimators":  200,
            "verbose":       -1,
        }

    booster = lgb.train(params, train_set, num_boost_round=200)
    return booster


def rerank(
    booster: "lgb.Booster",
    feat_df: pd.DataFrame,
    top_k: int = 20,
) -> dict[str, list[str]]:
    """
    Apply trained booster to feat_df, return top_k combo_ids per user.
    Returns {user_id: [combo_id_rank1, combo_id_rank2, ...]}
    """
    X = feat_df[FEATURE_COLS].values
    feat_df = feat_df.copy()
    feat_df["score"] = booster.predict(X)

    result = {}
    for uid, grp in feat_df.groupby("user_id"):
        top = grp.sort_values("score", ascending=False).head(top_k)
        result[uid] = top["combo_id"].tolist()

    return result
