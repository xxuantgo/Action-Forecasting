"""
Inference script: generate prediction.csv for submission.

Run:
  python predict.py --train_path dataset_trian.csv \
                    --test_path  dataset_test.csv  \
                    --mode full                     \
                    --out prediction.csv

Mode:
  baseline : frequency + recency only (fast, no model loading)
  full     : load SASRec + LightGBM from ./checkpoints/ and rerank
"""

import argparse
import os
import pickle

import pandas as pd

from src.data_processing import load_raw, clean, build_combo_vocab
from src.features import (
    user_features, combo_features, user_combo_features,
    build_transition_table, get_user_last_combo,
)
from src.recall import build_candidate_pool, global_popular_candidates
from src.baseline import predict_user as baseline_predict_user
from src.submission import build_submission, save_submission


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path",     default="dataset_trian.csv")
    p.add_argument("--test_path",      default="dataset_test.csv")
    p.add_argument("--mode",           choices=["baseline", "full"], default="baseline")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--max_candidates", type=int, default=150)
    p.add_argument("--out",            default="prediction.csv")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load & clean ───────────────────────────────────────────────
    print("Loading data...")
    df_train = clean(load_raw(args.train_path))
    df_test  = clean(load_raw(args.test_path))
    print(f"  Train: {len(df_train)} rows, {df_train['user_id'].nunique()} users")
    print(f"  Test : {len(df_test)} rows,  {df_test['user_id'].nunique()} users")

    # Combined context = train + test (test user's own history is their context)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Shared artifacts
    transition_tbl = build_transition_table(df_train)   # built from train only
    global_popular = global_popular_candidates(df_train, top_k=50)
    global_freq    = df_train.groupby("combo_id").size()

    test_users = df_test["user_id"].unique().tolist()
    print(f"  Generating predictions for {len(test_users)} test users...")

    # ── Candidate pool per test user ───────────────────────────────
    candidates_per_user: dict[str, list[str]] = {}
    for uid in test_users:
        candidates_per_user[uid] = build_candidate_pool(
            df_all, uid, transition_tbl, global_popular,
            max_candidates=args.max_candidates,
        )

    if args.mode == "baseline":
        # ── Baseline: score and rank ───────────────────────────────
        pred_rows = []
        for uid in test_users:
            user_pred = baseline_predict_user(
                df_all, uid, candidates_per_user[uid], global_freq
            )
            pred_rows.append(user_pred)
        sub_df = pd.concat(pred_rows, ignore_index=True)

    else:
        # ── Full: SASRec + LightGBM ────────────────────────────────
        ckpt = args.checkpoint_dir

        with open(os.path.join(ckpt, "sasrec.pkl"), "rb") as f:
            sasrec = pickle.load(f)
        with open(os.path.join(ckpt, "reranker.pkl"), "rb") as f:
            booster = pickle.load(f)

        from src.reranker import build_feature_matrix, rerank, FEATURE_COLS

        # SASRec scores for test users (using their own history as sequence)
        sasrec_scores: dict[str, dict[str, float]] = {}
        for uid in test_users:
            seq = (
                df_all[df_all["user_id"] == uid]
                .sort_values("acs_tm")["combo_id"]
                .tolist()
            )
            sasrec_scores[uid] = sasrec.predict_scores(seq)

        # Features for test user × candidate combos
        u_feat  = user_features(df_all)
        c_feat  = combo_features(df_all)
        uc_feat = user_combo_features(df_all)
        last_c  = get_user_last_combo(df_all)

        feat_df = build_feature_matrix(
            candidates_per_user, u_feat, c_feat, uc_feat,
            transition_tbl, last_c, sasrec_scores,
        )

        # Align feature columns to what reranker was trained on
        for col in FEATURE_COLS:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        ranked = rerank(booster, feat_df, top_k=20)
        sub_df = build_submission(ranked)

    # ── Save ───────────────────────────────────────────────────────
    save_submission(sub_df, args.out)


if __name__ == "__main__":
    main()
