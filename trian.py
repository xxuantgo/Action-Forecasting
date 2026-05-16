"""
Main training script.

Pipeline:
  1. Load & clean data
  2. Build validation split (last val_days of each train user)
  3. [Baseline] Score candidates with frequency + recency
  4. [Advanced] Train SASRec on train context
  5. [Advanced] Train LightGBM reranker using val labels
  6. Evaluate on validation set → log NDCG@20
  7. Save model artifacts to ./checkpoints/

Run:
  python train.py --train_path dataset_trian.csv --val_days 7 --mode full
"""

import argparse
import os
import pickle

import pandas as pd

from src.data_processing import load_raw, clean, split_train_val, build_combo_vocab
from src.features import (
    user_features, combo_features, user_combo_features,
    build_transition_table, get_user_last_combo,
)
from src.recall import build_candidate_pool, global_popular_candidates
from src.baseline import predict_user as baseline_predict_user
from src.evaluate import evaluate_all_users
from src.submission import build_submission


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="dataset_trian.csv")
    p.add_argument("--val_days",   type=int, default=7)
    p.add_argument("--mode",       choices=["baseline", "full"], default="baseline",
                   help="baseline: freq-only | full: SASRec + LightGBM reranker")
    p.add_argument("--max_candidates", type=int, default=150)
    p.add_argument("--out_dir",    default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load & clean ────────────────────────────────────────────
    print("Loading data...")
    df_raw = load_raw(args.train_path)
    df     = clean(df_raw)
    print(f"  {len(df)} records, {df['user_id'].nunique()} users")

    # ── 2. Validation split ────────────────────────────────────────
    df_ctx, df_val = split_train_val(df, val_days=args.val_days)
    print(f"  Context: {len(df_ctx)} rows | Val: {len(df_val)} rows")

    # ── 3. Shared artifacts ────────────────────────────────────────
    vocab          = build_combo_vocab(df_ctx)
    global_popular = global_popular_candidates(df_ctx, top_k=50)
    transition_tbl = build_transition_table(df_ctx)
    global_freq    = df_ctx.groupby("combo_id").size()

    # Save for inference
    with open(os.path.join(args.out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(args.out_dir, "global_popular.pkl"), "wb") as f:
        pickle.dump(global_popular, f)
    transition_tbl.to_parquet(os.path.join(args.out_dir, "transition_table.parquet"), index=False)

    # ── 4. Build candidate pool per validation user ────────────────
    val_users = df_ctx["user_id"].unique().tolist()
    candidates_per_user: dict[str, list[str]] = {}
    for uid in val_users:
        candidates_per_user[uid] = build_candidate_pool(
            df_ctx, uid, transition_tbl, global_popular,
            max_candidates=args.max_candidates,
        )

    # ── 5a. Baseline evaluation ────────────────────────────────────
    print("\n── Baseline evaluation ──")
    pred_rows = []
    for uid in val_users:
        user_pred = baseline_predict_user(
            df_ctx, uid, candidates_per_user[uid], global_freq
        )
        pred_rows.append(user_pred)

    pred_df = pd.concat(pred_rows, ignore_index=True)

    # Prepare val truth (add combo_id to val df)
    df_val_clean = clean(df_raw)   # full clean for consistent combo_id
    df_val_clean = df_val_clean[df_val_clean["user_id"].isin(val_users)]
    df_val_clean = df_val_clean[df_val_clean["acs_tm"] > df_ctx["acs_tm"].max()]

    if len(df_val_clean) > 0:
        result = evaluate_all_users(pred_df, df_val_clean)
        print(f"  Baseline NDCG@20 = {result['ndcg@20']:.4f}")
        for uid, score in result["per_user"].items():
            print(f"    {uid}: {score:.4f}")
    else:
        print("  No val labels available (val_days too large or data too small).")

    if args.mode == "baseline":
        print("\nBaseline mode: done.")
        return

    # ── 5b. Full mode: SASRec + LightGBM ──────────────────────────
    print("\n── Training SASRec ──")
    from src.models.sasrec import SASRec

    user_sequences = (
        df_ctx.sort_values("acs_tm")
        .groupby("user_id")["combo_id"]
        .apply(list)
        .to_dict()
    )
    sasrec = SASRec(hidden_dim=64, n_heads=2, n_layers=2, max_len=50, n_epochs=20)
    sasrec.fit(user_sequences)

    # Compute SASRec scores for each val user
    sasrec_scores: dict[str, dict[str, float]] = {}
    for uid, seq in user_sequences.items():
        sasrec_scores[uid] = sasrec.predict_scores(seq)

    with open(os.path.join(args.out_dir, "sasrec.pkl"), "wb") as f:
        pickle.dump(sasrec, f)

    print("\n── Building reranker features ──")
    from src.features import user_features, combo_features, user_combo_features
    from src.reranker import build_feature_matrix, train_reranker, rerank

    u_feat  = user_features(df_ctx)
    c_feat  = combo_features(df_ctx)
    uc_feat = user_combo_features(df_ctx)
    last_c  = get_user_last_combo(df_ctx)

    feat_df = build_feature_matrix(
        candidates_per_user, u_feat, c_feat, uc_feat,
        transition_tbl, last_c, sasrec_scores,
    )

    # Ground truth for reranker training: val combos
    if len(df_val_clean) > 0:
        truth_for_reranker = df_val_clean[["user_id", "combo_id"]].drop_duplicates()

        print("\n── Training LightGBM reranker ──")
        booster = train_reranker(feat_df, truth_for_reranker, use_lambdarank=True)

        if booster:
            with open(os.path.join(args.out_dir, "reranker.pkl"), "wb") as f:
                pickle.dump(booster, f)

            ranked = rerank(booster, feat_df, top_k=20)
            sub_df = build_submission(ranked)
            result2 = evaluate_all_users(sub_df, df_val_clean)
            print(f"  Reranker NDCG@20 = {result2['ndcg@20']:.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
