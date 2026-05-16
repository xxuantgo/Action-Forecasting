"""
Recall stage: generate a pool of candidate combos for each test user.

Strategy (in priority order):
  1. Personal history         — all combos the user has done before (highest precision)
  2. Markov next-combo        — combos with high P(next | last) from transition table
  3. User-similar CF          — combos from users with overlapping history (collaborative)
  4. Global popularity        — most frequent combos across all train users (fallback)

The union of these sources forms the candidate set that the reranker scores.
Target: ~100-200 candidates per user (trade-off recall ceiling vs. reranker cost).
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def personal_history_candidates(df: pd.DataFrame, user_id: str) -> list[str]:
    """All combo_ids this user has previously done, ordered by recency then freq."""
    user_df = df[df["user_id"] == user_id].sort_values("acs_tm")
    freq = user_df.groupby("combo_id").size()
    recency = user_df.groupby("combo_id")["acs_tm"].max()
    rank = pd.DataFrame({"freq": freq, "recency": recency})
    rank = rank.sort_values(["recency", "freq"], ascending=[False, False])
    return rank.index.tolist()


def markov_candidates(
    user_last_combo: str,
    transition_table: pd.DataFrame,
    top_k: int = 30,
) -> list[str]:
    """Combos with highest transition probability given the user's last action."""
    rows = transition_table[transition_table["from_combo"] == user_last_combo]
    rows = rows.sort_values("prob", ascending=False).head(top_k)
    return rows["to_combo"].tolist()


def global_popular_candidates(df: pd.DataFrame, top_k: int = 50) -> list[str]:
    """Top-K most frequent combos across all users (by occurrence count)."""
    freq = df.groupby("combo_id").size().sort_values(ascending=False)
    return freq.head(top_k).index.tolist()


def user_cf_candidates(
    df: pd.DataFrame,
    user_id: str,
    top_k_users: int = 20,
    top_k_combos: int = 50,
) -> list[str]:
    """
    User-based collaborative filtering.
    Find users with the most overlapping combo history, collect their combos.

    For scale (100K users), this should be replaced with an ANN-based approach
    (e.g., sparse cosine similarity via sklearn or faiss). The API stays the same.
    """
    user_combos: dict[str, set] = (
        df.groupby("user_id")["combo_id"]
        .apply(set)
        .to_dict()
    )

    if user_id not in user_combos:
        return []

    target_set = user_combos[user_id]
    if not target_set:
        return []

    # Jaccard similarity with every other user
    scores: list[tuple[float, str]] = []
    for uid, combo_set in user_combos.items():
        if uid == user_id:
            continue
        intersection = len(target_set & combo_set)
        union        = len(target_set | combo_set)
        if union > 0:
            scores.append((intersection / union, uid))

    scores.sort(reverse=True)
    similar_users = [uid for _, uid in scores[:top_k_users]]

    # Collect combos from similar users that the target user hasn't seen
    seen = target_set
    candidate_freq: dict[str, int] = defaultdict(int)
    for uid in similar_users:
        for c in user_combos[uid]:
            if c not in seen:
                candidate_freq[c] += 1

    sorted_candidates = sorted(candidate_freq, key=candidate_freq.get, reverse=True)
    return sorted_candidates[:top_k_combos]


def build_candidate_pool(
    df_context: pd.DataFrame,
    user_id: str,
    transition_table: pd.DataFrame,
    global_popular: list[str],
    max_candidates: int = 150,
) -> list[str]:
    """
    Merge all recall sources into a ranked candidate pool (deduped).
    Order of insertion controls priority when pool is truncated.
    """
    seen: set[str] = set()
    pool: list[str] = []

    def add(combos: list[str]):
        for c in combos:
            if c not in seen and len(pool) < max_candidates:
                seen.add(c)
                pool.append(c)

    # 1. Personal history (highest priority)
    add(personal_history_candidates(df_context, user_id))

    # 2. Markov: based on user's last action
    user_df = df_context[df_context["user_id"] == user_id]
    if len(user_df) > 0:
        last_combo = user_df.sort_values("acs_tm")["combo_id"].iloc[-1]
        add(markov_candidates(last_combo, transition_table))

    # 3. CF from similar users
    add(user_cf_candidates(df_context, user_id))

    # 4. Global popular (fallback)
    add(global_popular)

    return pool
