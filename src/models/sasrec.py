"""
SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Each unique combo_id is treated as an "item". The model is trained on
training users' action sequences to predict the next item given the past L items.

Architecture:
  Embedding → L × (Multi-head Self-Attention + FFN + LayerNorm) → Linear head

Training objective: binary cross-entropy (sampled negatives).

Usage:
  model = SASRec(n_items, hidden_dim, n_heads, n_layers, max_len)
  model.fit(sequences)                        # train
  scores = model.predict_scores(seq, vocab)   # score all items for a user
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SASRecModel(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len + 1, hidden_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq : (batch, seq_len) — item indices, 0=pad
        returns : (batch, hidden_dim) — representation of the last valid position
        """
        B, L = seq.shape
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        positions = positions * (seq > 0).long()   # zero out pad positions

        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)

        # causal mask: each position only attends to past positions
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)

        # take the last non-padding position as user representation
        seq_len = (seq > 0).sum(dim=1).clamp(min=1) - 1   # (B,)
        last_hidden = x[torch.arange(B), seq_len]           # (B, hidden_dim)
        return last_hidden

    def score_all_items(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Returns (batch, n_items+1) dot-product scores against all item embeddings.
        """
        hidden = self.forward(seq)          # (B, H)
        all_emb = self.item_emb.weight      # (n_items+1, H)
        return hidden @ all_emb.T           # (B, n_items+1)


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], max_len: int, n_items: int):
        self.samples   = []
        self.max_len   = max_len
        self.n_items   = n_items
        for seq in sequences:
            # sliding window: predict each position from its prefix
            for end in range(1, len(seq)):
                inp    = seq[max(0, end - max_len): end]
                target = seq[end]
                self.samples.append((inp, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, target = self.samples[idx]
        # left-pad to max_len
        padded = [0] * (self.max_len - len(inp)) + inp
        return torch.tensor(padded, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SASRec:
    """High-level wrapper around SASRecModel for training and inference."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        lr: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 256,
        n_negatives: int = 1,
        device: str = "auto",
    ):
        self.hidden_dim  = hidden_dim
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.max_len     = max_len
        self.dropout     = dropout
        self.lr          = lr
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.n_negatives = n_negatives
        self.device      = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.vocab: dict[str, int] = {}
        self.model: SASRecModel | None = None

    # ── public API ──────────────────────────────────────────────

    def fit(self, user_sequences: dict[str, list[str]]):
        """
        user_sequences: {user_id: [combo_id_0, combo_id_1, ...]} ordered by time.
        Builds vocab from all combo_ids seen, then trains the model.
        """
        all_combos = sorted({c for seq in user_sequences.values() for c in seq})
        self.vocab = {c: i + 1 for i, c in enumerate(all_combos)}   # 0 = pad
        n_items = len(self.vocab)

        int_sequences = [
            [self.vocab[c] for c in seq]
            for seq in user_sequences.values()
            if len(seq) >= 2
        ]

        dataset = SequenceDataset(int_sequences, self.max_len, n_items)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SASRecModel(
            n_items, self.hidden_dim, self.n_heads,
            self.n_layers, self.max_len, self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for seqs, targets in loader:
                seqs    = seqs.to(self.device)
                targets = targets.to(self.device)
                logits  = self.model.score_all_items(seqs)  # (B, n_items+1)
                loss    = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  SASRec epoch {epoch+1}/{self.n_epochs}  loss={total_loss/len(loader):.4f}")

        self.model.eval()

    def predict_scores(self, sequence: list[str]) -> dict[str, float]:
        """
        Given a user's sequence of combo_ids (ordered by time),
        returns a dict {combo_id: score} for all known combos.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_scores().")

        int_seq = [self.vocab[c] for c in sequence if c in self.vocab]
        if not int_seq:
            return {}

        padded = [0] * max(0, self.max_len - len(int_seq)) + int_seq[-self.max_len:]
        tensor = torch.tensor([padded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model.score_all_items(tensor)[0]   # (n_items+1,)

        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return {
            reverse_vocab[i]: float(logits[i])
            for i in range(1, len(self.vocab) + 1)
            if i in reverse_vocab
        }
