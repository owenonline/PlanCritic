#!/usr/bin/env python3
"""
Reward‑model training script with adaptive device selection:
CUDA ▸ MPS ▸ CPU (in that order of preference).

Usage (unchanged):
    python train_reward_model.py --domain some_domain
"""
import os
import argparse
import json
from collections import defaultdict
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import wandb

# PREFIX = "/Users/owenburns/workareas/Carnegie Mellon PlanCritic/PlanCritic/"

# ─────────────────────────── Device helpers ────────────────────────────
def pick_device() -> torch.device:
    """
    Pick the best available device in priority order:
    1. CUDA (one or many GPUs)
    2. Apple‑Silicon MPS
    3. CPU
    """
    if torch.cuda.is_available():
        print(f"[device] CUDA detected ‑ using {torch.cuda.device_count()} GPU(s)")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[device] Apple MPS detected ‑ using Apple GPU")
        return torch.device("mps")
    print("[device] Falling back to CPU")
    return torch.device("cpu")

DEVICE = pick_device()            # Global handle
USE_CUDA = DEVICE.type == "cuda"
DDP_BACKEND = "nccl" if USE_CUDA else "gloo"
BALANCE_REG_STRENGTH = 0.02

# ─────────────────────────── Arg‑parsing ───────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, required=True)
parser.add_argument("--prefix", type=str, default="/workspace/")
args = parser.parse_args()
DOMAIN = args.domain

OUTPUT_DIR = os.path.join(args.prefix, "domains", args.domain, "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────────────  Model definition  ────────────────────────
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.3
        )
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if any(w in name for w in ("weight_ih", "weight_hh")):
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, x):
        """Forward pass
        Args:
            x (Tensor): shape (batch, seq_len, embed_dim); rows that are all‑zeros are padding.
        Returns:
            Tensor: adherence probability for each plan (batch, 1)
        """
        # --- build a length tensor: how many *non‑zero* timesteps per plan
        lengths = (x.abs().sum(-1) > 0).sum(-1)            # (batch,)

        # --- pack the batch so the LSTM only sees real steps
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # --- LSTM over packed sequence
        _, (hn, _) = self.lstm(packed)

        # --- Classification head
        x = self.fc1(hn[-1])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# ───────────────────────────  Main training  ───────────────────────────
def main():
    print("PyTorch:", torch.__version__)
    print("Device :", DEVICE)

    wandb.login(key="57e77180c09aaee0b682991aef27ddc5ea72d9cb")
    os.environ["WANDB_LOG_MODEL"] = "True"
    wandb.init(project="huggingface", entity="owenonline")

    # Torch tensors
    data = np.load(f"{args.prefix}adherence_model_training/reward_model_embedded_data/{DOMAIN}.npz")
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=1024, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=1024, shuffle=False
    )

    model = LSTMNetwork(
        input_dim=X.shape[2],
        hidden_dim=512, output_dim=1, num_layers=2
    )

    # Only wrap in DataParallel when running on CUDA with >1 GPU
    if USE_CUDA and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    wandb.watch(model, log="all")

    best_loss, best_acc = float("inf"), 0.0
    EPOCH_BLOCKS, INNER_EPOCHS = 20, 5

    for block in range(EPOCH_BLOCKS):
        model.train()
        for epoch in range(INNER_EPOCHS):
            running = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()

                out   = model(bx)                           # (batch, 1) in (0,1)
                ce    = criterion(out, by)                  # usual BCE

                # ────────────── entropy-style regularizer ──────────────-------------
                # high entropy (≈0.5) ⇒ small penalty, low entropy (≈0 or 1) ⇒ large
                eps = 1e-7                                   # avoid log(0)
                entropy = -(out * (out + eps).log() +
                           (1 - out) * (1 - out + eps).log())   # (batch,1)
                reg = BALANCE_REG_STRENGTH * (-entropy.mean())  #  ↑penalise low entropy
                # --------------------------------------------------------------------

                loss = ce + reg
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running += loss.item()

            wandb.log({
                "epoch": block * INNER_EPOCHS + epoch + 1,
                "train_loss": running / len(train_loader),
                "reg_weight": BALANCE_REG_STRENGTH
            })

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        vloss, correct, preds_true, preds_false = 0.0, 0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                vloss += criterion(out, by).item()
                preds = (out > 0.5).float()
                correct += (preds == by).sum().item()
                preds_true += preds.sum().item()
                preds_false += (preds == 0).sum().item()

        vloss /= len(val_loader)
        acc = correct / len(y_val)
        ratio = preds_true / (preds_false + 1e-6)
        wandb.log({"val_loss": vloss, "val_acc": acc, "true_false_ratio": ratio})

        # Save best
        if acc > best_acc and vloss < best_loss:
            best_acc, best_loss = acc, vloss
            path = os.path.join(OUTPUT_DIR, "best_lstm_model.pth")
            torch.save(model.state_dict(), path)
            wandb.save(path)

        print(f"[block {block+1}/{EPOCH_BLOCKS}] "
              f"val_loss={vloss:.4f}  acc={acc:.4f}  "
              f"best_loss={best_loss:.4f}  best_acc={best_acc:.4f}")

    # ── Final save ──────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_model.pth"))
    wandb.save(os.path.join(OUTPUT_DIR, "lstm_model.pth"))
    print("Training complete.")

if __name__ == "__main__":
    main()
