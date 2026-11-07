"""Unified training loop — BCE only, dengan early stopping."""
from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict


def bce_loss(logits, targets, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)


def train_model(
    model: nn.Module,
    train_data: Dict[str, torch.Tensor],
    A: torch.Tensor,
    U: torch.Tensor,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 5e-3,
    patience: int = 30,
    is_graph_model: bool = True,
    device: str = "cpu",
) -> Dict[str, list]:
    """Train with early stopping on train loss (proxy).

    Note: real early stopping should use val set. For simplicity
    we use train loss delta — kalau train loss stagnan, kita stop.
    """
    model.to(device)
    A, U = A.to(device), U.to(device)

    ds = TensorDataset(
        train_data["concepts"],
        train_data["responses"],
        train_data["mask"],
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"loss": [], "time": []}
    best_loss = float("inf")
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0

        for c, r, m in dl:
            c, r, m = c.to(device), r.to(device), m.to(device)
            opt.zero_grad()

            if is_graph_model:
                out = model(c, r, A, U)
                logits = out["logits"]
            else:
                logits = model(c, r)

            loss = bce_loss(logits, r, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * c.size(0)
            n += c.size(0)

        ep_loss = tot / n
        hist["loss"].append(ep_loss)
        hist["time"].append(time.time() - t0)

        if ep % 10 == 0 or ep == 1:
            print(f"[ep {ep:03d}] loss={ep_loss:.4f} ({hist['time'][-1]:.1f}s)")

        # Early stopping
        if ep_loss < best_loss - 1e-4:
            best_loss = ep_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [early stop at ep {ep}] loss stagnan {patience} epoch")
                break

    return hist


@torch.no_grad()
def predict(model, data, A, U, is_graph_model=True, device="cpu"):
    model.eval().to(device)
    A, U = A.to(device), U.to(device)
    c = data["concepts"].to(device)
    r = data["responses"].to(device)
    m = data["mask"].to(device)

    if is_graph_model:
        out = model(c, r, A, U)
        logits = out["logits"]
    else:
        logits = model(c, r)

    return torch.sigmoid(logits).cpu().numpy(), r.cpu().numpy(), m.cpu().numpy()