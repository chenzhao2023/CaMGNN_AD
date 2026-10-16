"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    preds: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    if len(loader.dataset) == 0:
        raise ValueError("Training loader is empty.")

    model.train()
    total_loss = 0.0
    total_correct = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch, data.clin)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_correct += (logits.argmax(dim=1) == data.y).sum().item()

    dataset_size = len(loader.dataset)
    return total_loss / dataset_size, total_correct / dataset_size


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    positive_class_index: int = 1,
) -> EvalResult:
    if len(loader.dataset) == 0:
        raise ValueError("Evaluation loader is empty.")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch, data.clin)
            loss = criterion(logits, data.y)
            probs = F.softmax(logits, dim=1)

            pred = logits.argmax(dim=1)
            total_loss += loss.item() * data.num_graphs
            total_correct += (pred == data.y).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs[:, positive_class_index].cpu().numpy())

    dataset_size = len(loader.dataset)
    return EvalResult(
        loss=total_loss / dataset_size,
        accuracy=total_correct / dataset_size,
        preds=np.asarray(all_preds),
        labels=np.asarray(all_labels),
        probabilities=np.asarray(all_probs),
    )
