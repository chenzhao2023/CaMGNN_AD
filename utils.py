"""Small utilities shared by scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_class_weights(labels: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    labels = np.asarray(list(labels), dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot compute class weights for an empty label list.")
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    weights = labels.size / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def safe_binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probabilities))


def save_json(data: object, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
