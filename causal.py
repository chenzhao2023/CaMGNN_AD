"""Do-intervention utilities for CaMGNN."""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def get_node_percentiles(dataset: Sequence[Data]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if len(dataset) == 0:
        raise ValueError("Cannot compute node percentiles on an empty dataset.")

    region_names, region_maps = _align_regions(dataset)
    return _compute_region_percentiles(dataset, region_names, region_maps)


def _compute_region_percentiles(
    dataset: Sequence[Data],
    region_names: Sequence[str],
    region_maps: Sequence[Mapping[str, int]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    percentiles = {}
    for region_idx, region_name in enumerate(region_names):
        values = [
            data.x[region_map[region_name]].cpu().numpy()
            for data, region_map in zip(dataset, region_maps)
        ]
        features = np.stack(values)
        p10 = np.percentile(features, 10, axis=0)
        p90 = np.percentile(features, 90, axis=0)
        percentiles[region_idx] = (
            torch.tensor(p10, dtype=torch.float32),
            torch.tensor(p90, dtype=torch.float32),
        )
    return percentiles


def simulate_do_intervention_with_bootstrap(
    model: torch.nn.Module,
    dataset: Sequence[Data],
    device: torch.device,
    target_class: int = 1,
    bootstrap_samples: int = 1000,
    alpha: float = 0.05,
    intervention_feature_indices: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Estimate node-level average treatment effects with bootstrap CIs."""

    if len(dataset) == 0:
        raise ValueError("Cannot run causal intervention on an empty dataset.")

    model.eval()
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")

    region_names, region_maps = _align_regions(dataset)
    percentiles = _compute_region_percentiles(dataset, region_names, region_maps)
    feature_indices = _validate_feature_indices(dataset, intervention_feature_indices)
    causal_effects = []

    for target_region_idx, target_region_name in enumerate(region_names):
        p10, p90 = percentiles[target_region_idx]
        p10 = p10.to(device)
        p90 = p90.to(device)

        subject_effects = []
        subject_ites = []
        for data, region_map in zip(dataset, region_maps):
            target_node = region_map[target_region_name]
            edges = data.edge_index.clone()
            incoming_mask = edges[1] != target_node
            modified_edges = edges[:, incoming_mask].to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

            data_low = data.clone().to(device)
            data_low.edge_index = modified_edges
            if feature_indices is None:
                data_low.x[target_node] = p10
            else:
                data_low.x[target_node, feature_indices] = p10[feature_indices]

            data_high = data.clone().to(device)
            data_high.edge_index = modified_edges
            if feature_indices is None:
                data_high.x[target_node] = p90
            else:
                data_high.x[target_node, feature_indices] = p90[feature_indices]

            with torch.no_grad():
                logits_low = model(data_low.x, data_low.edge_index, batch, data_low.clin)
                logits_high = model(data_high.x, data_high.edge_index, batch, data_high.clin)
                prob_low = F.softmax(logits_low, dim=1)[0, target_class].item()
                prob_high = F.softmax(logits_high, dim=1)[0, target_class].item()

            ite = prob_high - prob_low
            subject_effects.append(ite)
            subject_ites.append(
                {
                    "subject_id": getattr(data, "subject_id", getattr(data, "image_id", "")),
                    "ite": ite,
                }
            )

        subject_effects_np = np.asarray(subject_effects)
        bootstrap_ates = []
        for _ in range(bootstrap_samples):
            indices = np.random.choice(
                len(subject_effects_np),
                size=len(subject_effects_np),
                replace=True,
            )
            bootstrap_ates.append(subject_effects_np[indices].mean())

        bootstrap_ates_np = np.asarray(bootstrap_ates)
        ate = subject_effects_np.mean()
        causal_effects.append(
            {
                "node_idx": target_region_idx,
                "region_name": target_region_name,
                "ate": ate,
                "abs_ace": abs(ate),
                "ci_lower": np.percentile(bootstrap_ates_np, 100 * (alpha / 2)),
                "ci_upper": np.percentile(bootstrap_ates_np, 100 * (1 - alpha / 2)),
                "ites": subject_ites,
            }
        )

    causal_effects.sort(key=lambda item: item["abs_ace"], reverse=True)
    return causal_effects


def _align_regions(dataset: Sequence[Data]) -> tuple[list[str], list[dict[str, int]]]:
    region_maps = []
    region_sets = []
    for data in dataset:
        names = list(getattr(data, "node_names", []))
        if len(names) != data.num_nodes:
            raise ValueError(
                f"Graph {getattr(data, 'image_id', '')} has {data.num_nodes} nodes "
                f"but {len(names)} node names."
            )
        if len(names) != len(set(names)):
            raise ValueError(
                f"Graph {getattr(data, 'image_id', '')} contains duplicate node names."
            )
        region_map = {name: idx for idx, name in enumerate(names)}
        region_maps.append(region_map)
        region_sets.append(set(region_map))

    common_regions = set.intersection(*region_sets)
    all_regions = set.union(*region_sets)
    excluded = sorted(all_regions - common_regions)
    if excluded:
        warnings.warn(
            "Causal analysis excludes regions not present in every graph: "
            + ", ".join(excluded),
            stacklevel=2,
        )

    first_names = list(getattr(dataset[0], "node_names"))
    region_names = [name for name in first_names if name in common_regions]
    if not region_names:
        raise ValueError("No brain regions are shared by all graphs.")
    return region_names, region_maps


def _validate_feature_indices(
    dataset: Sequence[Data],
    feature_indices: Sequence[int] | None,
) -> list[int] | None:
    if feature_indices is None:
        return None
    normalized = list(dict.fromkeys(int(index) for index in feature_indices))
    feature_dim = dataset[0].x.size(1)
    invalid = [index for index in normalized if index < 0 or index >= feature_dim]
    if invalid:
        raise ValueError(f"Invalid intervention feature indices for x_dim={feature_dim}: {invalid}")
    return normalized
