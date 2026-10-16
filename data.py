"""Data loading and multimodal feature preparation for CaMGNN."""

from __future__ import annotations

import csv
import os
import pickle
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
BASE_NODE_FEATURE_DIM = 5
DEFAULT_CLIN_NUM_COLS = ["Age", "PTEDUCAT", "APOE4"]
DEFAULT_CLIN_CAT_COLS = ["PTGENDER", "PTETHCAT", "PTRACCAT", "PTMARRY"]


def normalize_subject_id(value: object) -> str:
    """Normalize ADNI-style subject IDs such as 002_S_0413."""

    if value is None:
        return ""
    text = str(value).strip()
    match = re.search(r"\d{3}_S_\d{4}", text)
    return match.group(0) if match else text


def infer_subject_from_path(path: str | Path) -> str:
    return normalize_subject_id(path)


def load_graph(pkl_path: str | Path, label_map: Mapping[str, int] = LABEL_MAP) -> Data:
    """Load one NetworkX graph pickle and convert it to a PyG Data object."""

    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as handle:
        graph = pickle.load(handle)

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{pkl_path} is not a networkx.Graph instance.")

    num_nodes = graph.number_of_nodes()
    if not _has_zero_based_integer_nodes(graph):
        mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)

    edge_list = list(graph.edges)
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    node_features = []
    node_names = []
    for node_idx in range(num_nodes):
        node_attr = graph.nodes[node_idx]
        feature = np.asarray(node_attr.get("feature", [0, 0]), dtype=np.float32).reshape(-1)
        centroid = np.asarray(node_attr.get("centroid_mm", [0, 0, 0]), dtype=np.float32).reshape(-1)
        if feature.size != 2 or centroid.size != 3:
            raise ValueError(
                f"{pkl_path} node {node_idx} must have a 2-D `feature` and "
                f"3-D `centroid_mm`; got {feature.size} and {centroid.size}."
            )
        node_features.append(np.concatenate([feature, centroid]))
        node_names.append(node_attr.get("name", "Unknown_Region"))

    x = torch.from_numpy(np.asarray(node_features, dtype=np.float32))

    if edge_index.numel() > 0 and edge_index.max().item() >= num_nodes:
        raise ValueError(
            f"{pkl_path} has invalid edge_index: "
            f"max={edge_index.max().item()} >= num_nodes={num_nodes}"
        )

    label_name = graph.graph.get("label")
    if label_name is None:
        raise ValueError(f"{pkl_path} is missing graph-level label.")
    if label_name not in label_map:
        raise ValueError(f"{pkl_path} has unknown label {label_name!r}.")

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label_map[label_name]], dtype=torch.long),
    )
    data.image_id = pkl_path.stem
    data.label_name = label_name
    data.node_names = node_names
    return data


def load_split(split_file: str | Path, split_name: str) -> list[Data]:
    """Load all graphs from one split listed in a CSV/TSV file.

    The split file must contain `path` and `split` columns. A `subject` column is
    optional; otherwise the subject ID is inferred from the graph path.
    """

    split_file = Path(split_file)
    dataset = []
    with split_file.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(handle, dialect=dialect)

        _require_columns(reader.fieldnames, {"path", "split"}, str(split_file))
        target_split = split_name.strip().lower()

        for row in reader:
            if row["split"].strip().lower() != target_split:
                continue

            graph_path = _resolve_graph_path(row["path"].strip(), split_file)
            data = load_graph(graph_path)
            split_label = row.get("label", "").strip()
            if split_label and split_label != data.label_name:
                raise ValueError(
                    f"Label mismatch for {graph_path}: split={split_label!r}, "
                    f"graph={data.label_name!r}."
                )
            if row.get("subject"):
                data.subject_id = normalize_subject_id(row["subject"])
            else:
                data.subject_id = infer_subject_from_path(graph_path)
            dataset.append(data)

    return dataset


def filter_binary_dataset(
    dataset: Iterable[Data],
    negative_class_idx: int,
    positive_class_idx: int,
) -> list[Data]:
    """Filter a three-class dataset into one binary task and relabel to 0/1."""

    filtered = []
    for data in dataset:
        label = int(data.y.item())
        if label == negative_class_idx:
            item = data.clone()
            item.y = torch.tensor([0], dtype=torch.long)
            filtered.append(item)
        elif label == positive_class_idx:
            item = data.clone()
            item.y = torch.tensor([1], dtype=torch.long)
            filtered.append(item)
    return filtered


def compute_norm_stats(dataset: Sequence[Data], num_node_features: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute train-set normalization statistics for MRI/spatial node features."""

    if len(dataset) == 0:
        raise ValueError("Cannot compute normalization statistics on an empty dataset.")
    features = torch.cat([data.x[:, :num_node_features] for data in dataset], dim=0)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    return mean, std


def normalize_node_features(
    datasets: Iterable[Iterable[Data]],
    mean: torch.Tensor,
    std: torch.Tensor,
    num_node_features: int = 5,
) -> None:
    """Normalize the first `num_node_features` dimensions in-place."""

    for dataset in datasets:
        for data in dataset:
            data.x[:, :num_node_features] = (data.x[:, :num_node_features] - mean) / std


def fit_clinical_preprocess(
    train_df: pd.DataFrame,
    num_cols: Sequence[str] = DEFAULT_CLIN_NUM_COLS,
    cat_cols: Sequence[str] = DEFAULT_CLIN_CAT_COLS,
) -> dict[str, object]:
    """Fit simple numeric z-score and categorical one-hot preprocessing."""

    _require_columns(train_df.columns, set(num_cols) | set(cat_cols), "clinical table")
    if train_df.empty:
        raise ValueError("No clinical rows matched the training image IDs.")

    numeric = train_df[list(num_cols)].apply(pd.to_numeric, errors="coerce")
    median = numeric.median(numeric_only=True)
    numeric = numeric.fillna(median)
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0).replace(0, np.nan).fillna(1.0)

    categorical = train_df[list(cat_cols)].astype("string").fillna("UNK")
    dummies = pd.get_dummies(categorical, columns=list(cat_cols))

    return {
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
        "num_median": median,
        "num_mean": mean,
        "num_std": std,
        "dummy_cols": list(dummies.columns),
    }


def transform_clinical(df: pd.DataFrame, preprocess: Mapping[str, object]) -> np.ndarray:
    """Apply a fitted clinical preprocessing dictionary."""

    num_cols = list(preprocess["num_cols"])
    cat_cols = list(preprocess["cat_cols"])
    _require_columns(df.columns, set(num_cols) | set(cat_cols), "clinical table")

    numeric = df[num_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(preprocess["num_median"])
    numeric = (numeric - preprocess["num_mean"]) / (preprocess["num_std"] + 1e-6)

    categorical = df[cat_cols].astype("string").fillna("UNK")
    dummies = pd.get_dummies(categorical, columns=cat_cols)
    dummies = dummies.reindex(columns=preprocess["dummy_cols"], fill_value=0)

    return np.concatenate(
        [numeric.values.astype(np.float32), dummies.values.astype(np.float32)],
        axis=1,
    )


def prepare_pet_dataframe(
    pet_df: pd.DataFrame | None,
    subject_col: str = "patient_id",
) -> pd.DataFrame | None:
    """Add or normalize the `subject_id` column used for PET alignment."""

    if pet_df is None:
        return None

    pet_df = pet_df.copy()
    if "subject_id" in pet_df.columns:
        pet_df["subject_id"] = pet_df["subject_id"].astype(str).apply(normalize_subject_id)
    elif subject_col in pet_df.columns:
        pet_df["subject_id"] = pet_df[subject_col].astype(str).apply(normalize_subject_id)
    else:
        raise ValueError(
            "PET table must contain either `subject_id` or "
            f"the configured subject column {subject_col!r}."
        )

    duplicated = pet_df["subject_id"].duplicated(keep=False) & pet_df["subject_id"].ne("")
    if duplicated.any():
        examples = sorted(pet_df.loc[duplicated, "subject_id"].unique())[:5]
        raise ValueError(
            "PET table has multiple rows for the same subject. Select the "
            f"correct visit before training. Examples: {examples}"
        )
    return pet_df


def attach_clinical_and_pet(
    dataset: Sequence[Data],
    clinical_df: pd.DataFrame,
    pet_df: pd.DataFrame | None = None,
    preprocess: Mapping[str, object] | None = None,
    fit_on_train: bool = False,
    drop_missing: bool = True,
    image_id_col: str = "image_id",
    num_cols: Sequence[str] = DEFAULT_CLIN_NUM_COLS,
    cat_cols: Sequence[str] = DEFAULT_CLIN_CAT_COLS,
) -> tuple[list[Data], Mapping[str, object], int]:
    """Attach graph-level clinical features and node-level PET SUVR/masks."""

    required = {image_id_col} | set(num_cols) | set(cat_cols)
    _require_columns(clinical_df.columns, required, "clinical table")

    clinical_df = clinical_df.copy()
    clinical_df[image_id_col] = clinical_df[image_id_col].astype(str).str.strip()
    duplicated = clinical_df[image_id_col].duplicated(keep=False)
    if duplicated.any():
        examples = sorted(clinical_df.loc[duplicated, image_id_col].unique())[:5]
        raise ValueError(f"Clinical table has duplicate image IDs. Examples: {examples}")
    clinical_df = clinical_df[[image_id_col, *num_cols, *cat_cols]].rename(
        columns={image_id_col: "image_id"}
    )

    image_ids = [data.image_id for data in dataset]
    matched = clinical_df[clinical_df["image_id"].isin(image_ids)].copy()

    if fit_on_train:
        preprocess = fit_clinical_preprocess(matched, num_cols=num_cols, cat_cols=cat_cols)
    if preprocess is None:
        raise ValueError("A fitted clinical preprocess dictionary is required.")

    clinical_features = transform_clinical(matched, preprocess)
    id_to_feature = {
        image_id: clinical_features[idx]
        for idx, image_id in enumerate(matched["image_id"].tolist())
    }
    clin_dim = len(preprocess["num_cols"]) + len(preprocess["dummy_cols"])

    pet_lookup = _build_pet_lookup(pet_df)
    kept = []
    missing = 0
    for data in dataset:
        if data.x.size(1) != BASE_NODE_FEATURE_DIM:
            raise ValueError(
                f"Graph {data.image_id} has {data.x.size(1)} node features before PET "
                f"attachment; expected {BASE_NODE_FEATURE_DIM}. The graph may have "
                "already been processed."
            )
        clinical_vector = id_to_feature.get(data.image_id)
        if clinical_vector is None:
            missing += 1
            if drop_missing:
                continue
            clinical_vector = np.zeros((clin_dim,), dtype=np.float32)

        data.clin = torch.tensor(clinical_vector, dtype=torch.float32).view(1, -1)
        pet_features, pet_mask = _pet_features_for_graph(data, pet_lookup)
        data.x = torch.cat([data.x, pet_features, pet_mask], dim=1)
        kept.append(data)

    print(f"[Clinical Align] kept {len(kept)}/{len(dataset)}, missing {missing}")
    return kept, preprocess, clin_dim


def validate_subject_disjoint_splits(**splits: Sequence[Data]) -> None:
    """Raise if a subject appears in more than one data split."""

    subject_sets = {
        name: {getattr(data, "subject_id", "") for data in dataset}
        - {""}
        for name, dataset in splits.items()
    }
    split_names = list(subject_sets)
    for left_idx, left_name in enumerate(split_names):
        for right_name in split_names[left_idx + 1 :]:
            overlap = subject_sets[left_name] & subject_sets[right_name]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"Subject leakage between {left_name} and {right_name}: "
                    f"{len(overlap)} overlapping subjects. Examples: {examples}"
                )


def _has_zero_based_integer_nodes(graph: nx.Graph) -> bool:
    nodes = list(graph.nodes())
    return all(isinstance(node, int) for node in nodes) and set(nodes) == set(range(len(nodes)))


def _resolve_graph_path(raw_path: str, split_file: Path) -> Path:
    graph_path = Path(raw_path).expanduser()
    if not graph_path.is_absolute():
        graph_path = split_file.resolve().parent / graph_path
    return graph_path


def _require_columns(columns: Iterable[str] | None, required: set[str], table_name: str) -> None:
    if columns is None:
        raise ValueError(f"{table_name} has no header.")
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _build_pet_lookup(pet_df: pd.DataFrame | None) -> dict[str, pd.Series]:
    if pet_df is None:
        return {}
    _require_columns(pet_df.columns, {"subject_id"}, "PET table")
    lookup = {}
    for _, row in pet_df.iterrows():
        subject_id = normalize_subject_id(row["subject_id"])
        if subject_id and subject_id not in lookup:
            lookup[subject_id] = row
    return lookup


def _pet_features_for_graph(
    data: Data,
    pet_lookup: Mapping[str, pd.Series],
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = data.x.shape[0]
    pet_features = torch.zeros((num_nodes, 1), dtype=torch.float32)
    pet_mask = torch.zeros((num_nodes, 1), dtype=torch.float32)

    subject_id = getattr(data, "subject_id", "")
    pet_row = pet_lookup.get(subject_id)
    if pet_row is None:
        return pet_features, pet_mask

    node_names = getattr(data, "node_names", ["Unknown_Region"] * num_nodes)
    for node_idx, node_name in enumerate(node_names):
        column = f"SUVR__{str(node_name).replace('-', '_')}"
        if column in pet_row.index and pd.notna(pet_row[column]):
            pet_features[node_idx, 0] = float(pet_row[column])
            pet_mask[node_idx, 0] = 1.0

    return pet_features, pet_mask


# Backward-compatible name from the original script.
attach_clinical_features = attach_clinical_and_pet
