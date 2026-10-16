"""Train CaMGNN on ADNI-style graph, clinical, and optional PET data."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, roc_curve
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camgnn import CaMGNN, load_compatible_state_dict
from camgnn.causal import simulate_do_intervention_with_bootstrap
from camgnn.data import (
    attach_clinical_and_pet,
    compute_norm_stats,
    filter_binary_dataset,
    load_split,
    normalize_node_features,
    prepare_pet_dataframe,
    validate_subject_disjoint_splits,
)
from camgnn.engine import evaluate, train_one_epoch
from camgnn.losses import FocalLoss
from camgnn.utils import compute_class_weights, ensure_dir, get_device, safe_binary_auc, save_json, set_seed


TASKS = {
    "CN_vs_AD": (0, 2),
    "CN_vs_MCI": (0, 1),
    "MCI_vs_AD": (1, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CaMGNN.")
    parser.add_argument("--split-file", type=Path, required=True, help="CSV/TSV with path and split columns.")
    parser.add_argument("--clinical-file", type=Path, required=True, help="Clinical CSV with image_id and covariates.")
    parser.add_argument("--pet-file", type=Path, default=None, help="Optional PET SUVR CSV.")
    parser.add_argument("--pet-subject-col", default="patient_id", help="PET subject ID column if subject_id is absent.")
    parser.add_argument("--clinical-image-id-col", default="image_id", help="Clinical table image ID column.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for checkpoints and metrics.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS), help="Binary tasks to run.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--clin-hidden", type=int, default=32)
    parser.add_argument("--num-strata", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic PyTorch operations.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--keep-missing-clinical", action="store_true", help="Keep samples without clinical rows as zeros.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore existing checkpoints.")
    parser.add_argument("--run-causal", action="store_true", help="Run do-intervention after test evaluation.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap samples for causal analysis.")
    parser.add_argument(
        "--causal-feature-indices",
        nargs="+",
        type=int,
        default=None,
        help="Node feature indices to perturb. Default: all features, matching the original script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")

    train_data, val_data, test_data, clin_preprocess, clin_dim, norm_mean, norm_std = prepare_datasets(args)
    save_preprocessing(output_dir, clin_preprocess, norm_mean, norm_std)

    x_dim = train_data[0].x.size(-1)
    model_kwargs = {
        "x_dim": x_dim,
        "clin_dim": clin_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": 2,
        "heads": args.heads,
        "dropout": args.dropout,
        "num_layers": args.num_layers,
        "clin_hidden": args.clin_hidden,
        "num_strata": args.num_strata,
    }

    print(f"Device: {device}")
    print(f"Input dimensions: node={x_dim}, clinical={clin_dim}")
    print(f"Dataset sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    metrics_rows = []
    for task_name in args.tasks:
        row = run_task(
            task_name=task_name,
            class_indices=TASKS[task_name],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            model_kwargs=model_kwargs,
            args=args,
            device=device,
            checkpoint_dir=checkpoint_dir,
            metrics_dir=metrics_dir,
        )
        metrics_rows.append(row)

    metrics_path = metrics_dir / "summary.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    save_json({"tasks": metrics_rows}, metrics_dir / "summary.json")
    print(f"\nSaved summary metrics to {metrics_path}")


def prepare_datasets(args: argparse.Namespace):
    train_data = load_split(args.split_file, "train")
    val_data = load_split(args.split_file, "val")
    test_data = load_split(args.split_file, "test")
    validate_subject_disjoint_splits(train=train_data, val=val_data, test=test_data)

    clinical_df = pd.read_csv(args.clinical_file)
    pet_df = None
    if args.pet_file is not None:
        pet_df = prepare_pet_dataframe(pd.read_csv(args.pet_file), subject_col=args.pet_subject_col)

    train_data, clin_preprocess, clin_dim = attach_clinical_and_pet(
        train_data,
        clinical_df,
        pet_df=pet_df,
        preprocess=None,
        fit_on_train=True,
        drop_missing=not args.keep_missing_clinical,
        image_id_col=args.clinical_image_id_col,
    )
    val_data, _, _ = attach_clinical_and_pet(
        val_data,
        clinical_df,
        pet_df=pet_df,
        preprocess=clin_preprocess,
        fit_on_train=False,
        drop_missing=not args.keep_missing_clinical,
        image_id_col=args.clinical_image_id_col,
    )
    test_data, _, _ = attach_clinical_and_pet(
        test_data,
        clinical_df,
        pet_df=pet_df,
        preprocess=clin_preprocess,
        fit_on_train=False,
        drop_missing=not args.keep_missing_clinical,
        image_id_col=args.clinical_image_id_col,
    )

    if not train_data or not val_data or not test_data:
        raise ValueError("At least one split is empty after clinical alignment.")

    norm_mean, norm_std = compute_norm_stats(train_data, num_node_features=5)
    normalize_node_features([train_data, val_data, test_data], norm_mean, norm_std, num_node_features=5)
    return train_data, val_data, test_data, clin_preprocess, clin_dim, norm_mean, norm_std


def save_preprocessing(output_dir: Path, clin_preprocess: dict[str, object], norm_mean: torch.Tensor, norm_std: torch.Tensor) -> None:
    with (output_dir / "clinical_preprocess.pkl").open("wb") as handle:
        pickle.dump(clin_preprocess, handle)
    torch.save(
        {
            "norm_mean": norm_mean.cpu(),
            "norm_std": norm_std.cpu(),
            "normalized_node_dims": 5,
        },
        output_dir / "node_normalization.pt",
    )


def run_task(
    task_name: str,
    class_indices: tuple[int, int],
    train_data: list,
    val_data: list,
    test_data: list,
    model_kwargs: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
    metrics_dir: Path,
) -> dict[str, object]:
    print(f"\n{'=' * 20} {task_name} {'=' * 20}")

    task_train = filter_binary_dataset(train_data, *class_indices)
    task_val = filter_binary_dataset(val_data, *class_indices)
    task_test = filter_binary_dataset(test_data, *class_indices)
    _validate_task_dataset(task_name, task_train, task_val, task_test)

    train_loader = DataLoader(task_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(task_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(task_test, batch_size=args.batch_size, shuffle=False)

    model = CaMGNN(**model_kwargs).to(device)
    class_weights = compute_class_weights(
        labels=[int(data.y.item()) for data in task_train],
        num_classes=2,
        device=device,
    )
    criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)

    checkpoint_path = checkpoint_dir / f"camgnn_{task_name}.pt"
    if checkpoint_path.exists() and not args.force_retrain:
        print(f"Found checkpoint: {checkpoint_path}. Loading it for evaluation.")
        load_checkpoint(model, checkpoint_path, device, model_kwargs)
    else:
        train_task(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            args=args,
            device=device,
            checkpoint_path=checkpoint_path,
            model_kwargs=model_kwargs,
            task_name=task_name,
            class_indices=class_indices,
        )
        load_checkpoint(model, checkpoint_path, device, model_kwargs)

    test_result = evaluate(model, test_loader, criterion, device)
    test_auc = safe_binary_auc(test_result.labels, test_result.probabilities)
    print(f"[{task_name}] Test Acc: {test_result.accuracy:.4f} | Test AUC: {format_auc(test_auc)}")

    class_names = task_name.split("_vs_")
    report = classification_report(
        test_result.labels,
        test_result.preds,
        labels=[0, 1],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)
    (metrics_dir / f"classification_report_{task_name}.txt").write_text(report, encoding="utf-8")

    if test_auc is not None:
        fpr, tpr, _ = roc_curve(test_result.labels, test_result.probabilities)
        np.savez(metrics_dir / f"roc_{task_name}.npz", fpr=fpr, tpr=tpr, auc=test_auc)

    if args.run_causal:
        save_causal_outputs(
            model=model,
            task_name=task_name,
            task_test=task_test,
            device=device,
            bootstrap_samples=args.bootstrap_samples,
            intervention_feature_indices=args.causal_feature_indices,
            metrics_dir=metrics_dir,
        )

    return {
        "task": task_name,
        "train_size": len(task_train),
        "val_size": len(task_val),
        "test_size": len(task_test),
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "test_auc": test_auc,
        "checkpoint": str(checkpoint_path),
    }


def train_task(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    model_kwargs: dict[str, object],
    task_name: str,
    class_indices: tuple[int, int],
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=20,
    )

    best_val_acc = -1.0
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_result.accuracy)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{task_name}] Epoch {epoch:03d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_result.accuracy:.4f}"
            )

        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "task_name": task_name,
                    "class_indices": class_indices,
                    "best_val_accuracy": best_val_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[{task_name}] Early stopping at epoch {epoch}.")
                break


def save_causal_outputs(
    model: torch.nn.Module,
    task_name: str,
    task_test: list,
    device: torch.device,
    bootstrap_samples: int,
    intervention_feature_indices: list[int] | None,
    metrics_dir: Path,
) -> None:
    rankings = simulate_do_intervention_with_bootstrap(
        model,
        task_test,
        device,
        target_class=1,
        bootstrap_samples=bootstrap_samples,
        intervention_feature_indices=intervention_feature_indices,
    )
    regions = getattr(task_test[0], "node_names", [])
    summary_rows = []
    ite_rows = []
    for rank, item in enumerate(rankings, start=1):
        node_idx = int(item["node_idx"])
        region_name = str(item.get("region_name", regions[node_idx] if node_idx < len(regions) else f"Node_{node_idx}"))
        summary_rows.append(
            {
                "rank": rank,
                "node_idx": node_idx,
                "region_name": region_name,
                "ate": item["ate"],
                "abs_ace": item["abs_ace"],
                "ci_lower": item["ci_lower"],
                "ci_upper": item["ci_upper"],
            }
        )
        for ite in item["ites"]:
            ite_rows.append(
                {
                    "node_idx": node_idx,
                    "region_name": region_name,
                    "subject_id": ite["subject_id"],
                    "ite": ite["ite"],
                }
            )

    pd.DataFrame(summary_rows).to_csv(metrics_dir / f"causal_summary_{task_name}.csv", index=False)
    pd.DataFrame(ite_rows).to_csv(metrics_dir / f"causal_ite_{task_name}.csv", index=False)


def _validate_task_dataset(task_name: str, train_data: list, val_data: list, test_data: list) -> None:
    sizes = {"train": len(train_data), "val": len(val_data), "test": len(test_data)}
    empty = [name for name, size in sizes.items() if size == 0]
    if empty:
        raise ValueError(f"{task_name} has empty splits after binary filtering: {empty}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    expected_model_kwargs: dict[str, object],
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        saved_kwargs = checkpoint.get("model_kwargs")
        if saved_kwargs is not None and saved_kwargs != expected_model_kwargs:
            differences = {
                key: {"saved": saved_kwargs.get(key), "current": expected_model_kwargs.get(key)}
                for key in set(saved_kwargs) | set(expected_model_kwargs)
                if saved_kwargs.get(key) != expected_model_kwargs.get(key)
            }
            raise ValueError(
                f"Checkpoint {checkpoint_path} was created with different model settings: "
                f"{differences}. Use --force-retrain or a different --output-dir."
            )
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    load_compatible_state_dict(model, state_dict)


def format_auc(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
