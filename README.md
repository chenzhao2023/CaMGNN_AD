# CaMGNN

CaMGNN is a causal multimodal graph neural network for Alzheimer's disease classification. The code trains binary classifiers on brain graphs, clinical covariates, and optional PET SUVR regional features.

## Project Structure

```text
CaMGNN/
├── camgnn/
│   ├── data.py          # graph loading, split loading, clinical/PET alignment
│   ├── models.py        # CaMGNN model and backdoor causal module
│   ├── losses.py        # focal loss
│   ├── engine.py        # training and evaluation loops
│   ├── causal.py        # do-intervention/bootstrap analysis
│   └── utils.py
├── scripts/
│   └── train.py         # command-line training entry
├── requirements.txt
└── README.md
```

## Installation

The code is verified with Python 3.9.23, PyTorch 2.8.0+cu128, and PyTorch Geometric 2.6.1. Install a PyTorch build that matches your CUDA setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If `torch-geometric` installation fails, install it following the official PyG wheel selector for your PyTorch and CUDA versions.

## Data Format

### Split file

`--split-file` should be a CSV/TSV file containing at least:

```csv
path,split,subject
/path/to/I114210.pkl,train,002_S_0413
/path/to/I223344.pkl,val,003_S_0908
```

Required columns:

- `path`: path to a pickled `networkx.Graph`
- `split`: one of `train`, `val`, `test`

Optional column:

- `subject`: ADNI-style subject ID. If absent, CaMGNN tries to infer IDs like `002_S_0413` from the graph path.

Each graph pickle should contain:

- graph label in `graph.graph["label"]`, one of `CN`, `MCI`, `AD`
- node attribute `feature`, defaulting to 2 dimensions
- node attribute `centroid_mm`, defaulting to 3 dimensions
- optional node attribute `name`, used to match PET columns

Only load graph pickle files from a trusted source. Python pickle files can execute code while being deserialized.

### Clinical file

`--clinical-file` should contain `image_id` by default plus these columns:

```text
Age, PTEDUCAT, APOE4, PTGENDER, PTETHCAT, PTRACCAT, PTMARRY
```

If your image ID column has a different name, pass `--clinical-image-id-col`.

### PET file

PET is optional. If provided, the table should contain `subject_id` or `patient_id`, plus regional SUVR columns named like:

```text
SUVR__Left_Hippocampus
SUVR__Right_Hippocampus
```

Node names are mapped by replacing `-` with `_`, for example `Left-Hippocampus` maps to `SUVR__Left_Hippocampus`. CaMGNN appends two node features: PET SUVR and PET availability mask.

## Training

Run all three binary tasks:

```bash
python scripts/train.py \
  --split-file /path/to/splits.csv \
  --clinical-file /path/to/multimodal_ready_cleaned.csv \
  --pet-file /path/to/pet_msum_suvr.csv \
  --output-dir outputs
```

Run one task:

```bash
python scripts/train.py \
  --split-file /path/to/splits.csv \
  --clinical-file /path/to/multimodal_ready_cleaned.csv \
  --pet-file /path/to/pet_msum_suvr.csv \
  --tasks CN_vs_AD
```

Useful options:

```bash
--epochs 200
--batch-size 16
--lr 5e-4
--hidden-dim 64
--dropout 0.4
--force-retrain
--device cuda
--deterministic
```

## Causal Analysis

To run node-level do-intervention analysis after testing:

```bash
python scripts/train.py \
  --split-file /path/to/splits.csv \
  --clinical-file /path/to/multimodal_ready_cleaned.csv \
  --pet-file /path/to/pet_msum_suvr.csv \
  --tasks CN_vs_AD \
  --run-causal \
  --bootstrap-samples 1000 \
  --causal-feature-indices 0 1 5
```

Without `--causal-feature-indices`, the intervention changes all seven node dimensions to match the original research script. For a biological-feature-only sensitivity analysis, indices `0 1 5` perturb the two MRI features and PET SUVR while leaving centroid coordinates and the PET availability mask unchanged. Report the selected setting in the manuscript.

## Outputs

By default, files are written under `outputs/`:

```text
outputs/
├── checkpoints/
│   └── camgnn_CN_vs_AD.pt
├── metrics/
│   ├── summary.csv
│   ├── summary.json
│   ├── classification_report_CN_vs_AD.txt
│   └── roc_CN_vs_AD.npz
├── clinical_preprocess.pkl
└── node_normalization.pt
```

## Notes for Release

This repository contains code only. Do not commit private clinical tables, PET tables, graph pickles, trained checkpoints, or intermediate outputs unless they are explicitly approved for public release.
