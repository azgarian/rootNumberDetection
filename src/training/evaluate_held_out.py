#!/usr/bin/env python3
"""
Evaluate trained models on held-out test set.

For each model (single backbone or ensemble):
- Load best checkpoints from all folds
- Run inference on held-out test data
- Compute metrics (including DeLong stats)
- Save results

Usage:
    # Single model evaluation
    python evaluate_held_out.py \
        --run-dir outputs/alexnet \
        --data-dir data/held_out_test/crops \
        --model alexnet \
        --outdir held_out_results/alexnet
    
    # Ensemble evaluation
    python evaluate_held_out.py \
        --run-dir outputs \
        --data-dir data/held_out_test/crops \
        --model ensemble \
        --models alexnet efficientnet_b0 densenet121 \
        --outdir held_out_results/ensemble
"""

import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Ensure we can import helper modules
THIS_DIR = Path(__file__).resolve().parent
SYS_SRC_TRAINING = THIS_DIR / "src" / "training"
if str(SYS_SRC_TRAINING) not in sys.path:
    sys.path.append(str(SYS_SRC_TRAINING))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from dataloader import create_transforms
from metrics import evaluate_and_store, _compute_delong_auc_stats, compute_confusion_metrics
from models import build_model

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

TOOTH_CODES = ["14", "15", "24", "25"]


def _parse_patient_and_tooth(p: Path) -> Tuple[str, str]:
    stem = p.stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {p}")
    patient_id, tooth_part = parts
    tooth = tooth_part.split("_", 1)[0]
    return patient_id, tooth


def _load_expert_predictions(csv_path: Path) -> Dict[Tuple[str, str], int]:
    mapping: Dict[Tuple[str, str], int] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = (row.get("patient") or row.get("patient_id") or "").strip()
            if not patient:
                continue
            for tooth in TOOTH_CODES:
                col = f"expert pred {tooth}"
                val = row.get(col)
                if val is None:
                    continue
                val = val.strip()
                if val in {"", "-"}:
                    continue
                try:
                    root_count = int(round(float(val)))
                except ValueError:
                    continue
                if root_count not in {1, 2}:
                    continue
                mapping[(patient, tooth)] = root_count
    return mapping


def _save_high_quality_plots(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, out_dir: Path, title: str = "Held-Out Test") -> None:
    """Save confusion matrix and ROC plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix (normalized)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontweight='bold', fontsize=20)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Single Root", "Double Root"], size=16)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Single Root", "Double Root"], size=16)
    for lab in ax.get_yticklabels():
        lab.set_rotation(90)
        lab.set_verticalalignment('center')
        lab.set_horizontalalignment('center')
    ax.tick_params(axis='y', pad=10, labelsize=16)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color='black', fontsize=18)
    ax.set_xlabel('Predicted Label', fontsize=18); ax.set_ylabel('True Label', fontsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_held_out_hd.png", dpi=450, bbox_inches='tight')
    fig.savefig(out_dir / "confusion_held_out_hd.svg")
    plt.close(fig)
    
    # ROC Curve
    if y_prob is not None and len(y_prob) == len(y_true):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_title(title, fontweight='bold', fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.set_xlabel('False Positive Rate', fontsize=18); ax2.set_ylabel('True Positive Rate', fontsize=18)
        ax2.legend(loc='lower right', fontsize=18)
        fig2.tight_layout()
        fig2.savefig(out_dir / "roc_held_out_hd.png", dpi=450, bbox_inches='tight')
        fig2.savefig(out_dir / "roc_held_out_hd.svg")
        plt.close(fig2)


def evaluate_single_model(
    run_dir: Path,
    data_dir: Path,
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 8,
    image_size: int = 224,
) -> Dict:
    """Evaluate a single model: compute metrics per fold, then aggregate."""
    print(f"\n=== Evaluating {model_name} on held-out test ===")
    
    # Build dataset
    _, val_tf = create_transforms(image_size=image_size, use_aug=False)
    base_ds = datasets.ImageFolder(root=str(data_dir), transform=val_tf)
    test_loader = DataLoader(base_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    # Get true labels and paths (same for all folds)
    y_true = np.array([int(label) for _, label in base_ds.samples], dtype=np.int64)
    paths = [Path(p) for p, _ in base_ds.samples]
    
    # Per-premolar indices
    import re
    def _extract_tooth(p: Path) -> Optional[str]:
        m = re.search(r"_(14|15|24|25)(?:_|\.|$)", p.name)
        return m.group(1) if m else None
    
    teeth = [_extract_tooth(p) for p in paths]
    idx_first = np.array([i for i, t in enumerate(teeth) if t in {"14", "24"}], dtype=int)
    idx_second = np.array([i for i, t in enumerate(teeth) if t in {"15", "25"}], dtype=int)
    
    # Find all folds
    fold_dirs = sorted([d for d in run_dir.glob("fold_*") if d.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {run_dir}")
    
    print(f"Found {len(fold_dirs)} folds")
    
    # Evaluate each fold separately
    per_fold_metrics: List[Dict] = []
    all_probs: List[np.ndarray] = []
    
    for fold_dir in fold_dirs:
        best_ckpt = fold_dir / "best.pt"
        if not best_ckpt.exists():
            print(f"[SKIP] {fold_dir.name}: missing best.pt")
            continue
        
        # Load model
        model = build_model(model_name, num_classes=2, pretrained=False).to(device)
        state = torch.load(best_ckpt, map_location=device)
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Disable in-place activations
        for m in model.modules():
            if isinstance(m, (nn.ReLU, nn.SiLU, nn.LeakyReLU)) and getattr(m, "inplace", False):
                m.inplace = False
        
        # Predict
        probs_list = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                probs_list.extend(probs.tolist())
        
        fold_probs = np.array(probs_list, dtype=np.float64)
        fold_preds = (fold_probs >= 0.5).astype(int)
        all_probs.append(fold_probs)
        
        # Compute metrics for this fold
        fold_metrics = compute_confusion_metrics(y_true, fold_preds, fold_probs)
        
        # Add DeLong stats
        delong_stats = _compute_delong_auc_stats(y_true, fold_probs)
        if delong_stats:
            fold_metrics.update(delong_stats)
        
        # Per-premolar metrics
        def sub(arr, idx):
            return arr[idx] if len(idx) > 0 else np.array([], dtype=arr.dtype)
        
        m_first = compute_confusion_metrics(sub(y_true, idx_first), sub(fold_preds, idx_first), sub(fold_probs, idx_first))
        m_second = compute_confusion_metrics(sub(y_true, idx_second), sub(fold_preds, idx_second), sub(fold_probs, idx_second))
        
        fold_metrics["by_premolar"] = {
            "first_14_24": m_first,
            "second_15_25": m_second,
        }
        
        per_fold_metrics.append(fold_metrics)
        print(f"[{fold_dir.name}] acc={fold_metrics.get('accuracy', float('nan')):.4f} auc={fold_metrics.get('auc', float('nan')):.4f}")
    
    if not per_fold_metrics:
        raise RuntimeError("No valid folds found")
    
    # Aggregate across folds (mean ± SD)
    def mean_sd(ms: List[Dict], key: str) -> tuple:
        vals = [float(m.get(key, float('nan'))) for m in ms if key in m]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))
    
    base_keys = ["accuracy", "f1", "sensitivity", "specificity", "ppv", "npv", "auc"]
    delong_keys = ["auc_delong", "auc_delong_std", "auc_delong_var", "auc_delong_ci_low", "auc_delong_ci_high"]
    all_keys = base_keys + delong_keys
    
    summary = {}
    for k in all_keys:
        mu, sd = mean_sd(per_fold_metrics, k)
        summary[k] = {"mean": mu, "sd": sd}
    
    # Per-premolar aggregation
    def premolar_mean_sd(subkey: str, metric: str) -> tuple:
        vals = []
        for m in per_fold_metrics:
            bp = m.get("by_premolar", {})
            sk = bp.get(subkey, {})
            if metric in sk and np.isfinite(float(sk[metric])):
                vals.append(float(sk[metric]))
        if not vals:
            return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))
    
    summary["by_premolar_avg"] = {
        "first_14_24": {k: {"mean": premolar_mean_sd("first_14_24", k)[0], "sd": premolar_mean_sd("first_14_24", k)[1]} for k in base_keys},
        "second_15_25": {k: {"mean": premolar_mean_sd("second_15_25", k)[0], "sd": premolar_mean_sd("second_15_25", k)[1]} for k in base_keys},
    }
    
    # Also compute averaged predictions for overall plots and DeLong comparisons
    # For models with multiple folds, we average probabilities across folds to get
    # one prediction per sample. This is required for DeLong's test, which compares
    # per-sample predictions between two models (both must have shape [N]).
    P = np.stack(all_probs, axis=1)  # [N, num_folds]
    avg_probs = P.mean(axis=1)  # [N] - one probability per sample (averaged across folds)
    avg_preds = (avg_probs >= 0.5).astype(int)
    
    return {
        "per_fold": per_fold_metrics,
        "summary": summary,
        "y_true": y_true,
        "y_pred": avg_preds,
        "y_prob": avg_probs,
        "paths": paths,
        "n_folds": len(per_fold_metrics),
    }


def evaluate_expert(
    csv_path: Path,
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    image_size: int = 224,
) -> Dict:
    print("\n=== Evaluating expert predictions on held-out test ===")
    predictions = _load_expert_predictions(csv_path)
    if not predictions:
        raise ValueError(f"No expert predictions found in {csv_path}")

    _, val_tf = create_transforms(image_size=image_size, use_aug=False)
    base_ds = datasets.ImageFolder(root=str(data_dir), transform=val_tf)
    test_loader = DataLoader(base_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    y_true = np.array([int(label) for _, label in base_ds.samples], dtype=np.int64)
    paths = [Path(p) for p, _ in base_ds.samples]

    expert_probs = np.zeros(len(paths), dtype=np.float64)
    expert_preds = np.zeros(len(paths), dtype=np.int64)

    for idx, path in enumerate(paths):
        patient_id, tooth = _parse_patient_and_tooth(path)
        key = (patient_id, tooth)
        if key not in predictions:
            raise ValueError(f"Missing expert prediction for {patient_id}_{tooth}")
        root_count = predictions[key]
        prob = 1.0 if root_count == 2 else 0.0
        expert_probs[idx] = prob
        expert_preds[idx] = int(prob >= 0.5)

    fold_metrics = compute_confusion_metrics(y_true, expert_preds, expert_probs)
    delong_stats = _compute_delong_auc_stats(y_true, expert_probs)
    if delong_stats:
        fold_metrics.update(delong_stats)

    import re
    def _extract_tooth(p: Path) -> Optional[str]:
        m = re.search(r"_(14|15|24|25)(?:_|\.|$)", p.name)
        return m.group(1) if m else None

    teeth = [_extract_tooth(p) for p in paths]
    idx_first = np.array([i for i, t in enumerate(teeth) if t in {"14", "24"}], dtype=int)
    idx_second = np.array([i for i, t in enumerate(teeth) if t in {"15", "25"}], dtype=int)

    def sub(arr, idx):
        return arr[idx] if len(idx) > 0 else np.array([], dtype=arr.dtype)

    fold_metrics["by_premolar"] = {
        "first_14_24": compute_confusion_metrics(sub(y_true, idx_first), sub(expert_preds, idx_first), sub(expert_probs, idx_first)),
        "second_15_25": compute_confusion_metrics(sub(y_true, idx_second), sub(expert_preds, idx_second), sub(expert_probs, idx_second)),
    }

    per_fold_metrics = [fold_metrics]

    def mean_sd(ms: List[Dict], key: str) -> tuple:
        vals = [float(m.get(key, float('nan'))) for m in ms if key in m]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    base_keys = ["accuracy", "f1", "sensitivity", "specificity", "ppv", "npv", "auc"]
    delong_keys = ["auc_delong", "auc_delong_std", "auc_delong_var", "auc_delong_ci_low", "auc_delong_ci_high"]
    summary = {}
    for k in base_keys + delong_keys:
        mu, sd = mean_sd(per_fold_metrics, k)
        summary[k] = {"mean": mu, "sd": sd}

    def premolar_mean_sd(subkey: str, metric: str) -> tuple:
        vals = []
        for m in per_fold_metrics:
            bp = m.get("by_premolar", {}).get(subkey, {})
            if metric in bp and np.isfinite(float(bp[metric])):
                vals.append(float(bp[metric]))
        if not vals:
            return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    summary["by_premolar_avg"] = {
        "first_14_24": {k: {"mean": premolar_mean_sd("first_14_24", k)[0], "sd": premolar_mean_sd("first_14_24", k)[1]} for k in base_keys},
        "second_15_25": {k: {"mean": premolar_mean_sd("second_15_25", k)[0], "sd": premolar_mean_sd("second_15_25", k)[1]} for k in base_keys},
    }

    avg_probs = expert_probs.copy()
    avg_preds = expert_preds.copy()

    return {
        "per_fold": per_fold_metrics,
        "summary": summary,
        "y_true": y_true,
        "y_pred": avg_preds,
        "y_prob": avg_probs,
        "paths": paths,
        "n_folds": 1,
    }


def evaluate_ensemble(
    run_dirs: List[Path],
    model_names: List[str],
    data_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 8,
    image_size: int = 224,
) -> Dict:
    """Evaluate ensemble: for each fold, average across models, then aggregate across folds."""
    print(f"\n=== Evaluating Ensemble ({', '.join(model_names)}) on held-out test ===")
    
    # Build dataset
    _, val_tf = create_transforms(image_size=image_size, use_aug=False)
    base_ds = datasets.ImageFolder(root=str(data_dir), transform=val_tf)
    test_loader = DataLoader(base_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    # Get true labels and paths (same for all folds)
    y_true = np.array([int(label) for _, label in base_ds.samples], dtype=np.int64)
    paths = [Path(p) for p, _ in base_ds.samples]
    
    # Per-premolar indices
    import re
    def _extract_tooth(p: Path) -> Optional[str]:
        m = re.search(r"_(14|15|24|25)(?:_|\.|$)", p.name)
        return m.group(1) if m else None
    
    teeth = [_extract_tooth(p) for p in paths]
    idx_first = np.array([i for i, t in enumerate(teeth) if t in {"14", "24"}], dtype=int)
    idx_second = np.array([i for i, t in enumerate(teeth) if t in {"15", "25"}], dtype=int)
    
    # Find all folds (assume all models have same fold structure)
    first_run_fold_dirs = sorted([d for d in run_dirs[0].glob("fold_*") if d.is_dir()])
    if not first_run_fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {run_dirs[0]}")
    
    num_folds = len(first_run_fold_dirs)
    print(f"Found {num_folds} folds")
    
    # Evaluate each fold separately
    per_fold_metrics: List[Dict] = []
    all_ens_probs: List[np.ndarray] = []
    
    for fold_idx in range(num_folds):
        fold_name = f"fold_{fold_idx:02d}"
        
        # Collect predictions from all models for this fold
        fold_model_probs: List[np.ndarray] = []
        
        for model_name, run_dir in zip(model_names, run_dirs):
            fold_dir = run_dir / fold_name
            best_ckpt = fold_dir / "best.pt"
            
            if not best_ckpt.exists():
                print(f"[SKIP] {fold_name} for {model_name}: missing best.pt")
                continue
            
            # Load model
            model = build_model(model_name, num_classes=2, pretrained=False).to(device)
            state = torch.load(best_ckpt, map_location=device)
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Disable in-place activations
            for m in model.modules():
                if isinstance(m, (nn.ReLU, nn.SiLU, nn.LeakyReLU)) and getattr(m, "inplace", False):
                    m.inplace = False
            
            # Predict
            probs_list = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                    probs_list.extend(probs.tolist())
            
            fold_model_probs.append(np.array(probs_list, dtype=np.float64))
        
        if not fold_model_probs:
            print(f"[SKIP] {fold_name}: no valid model predictions")
            continue
        
        # Average across models for this fold
        P_fold = np.stack(fold_model_probs, axis=1)  # [N, num_models]
        ens_probs_fold = P_fold.mean(axis=1)  # [N]
        ens_preds_fold = (ens_probs_fold >= 0.5).astype(int)
        all_ens_probs.append(ens_probs_fold)
        
        # Compute metrics for this fold
        fold_metrics = compute_confusion_metrics(y_true, ens_preds_fold, ens_probs_fold)
        
        # Add DeLong stats
        delong_stats = _compute_delong_auc_stats(y_true, ens_probs_fold)
        if delong_stats:
            fold_metrics.update(delong_stats)
        
        # Per-premolar metrics
        def sub(arr, idx):
            return arr[idx] if len(idx) > 0 else np.array([], dtype=arr.dtype)
        
        m_first = compute_confusion_metrics(sub(y_true, idx_first), sub(ens_preds_fold, idx_first), sub(ens_probs_fold, idx_first))
        m_second = compute_confusion_metrics(sub(y_true, idx_second), sub(ens_preds_fold, idx_second), sub(ens_probs_fold, idx_second))
        
        fold_metrics["by_premolar"] = {
            "first_14_24": m_first,
            "second_15_25": m_second,
        }
        
        per_fold_metrics.append(fold_metrics)
        print(f"[{fold_name}] acc={fold_metrics.get('accuracy', float('nan')):.4f} auc={fold_metrics.get('auc', float('nan')):.4f}")
    
    if not per_fold_metrics:
        raise RuntimeError("No valid fold predictions found")
    
    # Aggregate across folds (mean ± SD)
    def mean_sd(ms: List[Dict], key: str) -> tuple:
        vals = [float(m.get(key, float('nan'))) for m in ms if key in m]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))
    
    base_keys = ["accuracy", "f1", "sensitivity", "specificity", "ppv", "npv", "auc"]
    delong_keys = ["auc_delong", "auc_delong_std", "auc_delong_var", "auc_delong_ci_low", "auc_delong_ci_high"]
    all_keys = base_keys + delong_keys
    
    summary = {}
    for k in all_keys:
        mu, sd = mean_sd(per_fold_metrics, k)
        summary[k] = {"mean": mu, "sd": sd}
    
    # Per-premolar aggregation
    def premolar_mean_sd(subkey: str, metric: str) -> tuple:
        vals = []
        for m in per_fold_metrics:
            bp = m.get("by_premolar", {})
            sk = bp.get(subkey, {})
            if metric in sk and np.isfinite(float(sk[metric])):
                vals.append(float(sk[metric]))
        if not vals:
            return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))
    
    summary["by_premolar_avg"] = {
        "first_14_24": {k: {"mean": premolar_mean_sd("first_14_24", k)[0], "sd": premolar_mean_sd("first_14_24", k)[1]} for k in base_keys},
        "second_15_25": {k: {"mean": premolar_mean_sd("second_15_25", k)[0], "sd": premolar_mean_sd("second_15_25", k)[1]} for k in base_keys},
    }
    
    # Also compute averaged predictions for overall plots and DeLong comparisons
    # For ensemble with multiple folds, we average probabilities across folds to get
    # one prediction per sample. This is required for DeLong's test, which compares
    # per-sample predictions between two models (both must have shape [N]).
    P = np.stack(all_ens_probs, axis=1)  # [N, num_folds]
    avg_probs = P.mean(axis=1)  # [N] - one probability per sample (averaged across folds)
    avg_preds = (avg_probs >= 0.5).astype(int)
    
    return {
        "per_fold": per_fold_metrics,
        "summary": summary,
        "y_true": y_true,
        "y_pred": avg_preds,
        "y_prob": avg_probs,
        "paths": paths,
        "n_folds": len(per_fold_metrics),
        "n_models": len(model_names),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on held-out test set")
    parser.add_argument("--run-dir", help="Path to model run directory (e.g., outputs/alexnet_deneme_8) or outputs/ for ensemble")
    parser.add_argument("--data-dir", required=True, help="Path to held-out test ImageFolder (e.g., data/held_out_test/crops)")
    parser.add_argument("--model", required=True, help="Model name (e.g., alexnet) or 'ensemble' or 'expert'")
    parser.add_argument("--models", nargs="+", help="Model names for ensemble (e.g., alexnet efficientnet_b0 densenet121)")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expert-csv", default="data/held_out_test/expert_pred.csv", help="CSV with expert predictions")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir) if args.run_dir else None
    data_dir = Path(args.data_dir)
    out_dir = Path(args.outdir)
    device = torch.device(args.device)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    model_lower = args.model.lower()

    if model_lower == "expert":
        csv_path = Path(args.expert_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Expert CSV not found: {csv_path}")
        results = evaluate_expert(
            csv_path,
            data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
    elif model_lower == "ensemble":
        if not args.models:
            raise ValueError("--models required when --model=ensemble")
        # Build run directories for each model
        run_dirs = []
        for model_name in args.models:
            # Try common patterns
            patterns = [
                f"outputs/{model_name}_deneme_8",
                f"outputs/{model_name}_deneme_5",
                f"outputs/{model_name}_kfold_42",
            ]
            found = None
            for pattern in patterns:
                p = Path(pattern)
                if p.exists():
                    found = p
                    break
            if not found:
                raise FileNotFoundError(f"Could not find run directory for {model_name}. Tried: {patterns}")
            run_dirs.append(found)
        
        results = evaluate_ensemble(
            run_dirs, args.models, data_dir, device,
            batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size
        )
    else:
        if run_dir is None or not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        results = evaluate_single_model(
            run_dir, data_dir, args.model, device,
            batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size
        )
    
    # Save results
    if "summary" in results:
        # Single model: has per_fold and summary
        output = {
            "per_fold": results["per_fold"],
            "summary": results["summary"],
        }
        (out_dir / "metrics_held_out.json").write_text(json.dumps(output, indent=2))
        
        # Print summary
        print(f"\n=== Held-Out Test Results (Mean ± SD across {results['n_folds']} folds) ===")
        summary = results["summary"]
        print(f"Accuracy: {summary.get('accuracy', {}).get('mean', float('nan')):.4f} ± {summary.get('accuracy', {}).get('sd', float('nan')):.4f}")
        print(f"F1: {summary.get('f1', {}).get('mean', float('nan')):.4f} ± {summary.get('f1', {}).get('sd', float('nan')):.4f}")
        print(f"AUC: {summary.get('auc', {}).get('mean', float('nan')):.4f} ± {summary.get('auc', {}).get('sd', float('nan')):.4f}")
        if "auc_delong" in summary:
            auc_mean = summary["auc_delong"]["mean"]
            auc_sd = summary["auc_delong"]["sd"]
            print(f"DeLong AUC: {auc_mean:.4f} ± {auc_sd:.4f}")
        print(f"Sensitivity: {summary.get('sensitivity', {}).get('mean', float('nan')):.4f} ± {summary.get('sensitivity', {}).get('sd', float('nan')):.4f}")
        print(f"Specificity: {summary.get('specificity', {}).get('mean', float('nan')):.4f} ± {summary.get('specificity', {}).get('sd', float('nan')):.4f}")
    else:
        # Ensemble: simplified for now (can be enhanced later)
        metrics = results["metrics"]
        (out_dir / "metrics_held_out.json").write_text(json.dumps(metrics, indent=2))
        
        print(f"\n=== Held-Out Test Results ===")
        print(f"Accuracy: {metrics.get('accuracy', float('nan')):.4f}")
        print(f"F1: {metrics.get('f1', float('nan')):.4f}")
        print(f"AUC: {metrics.get('auc', float('nan')):.4f}")
        if "auc_delong" in metrics:
            print(f"DeLong AUC: {metrics['auc_delong']:.4f} (95% CI: {metrics['auc_delong_ci_low']:.4f} - {metrics['auc_delong_ci_high']:.4f})")
        print(f"Sensitivity: {metrics.get('sensitivity', float('nan')):.4f}")
        print(f"Specificity: {metrics.get('specificity', float('nan')):.4f}")
    
    # Save arrays (averaged predictions for plots)
    np.save(out_dir / "y_true_held_out.npy", results["y_true"])
    np.save(out_dir / "y_pred_held_out.npy", results["y_pred"])
    np.save(out_dir / "y_prob_held_out.npy", results["y_prob"])
    np.save(out_dir / "paths_held_out.npy", np.array([str(p) for p in results["paths"]], dtype=object))
    
    # Save plots (using averaged predictions)
    model_name = args.model if args.model.lower() != "ensemble" else f"Ensemble ({', '.join(args.models)})"
    _save_high_quality_plots(results["y_true"], results["y_pred"], results["y_prob"], out_dir, title=f"Held-Out Test: {model_name}")
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()

