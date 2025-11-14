# File: src/training/ensemble_cross_backbone_test.py
# Cross-backbone ensemble on each fold's TEST patients (no leakage).
# For each fold:
#   - load each backbone's fold best.pt (selected by that fold's val acc)
#   - predict on that fold's TEST patients (never seen for training/selection)
#   - average probabilities across backbones
# Save per-fold metrics/plots and overall mean/SD across 5 folds.

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from dataloader import create_transforms
from models import build_model
from metrics import compute_confusion_metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ==========================
# CONFIG
# ==========================
DATA_DIR = Path("data/crops_all")  # ImageFolder with {'1_root_images','2_root_images'}
MODELS = ["alexnet", "efficientnet_b0", "densenet121"]  # backbones to ensemble
RUN_DIR_TEMPLATE = "results/model/{name}"      # each has fold_* with best.pt, test_patients.txt
name = "_".join(MODELS)
OUTPUT_DIR = Path(f"results/inference/ensemble_model")
BATCH_SIZE = 32
NUM_WORKERS = 8
IMAGE_SIZE = 224
SEED = 42
# ==========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _patient_id_from_path(p: Path) -> str:
    s = p.stem
    return s.split("_", 1)[0] if "_" in s else s

def _extract_tooth_from_path(p: Path) -> Optional[str]:
    import re
    m = re.search(r"_(14|15|24|25)(?:_|$|\.)", p.name)
    return m.group(1) if m else None

def _load_test_patients(fold_dir: Path) -> List[str]:
    f = fold_dir / "test_patients.txt"
    if not f.exists():
        raise FileNotFoundError(f"Missing test_patients.txt in {fold_dir}")
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]

def _indices_for_patients(ds: datasets.ImageFolder, pids: List[str]) -> List[int]:
    pid_set = set(pids)
    idxs: List[int] = []
    for i, (path_str, _) in enumerate(ds.samples):
        if _patient_id_from_path(Path(path_str)) in pid_set:
            idxs.append(i)
    if not idxs:
        raise RuntimeError("No samples matched the provided test patients.")
    return idxs

@torch.no_grad()
def _predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs: List[float] = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs, dtype=np.float64)

def _metrics_with_premolar(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, paths: List[Path]) -> Dict:
    m = compute_confusion_metrics(y_true, y_pred, y_prob)
    teeth = [_extract_tooth_from_path(p) for p in paths]
    idx_first = np.array([i for i, t in enumerate(teeth) if t in {"14", "24"}], dtype=int)
    idx_second = np.array([i for i, t in enumerate(teeth) if t in {"15", "25"}], dtype=int)

    def sub(a, idx): return a[idx] if len(idx) > 0 else np.array([], dtype=a.dtype)
    m_first = compute_confusion_metrics(sub(y_true, idx_first), sub(y_pred, idx_first), sub(y_prob, idx_first))
    m_second = compute_confusion_metrics(sub(y_true, idx_second), sub(y_pred, idx_second), sub(y_prob, idx_second))

    tooth_counts = {"14": 0, "15": 0, "24": 0, "25": 0}
    for t in teeth:
        if t in tooth_counts:
            tooth_counts[t] += 1

    return {
        **m,
        "by_premolar": {
            "first_14_24": m_first,
            "second_15_25": m_second,
            "counts": {
                "first_14_24": int(len(idx_first)),
                "second_15_25": int(len(idx_second)),
                "by_tooth": tooth_counts,
            },
        },
    }


def _save_high_quality_plots(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, out_dir: Path, fold_idx: int) -> None:
    """Save normalized confusion matrix and ROC in the same style/quality as inference.py.

    - Confusion matrix: normalized (row-wise), class names ['Single Root', 'Double Root']
      y-axis tick labels rotated 90°, centered, with padding; title 'Fold {n}' bold; dpi=450
    - ROC: title 'Fold {n}' bold; axes named as requested; dpi=450
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Fold {int(fold_idx) + 1}"

    # Confusion Matrix (normalized)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontweight='bold', fontsize=20)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Single Root", "Double Root"], size=16)  # class names
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Single Root", "Double Root"], size=16)  # rotated below
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
    fig.savefig(out_dir / "confusion_test_ensemble_hd.png", dpi=450, bbox_inches='tight')
    fig.savefig(out_dir / "confusion_test_ensemble_hd.svg")
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
        fig2.savefig(out_dir / "roc_test_ensemble_hd.png", dpi=450, bbox_inches='tight')
        fig2.savefig(out_dir / "roc_test_ensemble_hd.svg")
        plt.close(fig2)

def main():
    torch.manual_seed(int(SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(json.dumps({
        "data_dir": str(DATA_DIR),
        "models": MODELS,
        "runs": [RUN_DIR_TEMPLATE.format(name=m) for m in MODELS],
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
    }, indent=2))

    # Val/Test transforms (no aug)
    _, val_tfms = create_transforms(image_size=int(IMAGE_SIZE), use_aug=False)
    base = datasets.ImageFolder(root=str(DATA_DIR), transform=val_tfms)
    class_names = base.classes

    # Discover folds using the first backbone dir
    first_run = Path(RUN_DIR_TEMPLATE.format(name=MODELS[0])).resolve()
    fold_dirs = sorted([d for d in first_run.glob("fold_*") if d.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No folds found under {first_run}")

    per_fold_metrics: List[Dict] = []

    for fold_dir in fold_dirs:
        fidx = int(fold_dir.name.split("_")[-1])
        print(f"[Fold {fidx:02d}] Cross-backbone ensemble on TEST patients")

        # Test patients for this fold
        test_pids = _load_test_patients(fold_dir)
        test_indices = _indices_for_patients(base, test_pids)
        test_subset = Subset(base, test_indices)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

        # Predict with each backbone and average probs
        probs_list: List[np.ndarray] = []
        for m in MODELS:
            run_dir = Path(RUN_DIR_TEMPLATE.format(name=m)).resolve()
            ckpt = run_dir / f"fold_{fidx:02d}" / "best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint for {m}: {ckpt}")
            model = build_model(m, num_classes=2, pretrained=False).to(device)
            state = torch.load(ckpt, map_location=device)
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            for mm in model.modules():
                if isinstance(mm, (nn.ReLU, nn.SiLU, nn.LeakyReLU)) and getattr(mm, "inplace", False):
                    mm.inplace = False
            p = _predict_probs(model, test_loader, device)
            probs_list.append(p)

        P = np.stack(probs_list, axis=1)  # [N, M]
        ens_prob = P.mean(axis=1)         # equal-weight average
        ens_pred = (ens_prob >= 0.5).astype(int)

        # True labels and paths
        y_true: List[int] = []
        paths: List[Path] = []
        for idx in test_indices:
            sample_path, label = base.samples[idx]
            y_true.append(int(label))
            paths.append(Path(sample_path))
        y_true_np = np.array(y_true, dtype=np.int64)

        delong_additional: Dict[str, float] = {}
        try:
            from metrics import _compute_delong_auc_stats  # type: ignore
            delong_stats = _compute_delong_auc_stats(y_true_np, ens_prob)
            if delong_stats:
                delong_additional = delong_stats
        except Exception:
            pass

        m_fold = {
            **_metrics_with_premolar(y_true_np, ens_pred, ens_prob, paths),
            **delong_additional,
        }
        per_fold_metrics.append(m_fold)

        # Save per-fold
        out_fold = OUTPUT_DIR / f"fold_{fidx:02d}"
        _ensure_dir(out_fold)
        (out_fold / "metrics_test_ensemble.json").write_text(json.dumps(m_fold, indent=2))
        np.save(out_fold / "y_true_test_ensemble.npy", y_true_np)
        np.save(out_fold / "y_pred_test_ensemble.npy", ens_pred)
        np.save(out_fold / "y_prob_test_ensemble.npy", ens_prob)
        # Save paths for alignment between models
        np.save(out_fold / "paths_test_ensemble.npy", np.array([str(p) for p in paths], dtype=object))
        _save_high_quality_plots(y_true_np, ens_pred, ens_prob, out_fold, fold_idx=fidx)

        print(f"[Fold {fidx:02d}] test acc={m_fold.get('accuracy', float('nan')):.4f} f1={m_fold.get('f1', float('nan')):.4f}")

    # Mean/SD across folds
    def mean_sd(ms: List[Dict], key: str) -> Tuple[float, float]:
        vals = [float(m.get(key, float('nan'))) for m in ms if key in m]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals: return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    base_keys = ["accuracy","f1","sensitivity","specificity","ppv","npv","auc"]
    delong_keys = ["auc_delong","auc_delong_std","auc_delong_var","auc_delong_ci_low","auc_delong_ci_high"]
    keys = base_keys + delong_keys
    summary = {"per_fold": per_fold_metrics, "summary": {}}
    for k in keys:
        mu, sd = mean_sd(per_fold_metrics, k)
        summary["summary"][k] = {"mean": mu, "sd": sd}

    def premolar_mean_sd(subkey: str, metric: str) -> Tuple[float, float]:
        vals = []
        for m in per_fold_metrics:
            bp = m.get("by_premolar", {})
            sk = bp.get(subkey, {})
            if metric in sk and np.isfinite(float(sk[metric])):
                vals.append(float(sk[metric]))
        if not vals: return float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    summary["summary"]["by_premolar_avg"] = {
        "first_14_24": {k: {"mean": premolar_mean_sd("first_14_24", k)[0], "sd": premolar_mean_sd("first_14_24", k)[1]} for k in keys},
        "second_15_25": {k: {"mean": premolar_mean_sd("second_15_25", k)[0], "sd": premolar_mean_sd("second_15_25", k)[1]} for k in keys},
    }

    (OUTPUT_DIR / "summary_test_ensemble.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Wrote cross-backbone TEST ensemble to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()