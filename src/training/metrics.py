# File: src/training/metrics.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        roc_curve, auc, roc_auc_score,
    )
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

_Z_95 = 1.959963984540054  # 97.5th quantile of the standard normal (two-sided 95% CI)


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Return 1-based midranks for DeLong algorithm."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    sort_idx = np.argsort(x)
    sorted_x = x[sort_idx]
    midranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1) + 1.0
        midranks[i:j] = midrank
        i = j
    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DeLong AUC variance for multiple classifiers.

    predictions_sorted_transposed: array of shape (n_classifiers, n_examples) sorted by score descending.
    label_1_count: number of positive samples
    """
    if label_1_count <= 0:
        raise ValueError("Need at least one positive sample for DeLong variance.")
    n_classifiers, n_examples = predictions_sorted_transposed.shape
    n_neg = n_examples - label_1_count
    if n_neg <= 0:
        raise ValueError("Need at least one negative sample for DeLong variance.")

    positive_preds = predictions_sorted_transposed[:, :label_1_count]
    negative_preds = predictions_sorted_transposed[:, label_1_count:]

    tx = np.apply_along_axis(_compute_midrank, 1, positive_preds)
    ty = np.apply_along_axis(_compute_midrank, 1, negative_preds)
    tz = np.apply_along_axis(_compute_midrank, 1, predictions_sorted_transposed)

    aucs = (tz[:, :label_1_count].sum(axis=1) / label_1_count - (label_1_count + 1) / 2.0) / n_neg

    v01 = (tz[:, :label_1_count] - tx) / n_neg
    v10 = 1.0 - (tz[:, label_1_count:] - ty) / label_1_count

    sx = np.atleast_2d(np.cov(v01, bias=True))
    sy = np.atleast_2d(np.cov(v10, bias=True))
    delong_cov = sx / label_1_count + sy / n_neg
    return aucs, delong_cov


def _compute_delong_auc_stats(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[Dict[str, float]]:
    """Return AUC, variance, standard error, and 95% CI using DeLong's method."""
    if y_true.size == 0 or y_prob.size == 0:
        return None
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    mask = np.isfinite(y_prob)
    if not np.all(mask):
        y_true = y_true[mask]
        y_prob = y_prob[mask]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return None

    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None
    # Ensure positives are first as expected by the DeLong routine
    preds_vector = np.concatenate([pos_scores, neg_scores], axis=0)
    preds_mat = preds_vector.reshape(1, -1)
    try:
        aucs, cov = _fast_delong(preds_mat, int(pos))
    except ValueError:
        return None
    auc_val = float(aucs[0])
    var = float(np.maximum(cov[0, 0], 0.0))
    std = float(np.sqrt(var))
    ci_low = float(max(0.0, auc_val - _Z_95 * std))
    ci_high = float(min(1.0, auc_val + _Z_95 * std))
    return {
        "auc_delong": auc_val,
        "auc_delong_var": var,
        "auc_delong_std": std,
        "auc_delong_ci_low": ci_low,
        "auc_delong_ci_high": ci_high,
    }


def _ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    if y_true.size == 0:
        return {
            "accuracy": float("nan"), "f1": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "ppv": float("nan"), "npv": float("nan"), "auc": float("nan"),
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }
    if _HAVE_SK:
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    else:
        # Minimal fallbacks
        acc = float((y_true == y_pred).sum() / max(len(y_true), 1))
        f1 = float("nan")
        cm = np.array([[0, 0], [0, 0]], dtype=int)
        for t, p in zip(y_true.tolist(), y_pred.tolist()):
            if t == 0 and p == 0: cm[0, 0] += 1
            elif t == 0 and p == 1: cm[0, 1] += 1
            elif t == 1 and p == 0: cm[1, 0] += 1
            else: cm[1, 1] += 1
    tn, fp, fn, tp = (cm.ravel().tolist() if cm.size == 4 else (0, 0, 0, 0))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    if _HAVE_SK and y_prob is not None and len(y_prob) == len(y_true):
        try:
            auc_v = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc_v = float("nan")
    else:
        auc_v = float("nan")
    delong = _compute_delong_auc_stats(y_true, y_prob) if y_prob is not None else None
    out = {
        "accuracy": float(acc),
        "f1": float(f1),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "auc": float(auc_v),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if delong is not None:
        out.update(delong)
    return out


def _get_sample_paths_from_loader(loader: DataLoader) -> List[Path]:
    ds = loader.dataset
    try:
        # Subset(ImageFolder) case
        if hasattr(ds, "indices") and hasattr(ds, "dataset") and hasattr(ds.dataset, "samples"):
            base = ds.dataset
            return [Path(base.samples[i][0]) for i in ds.indices]
        # ImageFolder case
        if hasattr(ds, "samples"):
            return [Path(p) for p, _ in ds.samples]
    except Exception:
        pass
    return []


def _extract_tooth_from_path(p: Path) -> Optional[str]:
    # Look for _(14|15|24|25) in the stem (handles *_aug..., *_rep... etc.)
    s = p.stem
    m = re.search(r"_(14|15|24|25)(?:_|$)", s)
    if m:
        return m.group(1)
    # Fallback: check full name
    m2 = re.search(r"_(14|15|24|25)(?:_|\.|$)", p.name)
    return m2.group(1) if m2 else None


@torch.no_grad()
def evaluate_and_store(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    prefix: str,
    class_names: Optional[List[str]] = None,
    save_logits: bool = False,
) -> Dict[str, float]:
    model.eval()
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    all_probs: List[float] = []
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Overall metrics
    m = compute_confusion_metrics(y_true, y_pred, y_prob)

    # Per-premolar metrics (first: 14,24) (second: 15,25)
    paths: List[Path] = _get_sample_paths_from_loader(loader)
    teeth_notes: List[Optional[str]] = [_extract_tooth_from_path(p) for p in paths]
    idx_first = np.array([i for i, t in enumerate(teeth_notes) if t in {"14", "24"}], dtype=int)
    idx_second = np.array([i for i, t in enumerate(teeth_notes) if t in {"15", "25"}], dtype=int)

    def sub(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return arr[idx] if len(idx) > 0 else np.array([], dtype=arr.dtype)

    m_first = compute_confusion_metrics(sub(y_true, idx_first), sub(y_pred, idx_first), sub(y_prob, idx_first))
    m_second = compute_confusion_metrics(sub(y_true, idx_second), sub(y_pred, idx_second), sub(y_prob, idx_second))

    # Counts per tooth number (14/15/24/25) for reference
    tooth_counts: Dict[str, int] = {"14": 0, "15": 0, "24": 0, "25": 0}
    for t in teeth_notes:
        if t in tooth_counts:
            tooth_counts[t] += 1

    m_out = {
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

    (out_dir / f"metrics_{prefix}.json").write_text(json.dumps(m_out, indent=2))
    save_confusion_matrix_plot(y_true, y_pred, out_dir / f"confusion_{prefix}.png", class_names or ["0", "1"])
    save_roc_curve_plot(y_true, y_prob, out_dir / f"roc_{prefix}.png")

    if save_logits:
        np.save(out_dir / f"y_true_{prefix}.npy", y_true)
        np.save(out_dir / f"y_pred_{prefix}.npy", y_pred)
        np.save(out_dir / f"y_prob_{prefix}.npy", y_prob)
        # Save paths for alignment between models
        if paths:
            np.save(out_dir / f"paths_{prefix}.npy", np.array([str(p) for p in paths], dtype=object))

    return m_out


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path, class_names: List[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) if _HAVE_SK else np.array([[0, 0], [0, 0]])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    for i in range(min(2, cm.shape[0])):
        for j in range(min(2, cm.shape[1])):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    path = Path(path); _ensure_dir(path.parent); plt.savefig(path, dpi=180); plt.close()


def save_roc_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    path = Path(path); _ensure_dir(path.parent)
    try:
        if not _HAVE_SK:
            raise RuntimeError("sklearn not available")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(loc='lower right'); plt.tight_layout()
        plt.savefig(path, dpi=180); plt.close()
    except Exception:
        # fallback empty plot
        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (unavailable)'); plt.tight_layout()
        plt.savefig(path, dpi=180); plt.close()


def save_loss_curves(train_losses: List[float], val_losses: List[float], path: Path) -> None:
    path = Path(path); _ensure_dir(path.parent)
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='train_loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Curves'); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=180); plt.close()
    (path.parent / "loss_history.json").write_text(json.dumps({
        "train_loss": [float(x) for x in train_losses],
        "val_loss": [float(x) for x in val_losses],
    }, indent=2))


def _find_last_conv_layer(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    last = None
    for child in module.children():
        cand = _find_last_conv_layer(child)
        if isinstance(child, torch.nn.Conv2d):
            last = child
        if cand is not None:
            last = cand
    return last


@torch.no_grad()
def save_gradcam_overlays(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    out_dir: Path,
    class_names: List[str],
    max_images: int = 200,
    image_size: int = 224,
) -> None:
    model.eval()
    tl = _find_last_conv_layer(model)
    if tl is None:
        return
    # Disable in-place ReLUs
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU) and getattr(m, 'inplace', False):
            m.inplace = False

    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)
        output.register_hook(lambda g: gradients.append(g.detach().clone()))

    handle_f = tl.register_forward_hook(fwd_hook)
    out_dir = Path(out_dir); _ensure_dir(out_dir)

    from PIL import Image
    processed = 0
    for idx in range(len(dataset)):
        if processed >= int(max_images):
            break
        try:
            img_t, label = dataset[idx]
        except Exception:
            continue
        img_t = img_t.unsqueeze(0).to(device)
        label_int = int(label)
        activations.clear(); gradients.clear()
        img_t.requires_grad_(True)
        logits = model(img_t)
        pred_idx = int(logits.argmax(dim=1).item())
        score = logits[0, pred_idx]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        if not activations or not gradients:
            continue
        act = activations[-1]
        grad = gradients[-1]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).squeeze(0)
        cam = cam / (cam.max().clamp_min(1e-8))
        cam_np = cam.detach().cpu().numpy()

        # Try to get original path
        p = None
        try:
            from torch.utils.data import Subset as _Subset
            if isinstance(dataset, _Subset):
                base = dataset.dataset
                abs_idx = int(dataset.indices[idx])
                if hasattr(base, 'samples') and len(base.samples) > 0:
                    p = Path(base.samples[abs_idx][0])
            elif hasattr(dataset, 'samples') and len(dataset.samples) > 0:
                p = Path(dataset.samples[idx][0])
        except Exception:
            p = None

        # Save overlay/original
        if p is not None and p.exists():
            with Image.open(p).convert('RGB') as im:
                im_resized = im.resize((image_size, image_size))
                fig = plt.figure(figsize=(4, 4))
                plt.imshow(im_resized)
                plt.imshow(cam_np, cmap='jet', alpha=0.35)
                plt.axis('off')
                pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
                true_name = class_names[label_int] if 0 <= label_int < len(class_names) else str(label_int)
                plt.title(f"pred={pred_name} true={true_name}")
                stem = p.stem
                fig.savefig(out_dir / f"{stem}_overlay.png", dpi=180, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                im_resized.save(out_dir / f"{stem}_original.png")
        processed += 1
    handle_f.remove()