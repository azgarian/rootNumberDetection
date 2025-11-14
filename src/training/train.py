# File: src/training/train_kfold_fixed.py
# 5-fold CV (patient-grouped). Each fold:
# - Test: exactly TEST_PER_CLASS per class (whole-patient, disjoint across folds)
# - Val: exactly VAL_PER_CLASS per class (from remaining pool)
# - Train: all remaining
# Best per-fold by val accuracy; final report = average test metrics (incl. premolar averages)

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.nn.utils import clip_grad_norm_

from dataloader import create_transforms, oversample_indices_per_class
from metrics import evaluate_and_store, save_loss_curves
from models import build_model

# ==========================
# CONFIG (hardcoded)
# ==========================
DATA_DIR = Path("data/crops_all")  # ImageFolder root with classes {'1_root_images','2_root_images'}


K_FOLDS = 5
TEST_PER_CLASS = 32
VAL_PER_CLASS = 32

EPOCHS = 30
PATIENCE = 8
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 20
PRETRAINED = True
OVERSAMPLE_TRAIN_PER_CLASS = 3000
# ==========================


def _patient_id_from_path(p: Path) -> str:
    s = p.stem
    return s.split("_", 1)[0] if "_" in s else s


def _labels_and_pids(ds: datasets.ImageFolder) -> Tuple[List[int], List[str]]:
    labels, pids = [], []
    for path_str, lbl in ds.samples:
        labels.append(int(lbl))
        pids.append(_patient_id_from_path(Path(path_str)))
    return labels, pids


def _build_group_stats(pids: List[str], labels: List[int]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    classes = sorted(set(int(l) for l in labels))
    pos = {c: i for i, c in enumerate(classes)}
    g2idx: Dict[str, List[int]] = {}
    g2cnt: Dict[str, List[int]] = {}
    for i, gid in enumerate(pids):
        if gid not in g2idx:
            g2idx[gid] = []
            g2cnt[gid] = [0] * len(classes)
        g2idx[gid].append(i)
        j = pos[int(labels[i])]
        g2cnt[gid][j] += 1
    return g2idx, g2cnt


def _select_groups_min_per_class(groups: List[str], g2cnt: Dict[str, List[int]], per_class: int, rng: torch.Generator) -> List[str]:
    order = torch.randperm(len(groups), generator=rng).tolist()
    pool = [groups[i] for i in order]
    cur = [0, 0]
    sel: List[str] = []
    def ok(): return all(cur[c] >= int(per_class) for c in range(len(cur)))
    while pool and not ok():
        best_gid, best_def = None, None
        for gid in pool:
            cand = [cur[c] + g2cnt[gid][c] for c in range(len(cur))]
            deficit = sum(max(0, int(per_class) - cand[c]) for c in range(len(cur)))
            if best_def is None or deficit < best_def:
                best_def, best_gid = deficit, gid
        if best_gid is None:
            break
        sel.append(best_gid)
        cur = [cur[c] + g2cnt[best_gid][c] for c in range(len(cur))]
        pool.remove(best_gid)
    return sel


def _shrink_to_exact(sel: List[str], g2cnt: Dict[str, List[int]], per_class: int) -> List[str]:
    cur = [0, 0]
    for gid in sel:
        cur = [cur[c] + g2cnt[gid][c] for c in range(len(cur))]
    def over(): return sum(max(0, cur[j] - int(per_class)) for j in range(len(cur)))
    while over() > 0:
        best_gid, best_over = None, None
        for gid in list(sel):
            cand = [cur[j] - g2cnt[gid][j] for j in range(len(cur))]
            if any(cand[j] < int(per_class) for j in range(len(cand))):
                continue
            o = sum(max(0, cand[j] - int(per_class)) for j in range(len(cand)))
            if best_over is None or o < best_over:
                best_over, best_gid = o, gid
        if best_gid is None:
            break
        sel.remove(best_gid)
        cur = [cur[c] - g2cnt[best_gid][c] for c in range(len(cur))]
    return sel


def _select_exact_groups(groups: List[str], g2cnt: Dict[str, List[int]], per_class: int, rng: torch.Generator) -> List[str]:
    sel = _select_groups_min_per_class(groups, g2cnt, int(per_class), rng)
    sel = _shrink_to_exact(sel, g2cnt, int(per_class))
    # verify exact (best-effort if not perfectly exact)
    cur = [0, 0]
    for gid in sel:
        cur = [cur[c] + g2cnt[gid][c] for c in range(len(cur))]
    if not all(cur[c] == int(per_class) for c in range(len(cur))):
        print(f"[warn] could not hit exact per-class={per_class}; got {cur} (best-effort due to group granularity)")
    return sel


def _counts_for_indices(ds, idxs):
    from collections import Counter
    lbls = [int(ds.samples[i][1]) for i in idxs]
    return dict(Counter(lbls))


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        seen += bs
    return total_loss / max(seen, 1)


@torch.no_grad()
def eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        seen += bs
    return total_loss / max(seen, 1)


def main():

    parser = argparse.ArgumentParser(description='Train classifier for premolar root morphology')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    MODEL_NAME = args.model
    SEED = args.seed
    torch.manual_seed(int(SEED))
    OUTPUT_DIR = Path(f"results/model/{MODEL_NAME}")

    torch.manual_seed(int(SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(json.dumps({
        "data_dir": str(DATA_DIR),
        "k_folds": K_FOLDS,
        "test_per_class": TEST_PER_CLASS,
        "val_per_class": VAL_PER_CLASS,
        "epochs": EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "image_size": IMAGE_SIZE, "num_workers": NUM_WORKERS,
        "seed": SEED, "pretrained": PRETRAINED,
        "oversample_train_per_class": OVERSAMPLE_TRAIN_PER_CLASS,
        "model": MODEL_NAME
    }, indent=2))

    # Base dataset
    train_tf, val_tf = create_transforms(image_size=int(IMAGE_SIZE), use_aug=True)
    base_train = datasets.ImageFolder(root=str(DATA_DIR), transform=train_tf)
    base_val = datasets.ImageFolder(root=str(DATA_DIR), transform=val_tf)
    labels, pids = _labels_and_pids(base_train)
    g2idx, g2cnt = _build_group_stats(pids, labels)
    all_groups = list(g2idx.keys())

    # Build disjoint test groups per fold with exact per-class targets
    rng = torch.Generator().manual_seed(int(SEED))
    remaining_for_tests = all_groups.copy()
    folds_test_groups: List[List[str]] = []
    for f in range(K_FOLDS):
        if not remaining_for_tests:
            # if exhausted, recycle (rare)
            remaining_for_tests = all_groups.copy()
        sel_test = _select_exact_groups(remaining_for_tests, g2cnt, int(TEST_PER_CLASS), rng)
        folds_test_groups.append(sel_test)
        # remove selected from remaining to keep test disjoint across folds
        remset = set(sel_test)
        remaining_for_tests = [g for g in remaining_for_tests if g not in remset]

    all_metrics_test: List[Dict] = []

    for fidx in range(K_FOLDS):
        fold_dir = OUTPUT_DIR / f"fold_{fidx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_groups = set(folds_test_groups[fidx])
        # For validation, select exact per-class from the remaining pool
        available_for_val = [g for g in all_groups if g not in test_groups]
        sel_val = _select_exact_groups(available_for_val, g2cnt, int(VAL_PER_CLASS), rng)
        val_groups = set(sel_val)
        train_groups = [g for g in available_for_val if g not in val_groups]

        # Indices
        test_idx: List[int] = []
        val_idx: List[int] = []
        train_idx: List[int] = []
        for gid, idxs in g2idx.items():
            if gid in test_groups:
                test_idx.extend(idxs)
            elif gid in val_groups:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)

        # Log split sizes/counts
        print(f"[Fold {fidx+1}/{K_FOLDS}] sizes | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        print(f"[Fold {fidx+1}/{K_FOLDS}] val_counts={_counts_for_indices(base_val, val_idx)} test_counts={_counts_for_indices(base_val, test_idx)}")

        # Write patient IDs
        (fold_dir / "test_patients.txt").write_text("\n".join(sorted(test_groups)) + "\n")
        (fold_dir / "val_patients.txt").write_text("\n".join(sorted(val_groups)) + "\n")

        # Oversample train to fixed per-class
        expanded = oversample_indices_per_class(base_train, train_idx, target_per_class=int(OVERSAMPLE_TRAIN_PER_CLASS), seed=SEED + fidx)
        train_loader = DataLoader(Subset(base_train, expanded), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(Subset(base_val, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(Subset(base_val, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

        # Model/optim
        model = build_model(MODEL_NAME, num_classes=2, pretrained=bool(PRETRAINED)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_acc = float("-inf")
        best_path = fold_dir / "best.pt"
        hist_tr: List[float] = []
        hist_va: List[float] = []
        epochs_no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss = eval_loss(model, val_loader, criterion, device)
            hist_tr.append(float(tr_loss)); hist_va.append(float(va_loss))
            scheduler.step(epoch)
            # quick val acc for early stop (no files)
            with torch.no_grad():
                correct = 0; total = 0
                for images, targets in val_loader:
                    images = images.to(device); targets = targets.to(device)
                    logits = model(images)
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                va_acc = correct / max(total, 1)
            print(f"[Fold {fidx+1}/{K_FOLDS}] Epoch {epoch:03d} | tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | va_acc={va_acc:.4f}")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model": MODEL_NAME, "state_dict": model.state_dict()}, best_path)
                epochs_no_improve = 0
                print(f"[Fold {fidx+1}/{K_FOLDS}]  -> Saved new best (val_acc={best_acc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(PATIENCE):
                    print(f"[Fold {fidx+1}/{K_FOLDS}] Early stopping")
                    break

        save_loss_curves(hist_tr, hist_va, fold_dir / "loss_curves.png")

        # Evaluate best on fixed test for this fold
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["state_dict"])

        # Store VAL metrics for this fold
        m_val = evaluate_and_store(model, val_loader, device, fold_dir, prefix="val", class_names=["1_root_images", "2_root_images"])

        # Evaluate and store TEST metrics
        m_test = evaluate_and_store(model, test_loader, device, fold_dir, prefix="test", class_names=["1_root_images", "2_root_images"], save_logits=True)
        all_metrics_test.append(m_test)
        print(f"[Fold {fidx+1}/{K_FOLDS}] TEST: acc={m_test.get('accuracy', float('nan')):.4f}")

    # Averages
    def avg_metrics(ms: List[Dict]) -> Dict:
        if not ms: return {}
        keys = ["accuracy","f1","sensitivity","specificity","ppv","npv","auc"]
        out = {}
        for k in keys:
            vals = [float(m.get(k, float('nan'))) for m in ms if k in m]
            vals = [v for v in vals if not (isinstance(v, float) and (torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v))))]
            out[k] = float(sum(vals) / max(len(vals), 1)) if vals else float("nan")
        return out

    def avg_premolar(ms: List[Dict]) -> Dict:
        def do(subkey: str) -> Dict:
            keys = ["accuracy","f1","sensitivity","specificity","ppv","npv","auc"]
            out = {}
            for k in keys:
                vals = []
                for m in ms:
                    bp = m.get("by_premolar", {})
                    sk = bp.get(subkey, {})
                    if k in sk: vals.append(float(sk[k]))
                vals = [v for v in vals if not (isinstance(v, float) and (torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v))))]
                out[k] = float(sum(vals) / max(len(vals), 1)) if vals else float("nan")
            return out
        return {
            "first_14_24": do("first_14_24"),
            "second_15_25": do("second_15_25"),
        }

    summary = {
        "folds_test": all_metrics_test,
        "test_avg": {
            **avg_metrics(all_metrics_test),
            "by_premolar_avg": avg_premolar(all_metrics_test),
        },
        "config": {
            "data_dir": str(DATA_DIR),
            "k_folds": K_FOLDS,
            "test_per_class": TEST_PER_CLASS,
            "val_per_class": VAL_PER_CLASS,
            "epochs": EPOCHS, "patience": PATIENCE,
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "image_size": IMAGE_SIZE, "num_workers": NUM_WORKERS,
            "seed": SEED, "pretrained": PRETRAINED,
            "oversample_train_per_class": OVERSAMPLE_TRAIN_PER_CLASS,
            "model": MODEL_NAME,
        }
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done. Wrote:", OUTPUT_DIR)


if __name__ == "__main__":
    main()