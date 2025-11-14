import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision import datasets

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS

def _patient_id_from_stem(stem: str) -> str:
    # e.g., patient001_15 -> patient001 (split once from right)
    return stem.rsplit("_", 1)[0] if "_" in stem else stem

def create_transforms(image_size: int = 224, use_aug: bool = True) -> Tuple[Callable, Callable]:
    train_tfms = [
        T.Resize((image_size, image_size)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomRotation(degrees=5),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ] if use_aug else [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    val_tfms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(train_tfms), val_tfms

def _dataset_with_transform(root: Path, transform: Callable) -> datasets.ImageFolder:
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    return datasets.ImageFolder(root=str(root), transform=transform)

def _labels_and_pids(ds: datasets.ImageFolder) -> Tuple[List[int], List[str]]:
    labels: List[int] = []
    pids: List[str] = []
    for path_str, lbl in ds.samples:
        p = Path(path_str)
        labels.append(int(lbl))
        pids.append(_patient_id_from_stem(p.stem))
    return labels, pids

def _build_group_stats(pids: List[str], labels: List[int]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List[int]]:
    classes = sorted(set(int(l) for l in labels))
    pos = {c: i for i, c in enumerate(classes)}
    group_to_counts: Dict[str, List[int]] = {}
    group_to_indices: Dict[str, List[int]] = {}
    total = [0] * len(classes)
    for i, gid in enumerate(pids):
        if gid not in group_to_counts:
            group_to_counts[gid] = [0] * len(classes)
            group_to_indices[gid] = []
        group_to_indices[gid].append(i)
        j = pos[int(labels[i])]
        group_to_counts[gid][j] += 1
        total[j] += 1
    return group_to_indices, group_to_counts, total

def _l1(a: List[int], b: List[int]) -> int:
    return int(sum(abs(int(x) - int(y)) for x, y in zip(a, b)))

def _select_groups_for_target(available: List[str], group_to_counts: Dict[str, List[int]], target_counts: List[int], rng: random.Random) -> List[str]:
    pool = available.copy()
    rng.shuffle(pool)
    selected: List[str] = []
    cur = [0] * len(target_counts)
    while pool and sum(cur) < sum(target_counts):
        best_gid = None
        best_score = None
        for gid in pool:
            cand = [cur[c] + group_to_counts[gid][c] for c in range(len(cur))]
            score = _l1(cand, target_counts)
            if best_score is None or score < best_score:
                best_score = score
                best_gid = gid
        if best_gid is None:
            break
        selected.append(best_gid)
        cur = [cur[c] + group_to_counts[best_gid][c] for c in range(len(cur))]
        pool.remove(best_gid)
    return selected

def kfold_on_train_groups(
    train_root: Path,
    k: int = 5,
    seed: int = 42,
    image_size: int = 224,
    use_aug: bool = True,
    val_per_class: int = 32,
) -> List[Tuple[datasets.ImageFolder, Subset, Subset]]:
    train_tfms, val_tfms = create_transforms(image_size=image_size, use_aug=use_aug)
    base_train = _dataset_with_transform(train_root, transform=train_tfms)
    labels, pids = _labels_and_pids(base_train)

    group_to_indices, group_to_counts, total = _build_group_stats(pids, labels)
    remaining_groups = list(group_to_counts.keys())
    rng = random.Random(seed)

    def counts_of(groups: List[str]) -> List[int]:
        num_classes = len(total)
        c = [0] * num_classes
        for g in groups:
            gc = group_to_counts[g]
            for j in range(num_classes):
                c[j] += gc[j]
        return c

    def overshoot(cur: List[int], tgt: int) -> int:
        return sum(max(0, cur[j] - tgt) for j in range(len(cur)))

    folds_groups: List[List[str]] = []
    for i in range(k):
        if i == k - 1:
            # Last fold: best-effort exact; if impossible, take what remains
            target = [val_per_class] * len(total)
            sel = _select_groups_for_target(remaining_groups, group_to_counts, target, rng)
        else:
            target = [val_per_class] * len(total)
            sel = _select_groups_for_target(remaining_groups, group_to_counts, target, rng)

        # Shrink overshoot while keeping per-class >= target
        cur = counts_of(sel)
        while overshoot(cur, val_per_class) > 0:
            best_gid = None
            best_over = None
            for gid in sel:
                cand = [cur[j] - group_to_counts[gid][j] for j in range(len(cur))]
                if any(cand[j] < val_per_class for j in range(len(cand))):
                    continue
                o = overshoot(cand, val_per_class)
                if best_over is None or o < best_over:
                    best_over = o
                    best_gid = gid
            if best_gid is None:
                break
            sel.remove(best_gid)
            cur = counts_of(sel)

        folds_groups.append(sel)
        sel_set = set(sel)
        remaining_groups = [g for g in remaining_groups if g not in sel_set]

    # Build (train, val) subsets per fold
    folds: List[Tuple[datasets.ImageFolder, Subset, Subset]] = []
    for f in range(len(folds_groups)):
        val_set = set(folds_groups[f])
        tr_idx: List[int] = []
        va_idx: List[int] = []
        for gid, idxs in group_to_indices.items():
            (va_idx if gid in val_set else tr_idx).extend(idxs)
        ds_train = _dataset_with_transform(train_root, transform=train_tfms)
        ds_val = _dataset_with_transform(train_root, transform=val_tfms)
        folds.append((base_train, Subset(ds_train, tr_idx), Subset(ds_val, va_idx)))
    return folds

def oversample_indices_per_class(
    ds: datasets.ImageFolder,
    indices: List[int],
    target_per_class: int = 3000,
    seed: int = 42,
) -> List[int]:
    # Build class -> absolute indices mapping
    class_to_abs: Dict[int, List[int]] = {}
    for idx in indices:
        lbl = int(ds.samples[idx][1])
        class_to_abs.setdefault(lbl, []).append(idx)

    present = sorted(class_to_abs.keys())
    if not present:
        raise RuntimeError("No classes present in train subset.")
    for c in present:
        if len(class_to_abs[c]) == 0:
            raise RuntimeError(f"Class {c} has no samples in train subset; cannot oversample.")

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    expanded: List[int] = []
    for c in present:
        src = class_to_abs[c]
        choice = torch.randint(0, len(src), (int(target_per_class),), generator=gen).tolist()
        expanded.extend(src[i] for i in choice)

    if expanded:
        perm = torch.randperm(len(expanded), generator=gen).tolist()
        expanded = [expanded[i] for i in perm]
    return expanded

def build_cv_loaders(
    split_root: Path,
    k: int = 5,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    use_aug: bool = True,
    oversample_train_per_class: int = 3000,
) -> List[Dict[str, DataLoader]]:
    split_root = Path(split_root)
    train_root = split_root / "train"
    test_root = split_root / "test"
    folds = kfold_on_train_groups(train_root, k=k, seed=seed, image_size=image_size, use_aug=use_aug)

    # Fixed test loader if present
    test_loader: Optional[DataLoader] = None
    if test_root.exists():
        _, val_tfms = create_transforms(image_size=image_size, use_aug=False)
        test_ds = _dataset_with_transform(test_root, transform=val_tfms)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Build loaders per fold
    fold_loaders: List[Dict[str, DataLoader]] = []
    for fidx, (base_train, train_subset, val_subset) in enumerate(folds):
        # Oversample train to target per class using base_train index space
        expanded_indices = oversample_indices_per_class(base_train, train_subset.indices, target_per_class=int(oversample_train_per_class), seed=seed + fidx)
        train_expanded = Subset(base_train, expanded_indices)

        tr_loader = DataLoader(train_expanded, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        va_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        fold_loaders.append({
            "train": tr_loader,
            "val": va_loader,
            "test": test_loader,
        })
    return fold_loaders