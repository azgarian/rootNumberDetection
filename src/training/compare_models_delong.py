#!/usr/bin/env python3
"""
Paired DeLong test for comparing two models on the same test subjects.

Loads per-sample predictions from saved .npy files, aligns by sample path,
concatenates across all folds, and runs a paired DeLong test.

Usage:
    python src/training/compare_models_delong.py \
        --model1-dir results/inference/ensemble_model \
        --model2-dir results/inference/efficientnet_b0 \
        --outfile results/inference/ensemble_vs_efficientnet_delong.json \
        --model1-name "Ensemble" \
        --model2-name "EfficientNet-B0"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Import DeLong functions from metrics
import sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from metrics import _fast_delong, _compute_delong_auc_stats

_Z_95 = 1.959963984540054  # 97.5th quantile of the standard normal (two-sided 95% CI)


def _patient_id_from_path(p: Path) -> str:
    s = p.stem
    return s.split("_", 1)[0] if "_" in s else s


def _load_fold_predictions(fold_dir: Path, prefix: str = "test", fallback_dir: Optional[Path] = None) -> Optional[Tuple[np.ndarray, np.ndarray, List[Path]]]:
    """Load y_true, y_prob, and paths from a fold directory.
    
    If files not found, tries fallback_dir (e.g., original outputs directory).
    
    Returns (y_true, y_prob, paths) or None if files missing.
    """
    y_true_path = fold_dir / f"y_true_{prefix}.npy"
    y_prob_path = fold_dir / f"y_prob_{prefix}.npy"
    paths_path = fold_dir / f"paths_{prefix}.npy"
    
    # Try fallback if files don't exist
    if (not y_true_path.exists() or not y_prob_path.exists()) and fallback_dir is not None:
        fallback_fold = fallback_dir / fold_dir.name
        if fallback_fold.exists():
            y_true_path = fallback_fold / f"y_true_{prefix}.npy"
            y_prob_path = fallback_fold / f"y_prob_{prefix}.npy"
            paths_path = fallback_fold / f"paths_{prefix}.npy"
    
    if not y_true_path.exists() or not y_prob_path.exists():
        return None
    
    y_true = np.load(y_true_path)
    y_prob = np.load(y_prob_path)
    
    # Try to load paths, fallback to empty list
    if paths_path.exists():
        paths_array = np.load(paths_path, allow_pickle=True)
        paths = [Path(p) for p in paths_array]
    else:
        # If paths not saved, we can't align - return None for paths
        # This will require alternative alignment strategy
        paths = []
    
    return y_true, y_prob, paths


def _find_prefix_for_fold(fold_dir: Path) -> Optional[str]:
    """Try to find the correct prefix for a fold directory.
    Checks for 'test' and 'test_ensemble' patterns.
    """
    for prefix in ["test_ensemble", "test"]:
        y_true_path = fold_dir / f"y_true_{prefix}.npy"
        y_prob_path = fold_dir / f"y_prob_{prefix}.npy"
        if y_true_path.exists() and y_prob_path.exists():
            return prefix
    return None


def _align_predictions_by_path(
    y_true1: np.ndarray, y_prob1: np.ndarray, paths1: List[Path],
    y_true2: np.ndarray, y_prob2: np.ndarray, paths2: List[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align predictions from two models by matching file paths.
    
    Returns (y_true_aligned, y_prob1_aligned, y_prob2_aligned) for matched samples.
    """
    if len(paths1) == 0 or len(paths2) == 0:
        raise ValueError("Cannot align: paths not saved. Re-run inference/ensemble with path saving enabled.")
    
    if len(paths1) != len(y_true1) or len(paths1) != len(y_prob1):
        raise ValueError(f"Mismatch: paths1={len(paths1)}, y_true1={len(y_true1)}, y_prob1={len(y_prob1)}")
    if len(paths2) != len(y_true2) or len(paths2) != len(y_prob2):
        raise ValueError(f"Mismatch: paths2={len(paths2)}, y_true2={len(y_true2)}, y_prob2={len(y_prob2)}")
    
    # Build index from model2 paths
    path_to_idx2 = {Path(p).resolve(): i for i, p in enumerate(paths2)}
    
    # Find matches
    matched_indices1 = []
    matched_indices2 = []
    for i, p1 in enumerate(paths1):
        p1_resolved = Path(p1).resolve()
        if p1_resolved in path_to_idx2:
            matched_indices1.append(i)
            matched_indices2.append(path_to_idx2[p1_resolved])
    
    if len(matched_indices1) == 0:
        raise ValueError("No matching samples found between models. Check that they use the same test sets.")
    
    # Extract aligned arrays
    y_true_aligned = y_true1[matched_indices1]
    y_prob1_aligned = y_prob1[matched_indices1]
    y_prob2_aligned = y_prob2[matched_indices2]
    
    # Verify labels match (should be identical for same samples)
    if not np.array_equal(y_true_aligned, y_true2[matched_indices2]):
        mismatches = np.sum(y_true_aligned != y_true2[matched_indices2])
        print(f"[WARNING] {mismatches} label mismatches between models (using model1 labels)")
    
    return y_true_aligned, y_prob1_aligned, y_prob2_aligned


def _paired_delong_test(
    y_true: np.ndarray, y_prob1: np.ndarray, y_prob2: np.ndarray
) -> Dict[str, float]:
    """Run paired DeLong test comparing two models.
    
    Returns dict with AUCs, difference, variance, SE, CI, and p-value.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob1 = np.asarray(y_prob1, dtype=np.float64).ravel()
    y_prob2 = np.asarray(y_prob2, dtype=np.float64).ravel()
    
    if len(y_true) != len(y_prob1) or len(y_true) != len(y_prob2):
        raise ValueError("All arrays must have the same length")
    
    # Filter finite values
    mask = np.isfinite(y_prob1) & np.isfinite(y_prob2)
    y_true = y_true[mask]
    y_prob1 = y_prob1[mask]
    y_prob2 = y_prob2[mask]
    
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        raise ValueError("Need both positive and negative samples")
    
    # Arrange predictions with positives first, negatives second (as expected by DeLong routine)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    if not np.any(pos_mask) or not np.any(neg_mask):
        raise ValueError("Need both positive and negative samples after alignment")

    preds_model1 = np.concatenate([y_prob1[pos_mask], y_prob1[neg_mask]], axis=0)
    preds_model2 = np.concatenate([y_prob2[pos_mask], y_prob2[neg_mask]], axis=0)
    preds_mat = np.stack([preds_model1, preds_model2], axis=0)
    
    try:
        aucs, delong_cov = _fast_delong(preds_mat, int(pos))
    except ValueError as e:
        raise ValueError(f"DeLong computation failed: {e}")
    
    auc1 = float(aucs[0])
    auc2 = float(aucs[1])
    auc_diff = auc1 - auc2
    
    # Covariance matrix: [[var1, cov12], [cov21, var2]]
    var1 = float(np.maximum(delong_cov[0, 0], 0.0))
    var2 = float(np.maximum(delong_cov[1, 1], 0.0))
    cov12 = float(delong_cov[0, 1])
    
    # Variance of difference: var(AUC1 - AUC2) = var1 + var2 - 2*cov12
    var_diff = var1 + var2 - 2.0 * cov12
    se_diff = float(np.sqrt(np.maximum(var_diff, 0.0)))
    
    # 95% CI for difference
    ci_low = float(auc_diff - _Z_95 * se_diff)
    ci_high = float(auc_diff + _Z_95 * se_diff)
    
    # Two-sided z-test p-value
    if se_diff > 0:
        z_stat = auc_diff / se_diff
        try:
            from scipy.stats import norm
            p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        except ImportError:
            # Fallback: approximate using standard normal CDF approximation
            import math
            z_abs = abs(z_stat)
            # Approximation: 1 - norm.cdf(z) ≈ 0.5 * erfc(z/sqrt(2))
            p_value = float(math.erfc(z_abs / math.sqrt(2.0)))
    else:
        z_stat = float("nan")
        p_value = float("nan")
    
    # Individual 95% CIs
    se1 = float(np.sqrt(var1))
    se2 = float(np.sqrt(var2))
    ci1_low = float(max(0.0, auc1 - _Z_95 * se1))
    ci1_high = float(min(1.0, auc1 + _Z_95 * se1))
    ci2_low = float(max(0.0, auc2 - _Z_95 * se2))
    ci2_high = float(min(1.0, auc2 + _Z_95 * se2))
    
    return {
        "model1_auc": auc1,
        "model1_se": se1,
        "model1_ci_low": ci1_low,
        "model1_ci_high": ci1_high,
        "model2_auc": auc2,
        "model2_se": se2,
        "model2_ci_low": ci2_low,
        "model2_ci_high": ci2_high,
        "auc_difference": auc_diff,
        "se_difference": se_diff,
        "ci_difference_low": ci_low,
        "ci_difference_high": ci_high,
        "z_statistic": z_stat,
        "p_value": p_value,
        "n_samples": int(len(y_true)),
        "n_positive": int(pos),
        "n_negative": int(neg),
    }


def main():
    parser = argparse.ArgumentParser(description="Paired DeLong test comparing two models")
    parser.add_argument("--model1-dir", required=True, help="Directory with fold_*/y_*_test.npy files (e.g., delong_8/ensemble_model)")
    parser.add_argument("--model2-dir", required=True, help="Directory with fold_*/y_*_test.npy files (e.g., delong_8/efficientnet_b0)")
    parser.add_argument("--outfile", required=True, help="Output JSON file path")
    parser.add_argument("--model1-name", default="Model1", help="Name for model1 in output")
    parser.add_argument("--model2-name", default="Model2", help="Name for model2 in output")
    parser.add_argument("--prefix", default="test", help="Prefix for .npy files (default: 'test' auto-detects 'test' or 'test_ensemble')")
    parser.add_argument("--model1-fallback", help="Fallback directory to check if files not found (e.g., outputs/efficientnet_b0_deneme_8)")
    parser.add_argument("--model2-fallback", help="Fallback directory to check if files not found (e.g., outputs/efficientnet_b0_deneme_8)")
    args = parser.parse_args()
    
    model1_dir = Path(args.model1_dir)
    model2_dir = Path(args.model2_dir)
    outfile = Path(args.outfile)
    
    if not model1_dir.exists():
        raise FileNotFoundError(f"Model1 directory not found: {model1_dir}")
    if not model2_dir.exists():
        raise FileNotFoundError(f"Model2 directory not found: {model2_dir}")
    
    # Set up fallback directories
    model1_fallback = Path(args.model1_fallback) if args.model1_fallback else None
    model2_fallback = Path(args.model2_fallback) if args.model2_fallback else None
    
    if model1_fallback and not model1_fallback.exists():
        print(f"[WARNING] Model1 fallback directory not found: {model1_fallback}")
        model1_fallback = None
    if model2_fallback and not model2_fallback.exists():
        print(f"[WARNING] Model2 fallback directory not found: {model2_fallback}")
        model2_fallback = None
    
    # Find all folds
    folds1 = sorted([d for d in model1_dir.glob("fold_*") if d.is_dir()])
    folds2 = sorted([d for d in model2_dir.glob("fold_*") if d.is_dir()])
    
    if len(folds1) == 0:
        raise ValueError(f"No fold_* directories found in {model1_dir}")
    if len(folds2) == 0:
        raise ValueError(f"No fold_* directories found in {model2_dir}")
    
    if len(folds1) != len(folds2):
        print(f"[WARNING] Different number of folds: model1={len(folds1)}, model2={len(folds2)}")
    
    # Load and concatenate predictions from all folds
    all_y_true1 = []
    all_y_prob1 = []
    all_paths1 = []
    all_y_true2 = []
    all_y_prob2 = []
    all_paths2 = []
    
    # Auto-detect prefixes if default "test" was used
    use_auto_detect = (args.prefix == "test")
    prefix1 = args.prefix
    prefix2 = args.prefix
    
    if use_auto_detect and len(folds1) > 0:
        # Auto-detect on first fold
        detected1 = _find_prefix_for_fold(folds1[0])
        detected2 = _find_prefix_for_fold(folds2[0]) if len(folds2) > 0 else None
        if detected1:
            prefix1 = detected1
            print(f"[INFO] Auto-detected prefix for model1: {prefix1}")
        if detected2:
            prefix2 = detected2
            print(f"[INFO] Auto-detected prefix for model2: {prefix2}")
    
    for fold1, fold2 in zip(folds1, folds2):
        pred1 = _load_fold_predictions(fold1, prefix=prefix1, fallback_dir=model1_fallback)
        pred2 = _load_fold_predictions(fold2, prefix=prefix2, fallback_dir=model2_fallback)
        
        if pred1 is None:
            print(f"[SKIP] {fold1.name}: missing prediction files")
            continue
        if pred2 is None:
            print(f"[SKIP] {fold2.name}: missing prediction files")
            continue
        
        y_true1, y_prob1, paths1 = pred1
        y_true2, y_prob2, paths2 = pred2
        
        all_y_true1.append(y_true1)
        all_y_prob1.append(y_prob1)
        all_paths1.extend(paths1)
        all_y_true2.append(y_true2)
        all_y_prob2.append(y_prob2)
        all_paths2.extend(paths2)
    
    if len(all_y_true1) == 0:
        raise ValueError("No valid fold predictions found")
    
    # Concatenate
    y_true1_cat = np.concatenate(all_y_true1)
    y_prob1_cat = np.concatenate(all_y_prob1)
    y_true2_cat = np.concatenate(all_y_true2)
    y_prob2_cat = np.concatenate(all_y_prob2)
    
    print(f"Loaded {len(y_true1_cat)} samples from model1, {len(y_true2_cat)} from model2")
    
    # Align by paths
    try:
        y_true_aligned, y_prob1_aligned, y_prob2_aligned = _align_predictions_by_path(
            y_true1_cat, y_prob1_cat, all_paths1,
            y_true2_cat, y_prob2_cat, all_paths2,
        )
        print(f"Aligned {len(y_true_aligned)} matched samples")
    except ValueError as e:
        print(f"[ERROR] Alignment failed: {e}")
        raise
    
    # Run paired DeLong test
    try:
        results = _paired_delong_test(y_true_aligned, y_prob1_aligned, y_prob2_aligned)
    except Exception as e:
        print(f"[ERROR] DeLong test failed: {e}")
        raise
    
    # Format output
    output = {
        "model1_name": args.model1_name,
        "model2_name": args.model2_name,
        "model1_dir": str(model1_dir),
        "model2_dir": str(model2_dir),
        "n_folds": len(folds1),
        "results": results,
    }
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(output, indent=2))
    
    print(f"\n=== Paired DeLong Test Results ===")
    print(f"{args.model1_name} AUC: {results['model1_auc']:.4f} (95% CI: {results['model1_ci_low']:.4f} - {results['model1_ci_high']:.4f})")
    print(f"{args.model2_name} AUC: {results['model2_auc']:.4f} (95% CI: {results['model2_ci_low']:.4f} - {results['model2_ci_high']:.4f})")
    print(f"Difference: {results['auc_difference']:.4f} (95% CI: {results['ci_difference_low']:.4f} - {results['ci_difference_high']:.4f})")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"\nResults saved to: {outfile}")


if __name__ == "__main__":
    main()

