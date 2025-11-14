import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

BASE_METRICS = ["accuracy", "f1", "sensitivity", "specificity", "ppv", "npv", "auc"]
PREMOLAR_KEYS = ["first_14_24", "second_15_25"]
PAIRWISE_NAMES = [
    "Ensemble vs EfficientNet",
    "Ensemble vs AlexNet",
    "Ensemble vs DenseNet",
    "EfficientNet vs AlexNet",
    "EfficientNet vs DenseNet",
    "AlexNet vs DenseNet",
]
PAIRWISE_NAMES_WITH_EXPERT = [
    "Ensemble vs EfficientNet",
    "Ensemble vs AlexNet",
    "Ensemble vs DenseNet",
    "Ensemble vs Expert",
    "EfficientNet vs AlexNet",
    "EfficientNet vs DenseNet",
    "EfficientNet vs Expert",
    "AlexNet vs DenseNet",
    "AlexNet vs Expert",
    "DenseNet vs Expert",
]


def load_per_fold_results(kind: str) -> Dict[str, List[Dict]]:
    """Return per-fold metrics for each model for the requested dataset."""
    if kind == "inference":
        base = Path("results/inference")
        return {
            "ensemble": json.loads((base / "ensemble_model/summary_test_ensemble.json").read_text())["per_fold"],
            "efficientnet": json.loads((base / "efficientnet_b0/summary_test.json").read_text())["per_fold"],
            "alexnet": json.loads((base / "alexnet/summary_test.json").read_text())["per_fold"],
            "densenet": json.loads((base / "densenet121/summary_test.json").read_text())["per_fold"],
        }
    if kind == "held_out":
        base = Path("results/held_out_results")
        mapping = {
            "ensemble": base / "ensemble/metrics_held_out.json",
            "efficientnet": base / "efficientnet_b0/metrics_held_out.json",
            "alexnet": base / "alexnet/metrics_held_out.json",
            "densenet": base / "densenet121/metrics_held_out.json",
            "expert": base / "expert/metrics_held_out.json",
        }
        out = {}
        for key, path in mapping.items():
            if not path.exists():
                if key == "expert":
                    print(f"[WARNING] Expert metrics not found at {path}, skipping expert in analysis")
                    continue
                raise FileNotFoundError(f"Missing held-out metrics for {key}: {path}")
            out[key] = json.loads(path.read_text())["per_fold"]
        return out
    raise ValueError(f"Unknown dataset kind: {kind}")


def extract_metrics(folds: List[Dict], metric: str) -> List[float]:
    return [float(fold.get(metric, float("nan"))) for fold in folds]


def extract_premolar_series(folds: List[Dict], premolar_key: str, metric: str) -> List[float]:
    values: List[float] = []
    for fold in folds:
        val = fold.get("by_premolar", {}).get(premolar_key, {}).get(metric)
        values.append(float(val) if val is not None else float("nan"))
    return values


def mean_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    if n <= 1:
        return mean, mean, mean
    se = arr.std(ddof=1) / np.sqrt(n)
    margin = stats.t.ppf(1 - alpha / 2, df=n - 1) * se
    return mean, mean - margin, mean + margin


def cohens_d(x1: List[float], x2: List[float]) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 <= 1 or n2 <= 1:
        return float("nan")
    pooled = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled


def build_long_df(results: Dict, metrics: List[str]) -> pd.DataFrame:
    pretty = {"ensemble": "Ensemble", "efficientnet": "EfficientNet", "alexnet": "AlexNet", "densenet": "DenseNet", "expert": "Expert"}
    rows = []
    for metric in metrics:
        for model_key, vals in results[metric]["values"].items():
            model_name = pretty.get(model_key, model_key)
            for v in vals:
                rows.append({"Metric": metric.upper(), "Model": model_name, "Value": float(v)})
    return pd.DataFrame(rows)


def plot_metric_boxplot(df: pd.DataFrame, metric: str, out_path: Path):
    sns.set_theme(style="whitegrid")
    data = df[df["Metric"] == metric.upper()].copy()
    if data.empty:
        return
    order = ["Ensemble", "EfficientNet", "AlexNet", "DenseNet", "Expert"]
    # Filter order to only include models present in data
    available_models = data["Model"].unique()
    order = [m for m in order if m in available_models]
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=data, x="Model", y="Value", order=order, width=0.6)
    sns.stripplot(data=data, x="Model", y="Value", order=order, color="black", alpha=0.5, size=5, jitter=0.1)
    plt.title(f"{metric.upper()} across folds ({out_path.parent.name})")
    plt.ylim(0.65, 1.0)
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_mean_ci(df: pd.DataFrame, metric: str, out_path: Path):
    sns.set_theme(style="whitegrid")
    data = df[df["Metric"] == metric.upper()].copy()
    if data.empty:
        return
    order = ["Ensemble", "EfficientNet", "AlexNet", "DenseNet", "Expert"]
    # Filter order to only include models present in data
    available_models = data["Model"].unique()
    order = [m for m in order if m in available_models]
    rows = []
    for model in order:
        vals = data.loc[data["Model"] == model, "Value"].to_numpy(dtype=float)
        if len(vals):
            rows.append((model, *mean_ci(vals)))
    if not rows:
        return
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    lower = [r[1] - r[2] for r in rows]
    upper = [r[3] - r[1] for r in rows]
    yerr = np.array([lower, upper])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, means, yerr=yerr, fmt="o", color="C0", ecolor="C0", elinewidth=2, capsize=6, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"{metric.upper()} mean ± 95% CI ({out_path.parent.name})")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(max(0.0, min(r[2] for r in rows) - 0.05), min(1.0, max(r[3] for r in rows) + 0.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def align_values(values_map: Dict[str, List[float]]) -> Dict[str, List[float]]:
    min_len = min(len(vals) for vals in values_map.values())
    valid_indices = []
    for i in range(min_len):
        good = True
        for vals in values_map.values():
            if not np.isfinite(vals[i]):
                good = False
                break
        if good:
            valid_indices.append(i)
    aligned = {model: [float(values_map[model][i]) for i in valid_indices] for model in values_map}
    return aligned


def analyze_metric(metric_key: str, display_label: str, values_map: Dict[str, List[float]]):
    # Separate expert from models for alignment
    # Expert has different number of observations, so align models separately
    has_expert = "expert" in values_map
    expert_vals_orig = values_map.pop("expert", [])
    
    # Align models (which should all have the same number of folds)
    aligned_models = align_values(values_map)
    
    # Restore expert to values_map for later use
    if has_expert:
        values_map["expert"] = expert_vals_orig
    
    # Check lengths
    model_lengths = {len(v) for v in aligned_models.values()}
    if not model_lengths or 0 in model_lengths:
        print(f"[skip] {display_label}: insufficient overlapping data")
        return None

    # Extract model values
    ens_vals = aligned_models["ensemble"]
    eff_vals = aligned_models["efficientnet"]
    alex_vals = aligned_models["alexnet"]
    dense_vals = aligned_models["densenet"]
    
    # Expert values (not aligned with models)
    expert_vals = expert_vals_orig if has_expert else []
    
    # Check if expert has only 1 observation (single prediction set, not multiple folds)
    expert_single_obs = has_expert and len(expert_vals) == 1

    print(f"\n{'=' * 60}\nSTATISTICAL ANALYSIS FOR {display_label}\n{'=' * 60}")
    if expert_single_obs:
        print("[NOTE] Expert has only 1 observation (single prediction set).")
        print("       Wilcoxon and Friedman tests involving Expert are skipped.")
        print("       Expert is included in descriptive statistics only.")
        print("       For statistical comparisons with Expert, use DeLong tests on per-sample predictions.\n")

    def describe(name, vals):
        mean, lo, hi = mean_ci(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        if len(vals) == 1:
            print(f"{name:<13} {mean:.4f} (single observation, no CI/SD)")
        else:
            print(f"{name:<13} {mean:.4f} ± {sd:.4f} (95% CI: {lo:.4f}–{hi:.4f})")

    print("\nDescriptive statistics:")
    describe("Ensemble:", ens_vals)
    describe("EfficientNet:", eff_vals)
    describe("AlexNet:", alex_vals)
    describe("DenseNet:", dense_vals)
    if has_expert:
        describe("Expert:", expert_vals)

    # Only perform Wilcoxon tests between models with equal number of observations
    # Wrap in try-except to handle edge cases (identical values, insufficient pairs, etc.)
    def safe_wilcoxon(x, y, name):
        """Safely perform Wilcoxon test, returning None if it fails."""
        if len(x) < 2 or len(y) < 2:
            return None
        if len(x) != len(y):
            return None
        # Check if all values are identical (no variance)
        if len(x) > 0 and np.allclose(x, x[0]) and len(y) > 0 and np.allclose(y, y[0]):
            return None
        # Check if differences are all zero (no variation in paired differences)
        try:
            diffs = np.array(x, dtype=float) - np.array(y, dtype=float)
            if len(diffs) > 0 and np.allclose(diffs, 0.0):
                return None
        except:
            pass
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                return stats.wilcoxon(x, y)
        except (ValueError, RuntimeError) as e:
            # Handle cases like insufficient valid pairs, all zeros, etc.
            return None
    
    comparisons = {}
    skipped_comparisons = []
    for name, (x, y) in [
        ("Ensemble vs EfficientNet", (ens_vals, eff_vals)),
        ("Ensemble vs AlexNet", (ens_vals, alex_vals)),
        ("Ensemble vs DenseNet", (ens_vals, dense_vals)),
        ("EfficientNet vs AlexNet", (eff_vals, alex_vals)),
        ("EfficientNet vs DenseNet", (eff_vals, dense_vals)),
        ("AlexNet vs DenseNet", (alex_vals, dense_vals)),
    ]:
        res = safe_wilcoxon(x, y, name)
        if res is not None:
            comparisons[name] = res
        else:
            skipped_comparisons.append(name)
    
    # Skip Wilcoxon tests involving expert if expert has only 1 observation
    if has_expert and not expert_single_obs:
        for name, (x, y) in [
            ("Ensemble vs Expert", (ens_vals, expert_vals)),
            ("EfficientNet vs Expert", (eff_vals, expert_vals)),
            ("AlexNet vs Expert", (alex_vals, expert_vals)),
            ("DenseNet vs Expert", (dense_vals, expert_vals)),
        ]:
            res = safe_wilcoxon(x, y, name)
            if res is not None:
                comparisons[name] = res
            else:
                skipped_comparisons.append(name)
    
    if skipped_comparisons:
        print(f"\n[Skipped comparisons (insufficient data or no variance): {', '.join(skipped_comparisons)}]")
    
    alpha = 0.05
    corrected_alpha = alpha / len(comparisons) if comparisons else alpha
    if comparisons:
        print(f"\nBonferroni-corrected alpha = {corrected_alpha:.4f}")
        metric_results = []
        for name, res in comparisons.items():
            p_adj = min(float(res.pvalue) * len(comparisons), 1.0)
            sig = p_adj < alpha
            print(f"{name}: adj p={p_adj:.4f} -> {'SIGNIFICANT' if sig else 'ns'}")
            metric_results.append({"comparison": name, "adj_p": p_adj, "statistic": float(res.statistic), "significant_adj": sig})
    else:
        metric_results = []
        print("\n[No pairwise comparisons performed - insufficient data or all models have identical values]")

    # Friedman test requires equal number of observations per group
    # Skip if expert has only 1 observation
    friedman_res = None
    try:
        if has_expert and expert_single_obs:
            # Only test models with equal folds
            friedman_res = stats.friedmanchisquare(ens_vals, eff_vals, alex_vals, dense_vals)
            print(f"\nFriedman test (models only, Expert excluded due to single observation):")
            print(f"  statistic={friedman_res.statistic:.4f}, p-value={friedman_res.pvalue:.4f} ({'SIGNIFICANT' if friedman_res.pvalue < 0.05 else 'ns'})")
        elif has_expert:
            friedman_res = stats.friedmanchisquare(ens_vals, eff_vals, alex_vals, dense_vals, expert_vals)
            print(f"\nFriedman test: statistic={friedman_res.statistic:.4f}, p-value={friedman_res.pvalue:.4f} ({'SIGNIFICANT' if friedman_res.pvalue < 0.05 else 'ns'})")
        else:
            friedman_res = stats.friedmanchisquare(ens_vals, eff_vals, alex_vals, dense_vals)
            print(f"\nFriedman test: statistic={friedman_res.statistic:.4f}, p-value={friedman_res.pvalue:.4f} ({'SIGNIFICANT' if friedman_res.pvalue < 0.05 else 'ns'})")
    except (ValueError, RuntimeError) as e:
        print(f"\n[Friedman test skipped: {e}]")
        # Create a dummy result to avoid errors downstream
        class DummyFriedman:
            statistic = float('nan')
            pvalue = float('nan')
        friedman_res = DummyFriedman()

    result = {
        "display": display_label,
        "pairwise": metric_results,
        "friedman": friedman_res,
        "values": {
            "ensemble": ens_vals,
            "efficientnet": eff_vals,
            "alexnet": alex_vals,
            "densenet": dense_vals,
        },
    }
    if has_expert:
        result["values"]["expert"] = expert_vals
    
    return result


def analyze_dataset(per_fold_data: Dict[str, List[Dict]], label: str):
    print(f"\n{'=' * 70}\nANALYSIS GROUP: {label.upper()}\n{'=' * 70}")

    results = {}
    metric_order: List[str] = []
    ens = per_fold_data["ensemble"]
    eff = per_fold_data["efficientnet"]
    alex = per_fold_data["alexnet"]
    dense = per_fold_data["densenet"]
    has_expert = "expert" in per_fold_data
    expert = per_fold_data.get("expert", [])

    def process_metric(metric_key: str, display_label: str, values_map: Dict[str, List[float]]):
        analysis = analyze_metric(metric_key, display_label, values_map)
        if analysis:
            results[metric_key] = analysis
            metric_order.append(metric_key)

    for metric in BASE_METRICS:
        values_map = {
            "ensemble": extract_metrics(ens, metric),
            "efficientnet": extract_metrics(eff, metric),
            "alexnet": extract_metrics(alex, metric),
            "densenet": extract_metrics(dense, metric),
        }
        if has_expert:
            values_map["expert"] = extract_metrics(expert, metric)
        process_metric(metric, f"{metric.upper()} ({label})", values_map)

    for premolar_key in PREMOLAR_KEYS:
        for metric in BASE_METRICS:
            values_map = {
                "ensemble": extract_premolar_series(ens, premolar_key, metric),
                "efficientnet": extract_premolar_series(eff, premolar_key, metric),
                "alexnet": extract_premolar_series(alex, premolar_key, metric),
                "densenet": extract_premolar_series(dense, premolar_key, metric),
            }
            if has_expert:
                values_map["expert"] = extract_premolar_series(expert, premolar_key, metric)
            metric_key = f"{metric}_{premolar_key}"
            display = f"{metric.upper()} [{premolar_key}] ({label})"
            process_metric(metric_key, display, values_map)

    print(f"\n{'=' * 60}\nSUMMARY OF SIGNIFICANT DIFFERENCES ({label})\n{'=' * 60}")
    for metric_key in metric_order:
        sigs = [r for r in results[metric_key]["pairwise"] if r["significant_adj"]]
        if sigs:
            print(f"\n{results[metric_key]['display']}:")
            for comp in sigs:
                print(f"  - {comp['comparison']}: adj p={comp['adj_p']:.4f}")
        else:
            print(f"\n{results[metric_key]['display']}: No significant differences")

    # Check if expert has only 1 observation (check first metric as representative)
    expert_single_obs = False
    if has_expert and metric_order:
        first_metric_vals = results[metric_order[0]]["values"]
        expert_single_obs = "expert" in first_metric_vals and len(first_metric_vals["expert"]) == 1
    
    summary_rows = []
    # Only include expert comparisons in table if expert has multiple observations
    if has_expert and not expert_single_obs:
        pairwise_names_to_use = PAIRWISE_NAMES_WITH_EXPERT
    else:
        pairwise_names_to_use = PAIRWISE_NAMES
    for metric_key in metric_order:
        row = {"Metric": results[metric_key]["display"]}
        for name in pairwise_names_to_use:
            comp = next((r for r in results[metric_key]["pairwise"] if r["comparison"] == name), None)
            if comp:
                adj_p = comp["adj_p"]
                stars = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else ""
                row[name] = f"{adj_p:.4f}{stars}"
            else:
                row[name] = "N/A"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    print(f"\nSummary table (Bonferroni-adjusted p-values) [{label}]:")
    if expert_single_obs:
        print("[NOTE] Expert comparisons excluded from table (single observation). Use DeLong tests for Expert comparisons.")
    print(summary_df.to_string(index=False))

    print(f"\n{'=' * 60}\nEFFECT SIZES (Ensemble vs others) [{label}]\n{'=' * 60}")
    for metric_key in metric_order:
        vals = results[metric_key]["values"]
        effect_str = (f"{results[metric_key]['display']}: "
                      f"Ens vs Eff d={cohens_d(vals['ensemble'], vals['efficientnet']):.3f}; "
                      f"Ens vs Alex d={cohens_d(vals['ensemble'], vals['alexnet']):.3f}; "
                      f"Ens vs Dense d={cohens_d(vals['ensemble'], vals['densenet']):.3f}")
        if has_expert and "expert" in vals and not expert_single_obs:
            effect_str += f"; Ens vs Expert d={cohens_d(vals['ensemble'], vals['expert']):.3f}"
        elif has_expert and "expert" in vals and expert_single_obs:
            effect_str += "; Ens vs Expert d=N/A (expert has single observation)"
        print(effect_str)

    desc_rows = []
    model_list = [("ensemble", "Ensemble"), ("efficientnet", "EfficientNet"), ("alexnet", "AlexNet"), ("densenet", "DenseNet")]
    if has_expert:
        model_list.append(("expert", "Expert"))
    for metric_key in metric_order:
        row = {"Metric": results[metric_key]["display"]}
        for model_key, label_name in model_list:
            if model_key in results[metric_key]["values"]:
                vals = results[metric_key]["values"][model_key]
                mean, lo, hi = mean_ci(vals)
                if len(vals) == 1:
                    row[label_name] = f"{mean:.4f} (single obs)"
                else:
                    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    row[label_name] = f"{mean:.4f} ± {sd:.4f} ({lo:.4f}–{hi:.4f})"
        desc_rows.append(row)
    print(f"\nDescriptive summary [{label}]:")
    if expert_single_obs:
        print("[NOTE] Expert has single observation. For statistical comparisons with Expert, use DeLong tests (see compare_held_out_delong.py).")
    print(pd.DataFrame(desc_rows).to_string(index=False))

    df_long = build_long_df(results, metric_order)
    plot_dir = Path(f"results/plots_all/plots_{label}")
    plot_dir.mkdir(exist_ok=True, parents=True)
    for metric_key in metric_order:
        plot_metric_boxplot(df_long, metric_key, plot_dir / f"boxplot_{metric_key}.png")
    print(f"Saved boxplots to {plot_dir.resolve()}")

    ci_dir = Path(f"results/plots_all/plots_ci_{label}")
    ci_dir.mkdir(exist_ok=True, parents=True)
    for metric_key in metric_order:
        plot_metric_mean_ci(df_long, metric_key, ci_dir / f"{metric_key}_mean_95ci.png")
    print(f"Saved mean±CI plots to {ci_dir.resolve()}")


if __name__ == "__main__":
    inference_data = load_per_fold_results("inference")
    analyze_dataset(inference_data, "inference")

    held_out_data = load_per_fold_results("held_out")
    analyze_dataset(held_out_data, "held_out")
