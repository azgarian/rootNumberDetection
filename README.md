# Premolar Morphology Pipeline

End-to-end pipeline for recreating the dataset preprocessing, model training, evaluation, and statistical reporting used in this repository.

> **Environment**  
> Python ≥3.10 with PyTorch + torchvision, numpy, pandas, scipy, seaborn, matplotlib, scikit-learn, and Pillow installed. CUDA is optional but strongly recommended for training.

---

## 1. Data and Preprocessing

| Artifact | Description |
| --- | --- |
| `data/patient_root_info.csv` | Tooth-level metadata (root counts plus image-quality flags) used by `crop_images.ipynb` to generate `data/crops_all/`. |
| `data/held_out_test/patient_root_info.csv` | Same schema but for the held-out split (used by `crop_images.ipynb`). |
| `data/patient_to_id.csv` | Mapping between anonymized IDs and raw file names, used only inside preprocessing notebooks (e.g., `src/preprocess/name2ids.ipynb`) to rename raw JPGs before training. |
| `data/raw_images/` | Original panoramic JPGs whose filenames get anonymized via `name2ids.ipynb` before cropping. |
| `data/crops_all/` | ImageFolder root (class subdirectories `1_root_images/` and `2_root_images/`) created after all preprocessing/cropping steps. |
| `data/held_out_test/crops/` | Held-out ImageFolder used only for the final evaluation and model vs. expert comparison. |
| `data/held_out_test/expert_pred.csv` | Expert annotations per patient/tooth used by `evaluate_held_out.py --model expert`. |

> **Note on data availability**  
> None of the files above are bundled with this repository because they contain patient data. You must prepare them yourself (or request approved access) using the schema examples below.

### 1.1 CSV schema examples

`patient_root_info.csv` (and `held_out_test/patient_root_info.csv`)

Columns 2–5 hold the confirmed root counts per tooth (ground truth for comparison), while columns 6–9 store their image quality values

| patient_id | 14 | 24 | 15 | 25 | 14.1 | 24.1 | 15.1 | 25.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patient001 | 2 | 2 | 1 | 1 | 1 | 1 | 2 | 2 |
| patient002 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 |
| patient003 | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 1 |

`patient_to_id.csv`

| patient_name | patient_id | file_name |
| --- | --- | --- |
| name1 surname1 | patient001 | name1_surname1_pano.jpg |
| name2 surname2 | patient002 | name2_surname2_pano.jpg |

`held_out_test/expert_pred.csv`

Columns 2–5 hold the confirmed root counts per tooth (ground truth for comparison), while columns 6–9 store the expert predictions entered manually.

| patient | 14 | 15 | 24 | 25 | expert pred 14 | expert pred 15 | expert pred 24 | expert pred 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patient001 | 2 | 1 | 2 | 1 | 1 | 1 | 2 | 1 |
| patient002 | 2 | 1 | 1 | 1 | 2 | 2 | 1 | 1 |

### 1.1 Preprocess steps

1. **Volume frame selection** – crop coordinates were specified with the DentBB tool by drawing bounding boxes around the root area of interest.
2. **`src/preprocess/name2ids.ipynb`** – keeps the lookup tables (`patient_info.csv`, `patient_to_id.csv`) synchronized with the anonymized file naming scheme.
3. **`src/preprocess/crop_images.ipynb`** – crops the selected frames using the tooth-specific bounding boxes.

At the end of this stage you should have:

```
data/
  crops_all/
    1_root_images/
    2_root_images/
  held_out_test/
    crops/
    expert_pred.csv
  patient_info.csv
  patient_to_id.csv
```

---

## 2. Training

`src/training/train.py` performs grouped 5-fold cross-validation with patient-level disjoint splits, deterministic per-class sampling for validation/test sets, heavy oversampling of the training folds, and automated checkpointing.

```bash
python src/training/train.py --model alexnet
```

Key behaviors:

- Splits patients so that the same ID never appears in two folds (`_patient_id_from_path`, `_select_exact_groups`).
- Saves each fold under `results/model/<model>/fold_XX/` with `best.pt`, `test_patients.txt`, `val_patients.txt`, metrics, and loss curves.
- Uses `dataloader.py` for torchvision transforms and oversampling helpers, and `models.py` for the attention-enhanced classifier heads.

Modify the hard-coded constants at the top of `train.py` (data root, epochs, batch size, etc.) before launching if you need different settings.

---

## 3. Per-fold Inference & Visualization

After training, regenerate metrics and plots for every fold via `src/training/inference.py`. This script reloads the saved checkpoints, rebuilds the fold-specific test subsets, and writes:

- `metrics_test.json`, confusion matrix, ROC plots (standard and high-DPI)
- `y_true_test.npy`, `y_pred_test.npy`, `y_prob_test.npy`, `paths_test.npy` (used later for DeLong statistics)

```bash
python src/training/inference.py \
  --run-dir results/model/alexnet \
  --data-dir data/crops_all \
  --model alexnet \
  --outdir results/inference/alexnet
```

Repeat for every backbone you trained (e.g., `efficientnet_b0`, `densenet121`). The outputs live under `results/inference/<model>/fold_XX/`.

---

## 4. Cross-backbone Ensemble on CV Folds

`src/training/ensemble_cross_backbone.py` averages probabilities from the AlexNet, EfficientNet-B0, and DenseNet-121 folds that share the same patient partitions. For each fold it:

1. Loads the corresponding `best.pt` from every backbone.
2. Averages their per-sample probabilities.
3. Saves ensemble metrics and plots to `results/inference/ensemble_model/fold_XX/`.

Simply run:

```bash
python src/training/ensemble_cross_backbone.py
```

Ensure `results/model/<backbone>/fold_XX/best.pt` exists for all backbones before launching.

---

## 5. Held-out Evaluation

`src/training/evaluate_held_out.py` unifies the evaluation of:

- Individual backbones (`--model alexnet`, `--run-dir outputs/alexnet`, etc.)
- Expert annotations (`--model expert --expert-csv data/held_out_test/expert_pred.csv`)
- Probability-level ensembles (`--model ensemble --models alexnet efficientnet_b0 densenet121`)

Sample commands:

```bash
# Single backbone
python src/training/evaluate_held_out.py \
  --run-dir outputs/alexnet \
  --data-dir data/held_out_test/crops \
  --model alexnet \
  --outdir results/held_out_results/alexnet

# Expert baseline
python src/training/evaluate_held_out.py \
  --data-dir data/held_out_test/crops \
  --model expert \
  --expert-csv data/held_out_test/expert_pred.csv \
  --outdir results/held_out_results/expert

# Ensemble of trained models
python src/training/evaluate_held_out.py \
  --run-dir outputs \
  --data-dir data/held_out_test/crops \
  --model ensemble \
  --models alexnet efficientnet_b0 densenet121 \
  --outdir results/held_out_results/ensemble
```

Outputs include per-fold metrics, overall summaries, and the `.npy` files (`y_true_held_out.npy`, `y_prob_held_out.npy`, `paths_held_out.npy`) required for FDR-controlled statistical tests.

---

## 6. Statistical Comparisons & Reporting

### 6.1 Held-out AUC comparisons
`src/training/compare_held_out_delong.py` runs pairwise paired DeLong tests between any two held-out result folders. Example:

```bash
python src/training/compare_held_out_delong.py \
  --model1-dir results/held_out_results/ensemble \
  --model2-dir results/held_out_results/expert \
  --outfile results/held_out_results/ensemble_vs_expert_delong.json \
  --model1-name Ensemble \
  --model2-name Expert
```

The script aligns samples by path, computes the AUC difference, confidence interval, and p-value, and writes a JSON report suitable for the Methods section.

### 6.2 Comprehensive fold-level metrics
Run `src/training/comprehensive_metrics_analysis.py` after all per-fold `.json` summaries exist in `results/inference/` or `results/held_out_results/`. The script:

- Loads every fold’s metrics for each model (and the expert if available).
- Computes Wilcoxon signed-rank tests, Bonferroni corrections, and Friedman tests across base metrics and per-premolar subsets.
- Generates Seaborn boxplots and mean±CI figures under `results/statistics/<kind>/`.
- Prints textual summaries emphasizing statistically significant pairwise differences.

Invoke it with no arguments (defaults are embedded), or adjust the `load_per_fold_results` helper if you store outputs elsewhere.

---

## 7. Reproducibility Notes

- **Randomness**: `train.py` accepts `--seed`, which is propagated to PyTorch samplers, patient-group selection, and oversampling. Set it explicitly to match the paper’s results.
- **Data privacy**: Medical images and expert CSVs are not part of the repository. Provide equivalent synthetic data or obtain the proper approvals before re-running.
- **Extensibility**: To add a new backbone, implement it inside `models.py`, ensure it is recognized by `build_model`, and repeat the train → inference → evaluate steps above.

