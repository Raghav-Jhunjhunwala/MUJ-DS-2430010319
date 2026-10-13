# TabM-mini with Geo-Feature Engineering on California Housing

> **Batch F – NLP/Deep Learning Capstone (Manipal University Jaipur)**

## 1. Student Details

| Field | Value |
|---|---|
| Name | `Raghav Jhunjhunwala` |
| Registration Number | `2430010319` |
| Branch | `Data-Science` |
| Batch | F |
| GitHub Username | `Raghav-Jhunjhunwala` |
| Training Program | `Data Science & Deep Learning Capstone Training Program (Batch F)` |
| Project Title | TabM-mini (parameter-efficient deep ensemble) with Geographic Feature Engineering for California Housing Price Classification |
| Capstone (team) repo | `<URL>` |

## 2. Dataset

| Item | Value |
|---|---|
| Name | California Housing (Pace & Barry, 1997; 1990 US census) |
| Source | `sklearn.datasets.fetch_california_housing` |
| Samples / features | 20,640 districts / 8 raw features (+13 engineered in the proposed model) |
| Task | Binary classification: median house value **>= dataset median** (1) or below (0) |
| Classes | 2 (≈ balanced because the threshold is the median) |
| Split | 60 % train / 20 % validation / 20 % test, stratified, seed 42 |
| Preprocessing | Median-threshold labelling → (proposed) geo features → `QuantileTransformer(normal)` fitted on train only |

Exact counts: `results/class_distribution.csv`, `results/split_distribution.csv`, `results/preprocessing_steps.csv`.

## 3. Model

* **Architecture:** TabM-mini – *k = 8* MLPs that share weights through BatchEnsemble-style random ±1 input scalings, fed by **periodic numerical embeddings**; predictions are averaged at inference. Implemented from scratch in PyTorch (`src/model.py`).
* **Pretrained:** none (trained from scratch).
* **Proposed modification:** **geographic feature engineering** (see §8).
* Trainable parameters: see `results/results_geo.csv` (`Parameters` column).

## 4. Hyperparameters (final, from `results/best_params_geo.json`)

| Hyperparameter | Value |
|---|---|
| Learning rate | `<fill from best_params_geo.json>` |
| Batch size | 1024 |
| Epochs | max 80, early stopping (patience 10) |
| Optimizer | AdamW |
| Loss | Binary cross-entropy (applied to each of the k ensemble members) |
| Scheduler | CosineAnnealingLR |
| Dropout | `<fill>` |
| Weight decay | `<fill>` |
| Hidden size / layers | `<fill>` / `<fill>` |
| Embedding dim / sigma / n_freq | `<fill>` / `<fill>` / 8 |
| Ensemble size k | 8 |
| Seed | 42 |

## 5. Hyperparameter Tuning

Random search, 10 trials, selection by **validation BCE** (test set never used).

| Parameter | Values tested |
|---|---|
| lr | 1e-3, 2e-3, 4e-3 |
| weight decay | 1e-4, 1e-3, 1e-2 |
| hidden | 64, 128, 256 |
| layers | 2, 3 |
| dropout | 0.0, 0.1, 0.2 |
| d_embed | 8, 16 |
| sigma | 0.05, 0.1, 0.3 |

All trials and the winner: `results/tuning_trials_geo.csv`. **Reason for selection:** lowest validation loss (a proper scoring rule, less noisy than accuracy).

## 6. Results

> Run the scripts, then paste the tables from `results/summary.md` here.

| Metric (test) | Value |
|---|---|
| Accuracy / Precision / Recall / Specificity / F1 / AUC | `<results/results_geo.csv>` |
| 5-fold CV accuracy (mean ± std) | `<results/cv_geo.csv>` |
| Train / Val / Test accuracy (bias–variance) | `<results/train_val_test_metrics_geo.csv>` |

Figures (in `figures/`): `confusion_matrix_geo.png`, `training_curves_geo.png`, `roc_geo.png`, `pr_curve_geo.png`, `prob_hist_geo.png`, `feature_importance_geo.png`, `bias_variance_geo.png`, `error_map_geo.png`, `cv_accuracy_geo.png`, EDA plots `01–03`.

### CO4 – Comparison (same dataset, same split): `results/comparison.csv`, `figures/comparison.png`

Baselines re-trained by me on the identical split: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, sklearn MLP, PyTorch MLP, original TabM-mini, and GBDTs given the geo features too (fairness check).

> **Note on published SOTA:** the TabM, FT-Transformer, MLP-PLR papers report California Housing as **regression (RMSE)**, not this binary task, so their numbers are **not directly comparable**. I therefore re-implemented/re-ran comparable methods on my split and cite the papers for the methods themselves (`resources/references.md`). Write 3–4 sentences here explaining *why* the proposed model is better/similar/worse than each baseline, based on your table (e.g. whether GBDT wins, and why trees suit tabular data).

### CO5 – Ablation: `results/ablation.csv`, `figures/ablation.png`

| Experiment | Geo feats | Periodic emb. | Ensemble k | Accuracy | F1 | AUC |
|---|---|---|---|---|---|---|
| Baseline MLP | N | N | 1 | | | |
| + Periodic embeddings | N | Y | 1 | | | |
| + BatchEnsemble (TabM-mini) | N | Y | 8 | | | |
| **+ Geo features (Proposed)** | Y | Y | 8 | | | |

(mean ± std over 3 seeds; copy from `results/ablation.csv`)

## 7. Reproduce

```bash
pip install -r requirements.txt
python src/train.py --features geo     # proposed model (add --fast for a ~2 min run)
python src/train.py --features base    # original TabM-mini, needed by compare.py
python src/ablation.py
python src/compare.py
python src/test.py --features geo      # reload saved model & re-test
python src/make_tables.py              # results/summary.md
```
Or run `notebooks/experiments.ipynb` in Colab (GPU optional).

## 8. Innovation

**What is new compared with the baseline (original TabM-mini code / plain MLP)?**

1. **Geographic preprocessing strategy.** Prices in California depend on *where* a district is (coast, big cities), which is a diagonal, non-linear function of raw latitude/longitude. I add: rotated coordinates (15°/30°/45°), distance to LA, SF, San Diego, San Jose, Sacramento and to the nearest of them, plus rooms/bedrooms ratio. These feed the periodic-embedding TabM-mini.
2. **Controlled ablation** isolating each component (embeddings, ensembling, geo features) with multi-seed mean ± std, plus a fairness check giving GBDTs the same features.
3. **Rigorous protocol:** scaler fitted on train only, tuning on validation only, test touched once.

Justification and evidence: `results/ablation.csv`. *(Edit this section honestly after seeing your numbers – if geo features help little, say so and discuss why.)*

## 9. Folder Structure

```
MUJ-DS-<RegNo>/
├── README.md
├── requirements.txt
├── .gitignore
├── assignments/            # weekly training assignments
├── notebooks/experiments.ipynb
├── code/                   # run_all.sh (entry point) – source lives in src/
├── src/
│   ├── preprocessing.py    # loading, geo features, split, scaling, distribution tables
│   ├── model.py            # PeriodicEmbedding, TabMMini
│   ├── trainer.py          # training loop, prediction, metrics
│   ├── train.py            # full pipeline (EDA, tuning, CV, final, plots)
│   ├── ablation.py         # CO5
│   ├── compare.py          # CO4
│   ├── test.py             # reload model & test
│   └── make_tables.py
├── data/dataset_information.txt
├── results/                # CSVs, JSON, summary.md
├── figures/                # PNGs
├── models/                 # saved weights + model_description.txt
├── resources/              # references.md, github_workflow.md
├── presentations/          # slides + outline
└── capstone/               # link/notes for the team capstone repo
```
