# TabM-mini with Geo-Feature Engineering on California Housing

> **Batch D – NLP/Deep Learning Capstone (Manipal University Jaipur)**

## 1. Student Details

| Field | Value |
|---|---|
| Name | `Raghav Jhunjhunwala` |
| Registration Number | `2430010319` |
| Branch | `Data-Science` |
| Batch | D |
| GitHub Username | `Raghav-Jhunjhunwala` |
| Training Program | `Data Science & Deep Learning Capstone Training Program ` |
| Project Title | TabM-mini (parameter-efficient deep ensemble) with Geographic Feature Engineering for California Housing Price Classification |


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
* Trainable parameters: 69,248.
  
## 4. Hyperparameters (final, from `results/best_params_geo.json`)

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.002 |
| Batch size | 1024 |
| Epochs | max 80, early stopping (patience 10) |
| Optimizer | AdamW |
| Loss function | Binary cross-entropy (on each of the k ensemble members) |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.2 |
| Weight decay | 0.01 |
| Hidden size / layers | 128 / 2 |
| Embedding dim / sigma / n_freq | 16 / 0.3 / 8 |
| Ensemble size k | 8 |
| Seed | 42 |

## 5. Hyperparameter Tuning

Random search, 10 trials, selection by **validation BCE** (test set never used). Full log: `results/tuning_trials_geo.csv`.

| Parameter | Values tested |
|---|---|
| lr | 1e-3, 2e-3, 4e-3 |
| weight decay | 1e-4, 1e-3, 1e-2 |
| hidden | 64, 128, 256 |
| layers | 2, 3 |
| dropout | 0.0, 0.1, 0.2 |
| d_embed | 8, 16 |
| sigma | 0.05, 0.1, 0.3 |

Validation loss ranged from 0.2448 (best, trial 7) to 0.2991 (worst, trial 9). **Selected:** trial 7 – lowest validation loss. Small hidden size (64) was consistently worse; larger embeddings (16) and sigma 0.3 appeared in the best trials.

## 6. Results

**Test set (proposed model, `results/results_geo.csv`)**

| Accuracy | Precision | Recall | Specificity | F1 | Macro-F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9031 | 0.9071 | 0.8983 | 0.9079 | 0.9027 | 0.9031 | 0.9697 |

Per class (precision / recall / F1): class 0 = 0.90 / 0.91 / 0.90 (2063); class 1 = 0.91 / 0.90 / 0.90 (2065).

**5-fold cross-validation accuracy:** 0.8962, 0.8958, 0.8987, 0.8958, 0.8905 → **mean 0.8954 ± 0.0027**.

**Bias–variance (`results/train_val_test_metrics_geo.csv`)**

| Split | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Train | 0.9280 | 0.9282 | 0.9812 |
| Validation | 0.9067 | 0.9068 | 0.9689 |
| Test | 0.9031 | 0.9027 | 0.9697 |

The train–validation gap is ~2 points, so the model **generalizes well** (no serious overfitting or underfitting); validation and test agree closely.

**Feature importance** (permutation, AUC drop, `figures/feature_importance_geo.png`): top features are `Rot45_a` (0.066), `MedInc` (0.044), `AveOccup` (0.042), `Dist_NearestCity` (0.031) – three of the top five are engineered geo features, which supports the innovation.

Computational cost: ~26 s total pipeline on a Colab GPU; 69,248 trainable parameters.

Figures in `figures/`: `confusion_matrix_geo.png`, `training_curves_geo.png`, `roc_geo.png`, `pr_curve_geo.png`, `prob_hist_geo.png`, `feature_importance_geo.png`, `bias_variance_geo.png`, `error_map_geo.png`, `cv_accuracy_geo.png`, `01_class_distribution.png`, `02_feature_distributions.png`, `03_correlation_heatmap.png`.

### CO4 – Comparison on the same dataset and split (`results/comparison.csv`, `figures/comparison.png`)

| Model | Features | Accuracy | Precision | Recall | F1 | AUC | Parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost + geo feats | geo | 0.9099 | 0.9127 | 0.9065 | 0.9096 | 0.9722 | – |
| HistGradientBoosting + geo feats | geo | 0.9048 | 0.9058 | 0.9036 | 0.9047 | 0.9704 | – |
| **Proposed: TabM-mini + geo feats** | geo | 0.9031 | 0.9103 | 0.8944 | 0.9023 | 0.9687 | 69,248 |
| XGBoost | base | 0.9050 | 0.9082 | 0.9012 | 0.9047 | 0.9670 | – |
| HistGradientBoosting | base | 0.9043 | 0.9007 | 0.9090 | 0.9048 | 0.9658 | – |
| Random Forest (300 trees) | base | 0.8932 | 0.8996 | 0.8852 | 0.8924 | 0.9617 | – |
| TabM-mini (original) | base | 0.8907 | 0.8933 | 0.8877 | 0.8905 | 0.9593 | 37,320 |
| sklearn MLP (128-64) | base | 0.8706 | 0.8837 | 0.8538 | 0.8685 | 0.9450 | 9,473 |
| PyTorch MLP | base | 0.8685 | 0.8749 | 0.8600 | 0.8674 | 0.9433 | 17,801 |
| Logistic Regression | base | 0.8389 | 0.8352 | 0.8446 | 0.8399 | 0.9195 | 9 |

(Deep models here were retrained inside `compare.py`, so they differ by a few thousandths from the `train.py` run above.)

**Analysis.**
* The proposed model beats every *neural* baseline: +1.2 points accuracy over the original TabM-mini, +3.5 over a plain MLP, and +6.4 over logistic regression – nonlinear models clearly matter for this location-driven problem.
* It does **not** beat the best gradient-boosted trees. XGBoost with the same geo features is ahead (accuracy 0.9099, AUC 0.9722), and even plain XGBoost (0.9050) is slightly ahead of the proposed accuracy. This matches the published finding that GBDTs remain very strong on small tabular data; trees split on raw latitude/longitude naturally, while a neural net must learn that structure, which the geo features help with.
* The proposed model has the highest precision among the top group (0.9103) but lower recall (0.8944), i.e. it is slightly more conservative.
* **Note on published SOTA:** the TabM, FT-Transformer and periodic-embedding papers report California Housing as *regression (RMSE)*, so their numbers are not comparable to this binary task. I therefore re-ran comparable methods on my own split and cite the papers for the methods (`resources/references.md`).

### CO5 – Ablation (`results/ablation.csv`, `figures/ablation.png`; mean ± std over 3 seeds, test set)

| Experiment | Geo feats | Periodic emb. | Ensemble k | Params | Accuracy | F1 | AUC |
|---|:-:|:-:|:-:|---:|---:|---:|---:|
| Baseline MLP | ✗ | ✗ | 1 | 17,801 | 0.8681 ± 0.0011 | 0.8660 ± 0.0010 | 0.9430 ± 0.0002 |
| + Periodic embeddings | ✗ | ✓ | 1 | 35,521 | 0.8903 ± 0.0029 | 0.8894 ± 0.0031 | 0.9612 ± 0.0008 |
| + BatchEnsemble k=8 (TabM-mini) | ✗ | ✓ | 8 | 37,320 | 0.8924 ± 0.0035 | 0.8921 ± 0.0037 | 0.9608 ± 0.0003 |
| **+ Geo features (Proposed)** | ✓ | ✓ | 8 | 69,248 | **0.9046 ± 0.0007** | **0.9040 ± 0.0004** | **0.9698 ± 0.0006** |
| MLP + Geo features only | ✓ | ✗ | 1 | 19,478 | 0.8907 ± 0.0021 | 0.8900 ± 0.0020 | 0.9600 ± 0.0004 |

**Findings.** Periodic embeddings give the largest single gain (+2.2 points accuracy). BatchEnsemble adds only +0.2 points, which is within one standard deviation – an honest, small effect here. Geo features add a further +1.2 points on top of TabM-mini and are also useful on their own: a plain MLP with geo features (0.8907) matches the original TabM-mini. Combining embeddings and geo features gives the best neural result, with low variance across seeds.


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

**What is new compared with the baseline / state-of-the-art methods?**

1. **Geographic preprocessing strategy.** House prices depend on *where* a district is (coast, big cities), which is a diagonal, non-linear function of raw latitude/longitude. I add rotated coordinates (15°/30°/45°), distance to LA, SF, San Diego, San Jose, Sacramento and to the nearest of them, plus a rooms/bedrooms ratio (13 new features). These feed the periodic-embedding TabM-mini.
2. **Controlled ablation** isolating each component (embeddings, ensembling, geo features) with multi-seed mean ± std, plus a fairness check giving gradient-boosted trees the same features.
3. **Rigorous protocol:** scaler fitted on train only, tuning on validation only, test set used once for final reporting.

**Evidence:** geo features raise accuracy from 0.8924 to 0.9046 (+1.2 pts) and AUC from 0.9608 to 0.9698 in the ablation, and engineered features rank among the top permutation-importance features. **Limitation:** XGBoost with the same features still scores slightly higher (0.9099), so the contribution is an improvement of the neural model, not a new overall best.

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
