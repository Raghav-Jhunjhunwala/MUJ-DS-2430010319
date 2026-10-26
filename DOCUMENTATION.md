# Project Documentation

**Title:** TabM-mini (parameter-efficient deep ensemble) with Geographic Feature Engineering for California Housing Price Classification
**Author:** Raghav Jhunjhunwala | **Reg. No.:** 2430010319 | **Branch:** Data Science | **Batch:** F
**GitHub:** Raghav-Jhunjhunwala | **Program:** Data Science & Deep Learning Capstone Training Program (Batch F)

For setup and run commands see [INSTALLATION.md](INSTALLATION.md). For the full results tables see [README.md](README.md).

## 1. Problem statement

Given the 8 census features of a California district (income, house age, rooms, bedrooms, population,
occupancy, latitude, longitude), predict whether its median house value is **at or above the dataset median (class 1)**
or below it (class 0). Because the threshold is the median, the two classes are almost balanced.

## 2. Dataset

- **Source:** `sklearn.datasets.fetch_california_housing` (Pace & Barry, 1997; 1990 US census), downloaded automatically.
- **Size:** 20,640 districts, 8 raw features.
- **Split:** 60% train / 20% validation / 20% test, stratified, seed 42.
- Details: `dataset_information.txt`.

## 3. Method

1. **Preprocessing** (`src/preprocessing.py`): median-threshold labelling, optional geo features,
   stratified split, `QuantileTransformer(normal)` fitted on the training set only (no data leakage).
2. **Geographic feature engineering (the innovation):** 13 extra features, giving 21 in total:
   - rotated latitude/longitude at 15°, 30° and 45°, to capture diagonal coastlines;
   - distance to Los Angeles, San Francisco, San Diego, San Jose, Sacramento, and to the nearest of them;
   - rooms-per-bedroom ratio.
3. **Model** (`src/model.py`): TabM-mini built from scratch in PyTorch. Periodic (sin/cos) numerical embeddings feed
   k = 8 MLPs that share weights through BatchEnsemble-style random ±1 scalings. The 8 predictions are averaged.
   69,248 trainable parameters, no pretrained weights.
4. **Training** (`src/trainer.py`, `src/train.py`): AdamW, cosine annealing, binary cross-entropy on each ensemble member,
   early stopping (patience 10, max 80 epochs).
5. **Tuning:** 10-trial random search, selected by validation loss only (the test set is never used for choices).
6. **Evaluation:** test metrics, 5-fold cross-validation, train/validation/test bias-variance check,
   permutation feature importance, and an error map.

## 4. Source code guide

| File | Purpose |
|---|---|
| `src/preprocessing.py` | Load data, geo features, split, scaling, distribution tables |
| `src/model.py` | `PeriodicEmbedding` and `TabMMini` classes, parameter counting |
| `src/trainer.py` | Model builder, training loop, prediction, metrics, seeding |
| `src/train.py` | Full pipeline: EDA, tuning, CV, final training, test evaluation, plots |
| `src/ablation.py` | Ablation study over 3 seeds (CO5) |
| `src/compare.py` | Comparison with baseline models on the same split (CO4) |
| `src/test.py` | Reload the saved model and re-evaluate on the test set |
| `src/make_tables.py` | Convert `results/*.csv` into `results/summary.md` |
| `code/run_all.sh` | Runs the whole pipeline in order |
| `notebooks/experiments.ipynb` | Notebook version of the pipeline for Colab |

## 5. Repository layout

| Path | Contents |
|---|---|
| `assignments/` | Weekly training assignments |
| `notebooks/` | Experiment notebook |
| `code/` | Entry-point script (`run_all.sh`) |
| `src/` | Source code |
| `models/` | Saved weights and `model_description.txt` |
| `results/` | CSV/JSON metrics and tables |
| `figures/` | Plots |
| `screenshots/` | Screenshots of the project running |
| `presentations/` | Project presentation |
| `resources/` | References and GitHub workflow notes |
| `capstone/` | Capstone notes (solo project) |

## 6. Key results

| Measure | Value |
|---|---|
| Test accuracy | 0.9031 |
| Test F1 / Macro-F1 | 0.9027 / 0.9031 |
| Test ROC-AUC | 0.9697 |
| 5-fold CV accuracy | 0.8954 ± 0.0027 |
| Train / validation / test accuracy | 0.9280 / 0.9067 / 0.9031 |

- **Ablation:** periodic embeddings gave the largest gain (+2.2 points); BatchEnsemble added +0.2 (within noise);
  geo features added +1.2 points on top of TabM-mini (0.8924 → 0.9046).
- **Comparison:** the model beats all neural baselines but not the best gradient-boosted trees
  (XGBoost with geo features: 0.9099 accuracy).
- **Generalization:** the ~2-point train-validation gap shows no serious overfitting.

## 7. Limitations

- One dataset, and a binary threshold task rather than regression.
- Tree models (XGBoost) remain slightly stronger on this data.
- The BatchEnsemble gain is small and within one standard deviation.
- Results can vary by a few thousandths between machines.

## 8. References

See `resources/references.md` (TabM, periodic embeddings, FT-Transformer, BatchEnsemble, XGBoost, Random Forests,
California Housing, scikit-learn).

## 9. Reproducibility

Fixed seed (42), scaler fitted on train only, hyperparameters chosen on validation data only, and the saved model
can be re-tested with `python src/test.py --features geo`.
