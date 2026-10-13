"""Main pipeline: EDA -> tuning -> CV -> final training -> test evaluation -> plots.

Usage:
    python src/train.py                 # proposed model (TabM-mini + geo features)
    python src/train.py --features base # original TabM-mini on the 8 raw features
    python src/train.py --fast          # quick run (~2 min), lower accuracy
"""
import argparse, json, random, sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score, roc_auc_score)
import preprocessing as pp
from model import count_params
from trainer import (train_model, predict_proba, compute_metrics, set_seed, device, SEED)

ap = argparse.ArgumentParser()
ap.add_argument("--features", choices=["base", "geo"], default="geo")
ap.add_argument("--fast", action="store_true")
args = ap.parse_args()
if args.fast:
    N_TRIALS, TUNE_EP, CV_FOLDS, CV_EP, FINAL_EP = 3, 8, 3, 8, 30
else:
    N_TRIALS, TUNE_EP, CV_FOLDS, CV_EP, FINAL_EP = 10, 25, 5, 25, 80
TAG = args.features
R, FG, M = "results", "figures", "models"
for d in (R, FG, M):
    os.makedirs(d, exist_ok=True)
sns.set_style("whitegrid"); set_seed(); t_start = time.time()
print("Device:", device)

# ---------------- data ----------------
X, y, names = pp.load_data()
X_raw_names = names
if args.features == "geo":
    X, names = pp.add_geo_features(X, names)
Xtr, Xva, Xte, ytr, yva, yte = pp.split_data(X, y)
(Xs_tr, Xs_va, Xs_te), qt = pp.quantile_scale(Xtr, Xva, Xte)
cls_tbl, split_tbl = pp.distribution_tables(y, ytr, yva, yte)
cls_tbl.to_csv(f"{R}/class_distribution.csv", index=False)
split_tbl.to_csv(f"{R}/split_distribution.csv", index=False)
pd.DataFrame({"Step": ["Task", "Features used", "Split", "Scaling", "Leakage control"],
              "Detail": ["Binary: MedHouseVal >= median", f"{len(names)} ({TAG})",
                         "60/20/20 stratified, seed 42",
                         "QuantileTransformer(n_quantiles=1000, normal)",
                         "scaler fitted on train only"]}).to_csv(f"{R}/preprocessing_steps.csv", index=False)
print(cls_tbl, "\n", split_tbl)

# EDA figures (only once, on the raw features)
if args.features == "geo":
    df = pd.DataFrame(X[:, :8], columns=X_raw_names); df["Price>=Median"] = y
    plt.figure(figsize=(5, 4)); c = np.bincount(y)
    plt.bar(["Below median (0)", "Above median (1)"], c, color=["#4c72b0", "#dd8452"])
    for i, v in enumerate(c): plt.text(i, v + 100, str(v), ha="center")
    plt.title("Class Distribution"); plt.savefig(f"{FG}/01_class_distribution.png", dpi=200, bbox_inches="tight"); plt.close()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.flatten(), X_raw_names):
        hi = df[f].quantile(0.99)
        for cl, col in [(0, "#4c72b0"), (1, "#dd8452")]:
            ax.hist(df.loc[df["Price>=Median"] == cl, f].clip(upper=hi), bins=40, alpha=.6, color=col, label=f"class {cl}")
        ax.set_title(f)
    axes[0, 0].legend(); plt.tight_layout(); plt.savefig(f"{FG}/02_feature_distributions.png", dpi=200); plt.close()
    plt.figure(figsize=(9, 7)); sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap"); plt.savefig(f"{FG}/03_correlation_heatmap.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------------- tuning ----------------
space = dict(lr=[1e-3, 2e-3, 4e-3], wd=[1e-4, 1e-3, 1e-2], hidden=[64, 128, 256],
             n_layers=[2, 3], dropout=[0.0, 0.1, 0.2], d_embed=[8, 16], sigma=[0.05, 0.1, 0.3])
rng = random.Random(SEED); trials, best_params, best_val = [], None, np.inf
print("\nHYPERPARAMETER TUNING")
for t in range(N_TRIALS):
    prm = {k: rng.choice(v) for k, v in space.items()}
    _, vl, h = train_model(prm, Xs_tr, ytr, Xs_va, yva, epochs=TUNE_EP, patience=5)
    trials.append({**prm, "val_loss": vl, "val_acc": max(h["val_acc"]), "seconds": round(h["train_seconds"], 1)})
    print(f"Trial {t+1}/{N_TRIALS} val_loss={vl:.4f}", prm, flush=True)
    if vl < best_val: best_val, best_params = vl, prm
pd.DataFrame(trials).sort_values("val_loss").to_csv(f"{R}/tuning_trials_{TAG}.csv", index=False)
json.dump({"space": space, "best": best_params}, open(f"{R}/best_params_{TAG}.json", "w"), indent=2)
print("Best:", best_params)

# ---------------- cross-validation ----------------
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED); cv_scores = []
for fold, (a, b) in enumerate(cv.split(Xtr, ytr), 1):
    sc = pp.QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED).fit(Xtr[a])
    Xa, Xb, ya, yb = train_test_split(sc.transform(Xtr[a]).astype(np.float32), ytr[a],
                                      test_size=0.15, stratify=ytr[a], random_state=SEED)
    m, _, _ = train_model(best_params, Xa, ya, Xb, yb, epochs=CV_EP, patience=5)
    cv_scores.append(((predict_proba(m, sc.transform(Xtr[b]).astype(np.float32)) >= .5) == ytr[b]).mean())
    print(f"CV fold {fold}: {cv_scores[-1]:.4f}", flush=True)
cv_scores = np.array(cv_scores)
pd.DataFrame({"Fold": range(1, CV_FOLDS + 1), "Accuracy": cv_scores}).to_csv(f"{R}/cv_{TAG}.csv", index=False)
print(f"CV mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
plt.figure(figsize=(7, 4.5)); plt.bar([f"Fold {i+1}" for i in range(CV_FOLDS)], cv_scores, color="#55a868")
plt.axhline(cv_scores.mean(), color="red", ls="--", label=f"Mean = {cv_scores.mean():.4f}")
plt.ylim(max(0, cv_scores.min() - .05), 1); plt.legend(); plt.title(f"{CV_FOLDS}-Fold CV Accuracy ({TAG})")
plt.savefig(f"{FG}/cv_accuracy_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------------- final model ----------------
model, _, hist = train_model(best_params, Xs_tr, ytr, Xs_va, yva, epochs=FINAL_EP, patience=10)
import torch
torch.save(model.state_dict(), f"{M}/tabm_{TAG}.pt")
n_params = count_params(model)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(hist["train_loss"], label="Train loss"); ax[0].plot(hist["val_loss"], label="Validation loss")
ax[0].set_title("Training Curves"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("BCE"); ax[0].legend()
ax[1].plot(hist["val_acc"], color="green"); ax[1].set_title("Validation Accuracy per Epoch")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); plt.tight_layout()
plt.savefig(f"{FG}/training_curves_{TAG}.png", dpi=200); plt.close()
pd.DataFrame({k: hist[k] for k in ["train_loss", "val_loss", "val_acc"]}).to_csv(f"{R}/history_{TAG}.csv", index=False)

# ---------------- evaluation ----------------
p_tr, p_va, p_te = (predict_proba(model, a) for a in (Xs_tr, Xs_va, Xs_te))
m_tr, m_va, m_te = (compute_metrics(yy, pp_) for yy, pp_ in [(ytr, p_tr), (yva, p_va), (yte, p_te)])
bv = pd.DataFrame({"Train": m_tr, "Validation": m_va, "Test": m_te}).T
bv.to_csv(f"{R}/train_val_test_metrics_{TAG}.csv")
gap = m_tr["Accuracy"] - m_va["Accuracy"]
verdict = ("Possible overfitting" if gap > .05 else
           "Possible underfitting" if m_tr["Accuracy"] < .75 and m_va["Accuracy"] < .75 else "Good generalization")
print("\n", bv.round(4), "\nVerdict:", verdict)
rep = classification_report(yte, (p_te >= .5).astype(int), output_dict=True)
pd.DataFrame(rep).T.to_csv(f"{R}/classification_report_{TAG}.csv")
print(classification_report(yte, (p_te >= .5).astype(int)))
res = {**m_te, "Parameters": n_params, "TrainSeconds": round(hist["train_seconds"], 1),
       "CV_mean_acc": cv_scores.mean(), "CV_std_acc": cv_scores.std(), "Verdict": verdict, "Features": TAG}
pd.DataFrame([res]).to_csv(f"{R}/results_{TAG}.csv", index=False)

cm = confusion_matrix(yte, (p_te >= .5).astype(int))
plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                        xticklabels=["Below", "Above"], yticklabels=["Below", "Above"])
plt.title(f"Confusion Matrix ({TAG})"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig(f"{FG}/confusion_matrix_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()
pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_csv(f"{R}/confusion_matrix_{TAG}.csv")
fpr, tpr, _ = roc_curve(yte, p_te); auc = roc_auc_score(yte, p_te)
plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=3, label=f"AUC = {auc:.4f}"); plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
plt.savefig(f"{FG}/roc_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()
pr, rc, _ = precision_recall_curve(yte, p_te)
plt.figure(figsize=(7, 6)); plt.plot(rc, pr, lw=3, color="purple", label=f"AP = {average_precision_score(yte, p_te):.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
plt.savefig(f"{FG}/pr_curve_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()
plt.figure(figsize=(8, 5))
plt.hist(p_te[yte == 0], bins=40, alpha=.65, label="Actual: below"); plt.hist(p_te[yte == 1], bins=40, alpha=.65, label="Actual: above")
plt.axvline(.5, color="k", ls="--"); plt.legend(); plt.title("Predicted Probability Distribution")
plt.savefig(f"{FG}/prob_hist_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()

# permutation importance (validation, AUC drop)
base = roc_auc_score(yva, p_va); prng = np.random.default_rng(SEED); imp = []
for j in range(Xs_va.shape[1]):
    d = []
    for _ in range(3):
        Xp = Xs_va.copy(); Xp[:, j] = prng.permutation(Xp[:, j])
        d.append(base - roc_auc_score(yva, predict_proba(model, Xp)))
    imp.append(np.mean(d))
fi = pd.DataFrame({"Feature": names, "Importance": imp}).sort_values("Importance", ascending=False)
fi.to_csv(f"{R}/feature_importance_{TAG}.csv", index=False); print(fi.head(10))
plt.figure(figsize=(10, max(6, .3 * len(fi)))); plt.barh(fi["Feature"][::-1], fi["Importance"][::-1], color="teal")
plt.title("Permutation Importance (AUC drop)"); plt.savefig(f"{FG}/feature_importance_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()

# train/val/test bar + geographic error map
plt.figure(figsize=(6, 5)); s = [m_tr["Accuracy"], m_va["Accuracy"], m_te["Accuracy"]]
plt.bar(["Train", "Validation", "Test"], s, color=["green", "orange", "blue"]); plt.ylim(0, 1)
for i, v in enumerate(s): plt.text(i, v + .01, f"{v:.3f}", ha="center")
plt.title("Bias-Variance Analysis"); plt.savefig(f"{FG}/bias_variance_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()
lat, lon = Xte[:, names.index("Latitude")], Xte[:, names.index("Longitude")]; ok = (p_te >= .5) == yte
plt.figure(figsize=(8, 7)); plt.scatter(lon[ok], lat[ok], s=6, c="#55a868", alpha=.35, label="Correct")
plt.scatter(lon[~ok], lat[~ok], s=10, c="#c44e52", alpha=.9, label="Wrong"); plt.legend(markerscale=2)
plt.title("Test-set errors on the California map"); plt.savefig(f"{FG}/error_map_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close()
print(f"\nParameters: {n_params}  |  Total runtime: {time.time()-t_start:.0f}s")
