"""CO4: compare the proposed model with other methods on the SAME dataset, SAME split.
Run AFTER train.py."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import preprocessing as pp
from model import count_params
from trainer import train_model, predict_proba, compute_metrics

X, y, names = pp.load_data()
Xg, _ = pp.add_geo_features(X, names)
rows = []

def add(name, proba, params, secs, feats):
    m = compute_metrics(yte, proba); m.update({"Model": name, "Parameters": params, "Features": feats, "Seconds": round(secs, 1)})
    rows.append(m); print(name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)

def sk_models():
    out = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "Random Forest (300 trees)": RandomForestClassifier(300, n_jobs=-1, random_state=42),
        "HistGradientBoosting (GBDT)": HistGradientBoostingClassifier(random_state=42),
        "sklearn MLP (128-64)": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=300, early_stopping=True, random_state=42)),
    }
    try:
        from xgboost import XGBClassifier
        out["XGBoost"] = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=42, eval_metric="logloss")
    except ImportError:
        pass
    return out

def n_params(name, mdl, nfeat):
    if name.startswith("Logistic"): return nfeat + 1
    if name.startswith("sklearn MLP"):
        c, b = mdl[-1].coefs_, mdl[-1].intercepts_
        return sum(w.size for w in c) + sum(v.size for v in b)
    return None  # tree ensembles: not parameter-counted

for feat_name, XX in [("base", X), ("geo", Xg)]:
    Xtr, Xva, Xte, ytr, yva, yte = pp.split_data(XX, y)
    for name, mdl in sk_models().items():
        if feat_name == "geo" and name not in ("HistGradientBoosting (GBDT)", "XGBoost"):
            continue          # only give the strongest baselines the geo features (fairness check)
        t = time.time(); mdl.fit(Xtr, ytr)
        add(name + (" + geo feats" if feat_name == "geo" else ""), mdl.predict_proba(Xte)[:, 1],
            n_params(name, mdl, XX.shape[1]), time.time() - t, feat_name)
    # deep models
    (a, b, c), _ = pp.quantile_scale(Xtr, Xva, Xte)
    pf = "results/best_params_%s.json" % feat_name
    prm = json.load(open(pf))["best"] if os.path.exists(pf) else dict(lr=2e-3, wd=1e-3, hidden=128, n_layers=3, dropout=0.1, d_embed=8, sigma=0.1)
    if feat_name == "base":
        m, _, h = train_model({**prm, "k": 1, "use_embedding": False}, a, ytr, b, yva, epochs=60, patience=10)
        add("PyTorch MLP", predict_proba(m, c), count_params(m), h["train_seconds"], "base")
        m, _, h = train_model({**prm, "k": 8}, a, ytr, b, yva, epochs=60, patience=10)
        add("TabM-mini (original)", predict_proba(m, c), count_params(m), h["train_seconds"], "base")
    else:
        m, _, h = train_model({**prm, "k": 8}, a, ytr, b, yva, epochs=60, patience=10)
        add("Proposed: TabM-mini + geo feats", predict_proba(m, c), count_params(m), h["train_seconds"], "geo")

df = pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Parameters", "Features", "Seconds"]]
df = df.sort_values("AUC", ascending=False)
df.to_csv("results/comparison.csv", index=False); print(df.round(4).to_string(index=False))
plt.figure(figsize=(10, 6)); d = df.sort_values("Accuracy")
plt.barh(d["Model"], d["Accuracy"], color=["orange" if "Proposed" in m else "gray" for m in d["Model"]])
plt.xlim(d["Accuracy"].min() - .01, d["Accuracy"].max() + .01); plt.xlabel("Test accuracy")
plt.title("Comparison on the same test split"); plt.savefig("figures/comparison.png", dpi=200, bbox_inches="tight")
