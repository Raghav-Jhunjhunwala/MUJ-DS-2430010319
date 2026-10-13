"""CO5 ablation: contribution of each component, mean +- std over several seeds.
Run AFTER train.py (re-uses the tuned hyper-parameters)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocessing as pp
from model import count_params
from trainer import train_model, predict_proba, compute_metrics

SEEDS = [0, 1, 2]
EPOCHS = 60
P = "results/best_params_geo.json"
base = json.load(open(P))["best"] if os.path.exists(P) else dict(lr=2e-3, wd=1e-3, hidden=128, n_layers=3, dropout=0.1, d_embed=8, sigma=0.1)

X, y, names = pp.load_data()
Xg, names_g = pp.add_geo_features(X, names)
variants = [  # name, features, k, periodic embedding
    ("Baseline MLP",                       "base", 1, False),
    ("+ Periodic embeddings",              "base", 1, True),
    ("+ BatchEnsemble k=8 (TabM-mini)",    "base", 8, True),
    ("+ Geo features (PROPOSED)",          "geo",  8, True),
    ("MLP + Geo features only",            "geo",  1, False),
]
rows = []
for name, feat, k, emb in variants:
    XX = X if feat == "base" else Xg
    Xtr, Xva, Xte, ytr, yva, yte = pp.split_data(XX, y)
    (a, b, c), _ = pp.quantile_scale(Xtr, Xva, Xte)
    ms = []
    for s in SEEDS:
        m, _, _ = train_model({**base, "k": k, "use_embedding": emb}, a, ytr, b, yva, epochs=EPOCHS, patience=10, seed=s)
        ms.append(compute_metrics(yte, predict_proba(m, c)))
    df = pd.DataFrame(ms)
    row = {"Experiment": name, "Geo features": "Y" if feat == "geo" else "N", "Periodic emb.": "Y" if emb else "N",
           "Ensemble k": k, "Params": count_params(m)}
    for col in ["Accuracy", "F1", "AUC"]:
        row[col] = f"{df[col].mean():.4f} ± {df[col].std():.4f}"
        row[col + "_mean"] = df[col].mean()
    rows.append(row); print(row, flush=True)
out = pd.DataFrame(rows); os.makedirs("results", exist_ok=True)
out.to_csv("results/ablation.csv", index=False)
plt.figure(figsize=(9, 5))
plt.barh(out["Experiment"][::-1], out["Accuracy_mean"][::-1], color="teal")
plt.xlim(out["Accuracy_mean"].min() - .01, out["Accuracy_mean"].max() + .01)
plt.xlabel("Test accuracy (mean over seeds)"); plt.title("Ablation study")
plt.savefig("figures/ablation.png", dpi=200, bbox_inches="tight")
