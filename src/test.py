"""Reload the saved model and re-evaluate on the test split (reproducibility check).
Usage: python src/test.py --features geo"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
import preprocessing as pp
from trainer import build_model, predict_proba, compute_metrics, device

ap = argparse.ArgumentParser(); ap.add_argument("--features", default="geo"); a = ap.parse_args()
X, y, names = pp.load_data()
if a.features == "geo": X, names = pp.add_geo_features(X, names)
Xtr, Xva, Xte, ytr, yva, yte = pp.split_data(X, y)
(_, _, Xs_te), _ = pp.quantile_scale(Xtr, Xva, Xte)
prm = json.load(open(f"results/best_params_{a.features}.json"))["best"]
m = build_model(Xs_te.shape[1], prm); m.load_state_dict(torch.load(f"models/tabm_{a.features}.pt", map_location=device))
for k, v in compute_metrics(yte, predict_proba(m, Xs_te)).items(): print(f"{k:12s}: {v:.4f}")
