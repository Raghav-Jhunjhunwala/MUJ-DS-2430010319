"""Training / prediction / metric utilities shared by all scripts."""
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from model import TabMMini

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_proba(model, X, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(xb)).mean(1).cpu().numpy())
    return np.concatenate(out)


def build_model(n_features, params):
    return TabMMini(
        n_features=n_features, k=params.get("k", 8),
        use_embedding=params.get("use_embedding", True),
        d_embed=params.get("d_embed", 8), sigma=params.get("sigma", 0.1),
        hidden=params.get("hidden", 128), n_layers=params.get("n_layers", 3),
        dropout=params.get("dropout", 0.1)).to(device)


def train_model(params, Xtr, ytr, Xva, yva, epochs=60, patience=8, batch_size=1024,
                seed=SEED, max_seconds=None):
    """AdamW + cosine schedule + early stopping on validation BCE."""
    set_seed(seed)
    model = build_model(Xtr.shape[1], params)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
    best_loss, best_state, wait, t0 = np.inf, None, 0, time.time()
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train(); total = 0.0
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb[:, None].expand_as(logits))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        p = torch.tensor(predict_proba(model, Xva), device=device)
        val_loss = F.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), yva_t).item()
        val_acc = ((p > 0.5).float() == yva_t).float().mean().item()
        hist["train_loss"].append(total / len(Xtr)); hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        sched.step()
        if val_loss < best_loss - 1e-5:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if max_seconds and time.time() - t0 > max_seconds:
            break
    model.load_state_dict(best_state)
    hist["train_seconds"] = time.time() - t0
    return model, best_loss, hist


def compute_metrics(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, pred),
        "Macro-F1": f1_score(y_true, pred, average="macro"),
        "AUC": roc_auc_score(y_true, proba),
    }
