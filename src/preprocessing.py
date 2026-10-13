"""Data loading, feature engineering, splitting and scaling."""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

SEED = 42

# (lat, lon) of major population centres, used for the "geo" features
CITIES = {
    "LosAngeles": (34.05, -118.24), "SanFrancisco": (37.77, -122.42),
    "SanDiego": (32.72, -117.16), "SanJose": (37.34, -121.89),
    "Sacramento": (38.58, -121.49),
}


def load_data():
    """California Housing -> binary task: median house value >= dataset median."""
    h = fetch_california_housing()
    X = h.data.astype(np.float32)
    y = h.target
    y_class = (y >= np.median(y)).astype(int)
    return X, y_class, list(h.feature_names)


def add_geo_features(X, names):
    """INNOVATION (preprocessing): ratio + geographic features.

    - rooms/bedrooms ratio
    - lat/lon rotated by 15/30/45 degrees (lets axis-aligned splits and MLPs
      capture diagonal coastlines)
    - distance (degrees, lon scaled by cos(lat)) to 5 big cities + nearest city
    """
    lat = X[:, names.index("Latitude")]
    lon = X[:, names.index("Longitude")]
    rooms = X[:, names.index("AveRooms")]
    beds = X[:, names.index("AveBedrms")]
    extra, extra_names = [rooms / np.maximum(beds, 1e-3)], ["RoomsPerBedroom"]
    for deg in (15, 30, 45):
        th = np.deg2rad(deg)
        extra.append(lon * np.cos(th) + lat * np.sin(th)); extra_names.append(f"Rot{deg}_a")
        extra.append(lat * np.cos(th) - lon * np.sin(th)); extra_names.append(f"Rot{deg}_b")
    dists = []
    for city, (clat, clon) in CITIES.items():
        d = np.sqrt((lat - clat) ** 2 + ((lon - clon) * np.cos(np.deg2rad(lat))) ** 2)
        dists.append(d); extra.append(d); extra_names.append(f"Dist_{city}")
    extra.append(np.min(dists, axis=0)); extra_names.append("Dist_NearestCity")
    X_new = np.column_stack([X] + extra).astype(np.float32)
    return X_new, names + extra_names


def split_data(X, y, seed=SEED):
    """60 / 20 / 20 stratified train / validation / test split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def quantile_scale(X_tr, *others, seed=SEED):
    """Quantile -> Normal scaling, fitted on the training set only (no leakage)."""
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=seed)
    out = [qt.fit_transform(X_tr).astype(np.float32)]
    out += [qt.transform(o).astype(np.float32) for o in others]
    return out, qt


def distribution_tables(y, y_tr, y_va, y_te):
    """Dataset / split / imbalance tables required by the guidelines (Section 5A)."""
    n = len(y)
    cls = pd.DataFrame({"Class": ["0 (below median)", "1 (above median)"],
                        "Samples": np.bincount(y),
                        "Percent": np.round(100 * np.bincount(y) / n, 2)})
    cls["Imbalance ratio (majority/minority)"] = round(max(np.bincount(y)) / min(np.bincount(y)), 3)
    rows = []
    for name, yy in [("Train", y_tr), ("Validation", y_va), ("Test", y_te)]:
        c = np.bincount(yy, minlength=2)
        rows.append({"Split": name, "Samples": len(yy), "Percent": round(100 * len(yy) / n, 2),
                     "Class0": c[0], "Class1": c[1]})
    return cls, pd.DataFrame(rows)
