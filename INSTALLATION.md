# Installation Guide

Project: **TabM-mini with Geographic Feature Engineering on California Housing**
Author: Raghav Jhunjhunwala (Reg. No. 2430010319), Batch D

## 1. Requirements

| Item | Needed |
|---|---|
| Python | 3.10 or newer |
| Internet | First run only (the dataset is downloaded automatically by scikit-learn) |
| GPU | Optional. A full run works on CPU; a GPU (e.g. Colab) is faster |
| Disk space | Under 1 GB |

## 2. Option A: Run on your own computer

```bash
# 1. Clone the repository
git clone https://github.com/Raghav-Jhunjhunwala/MUJ-DS-2430010319.git
cd MUJ-DS-2430010319

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Check the install
python -c "import torch, sklearn, pandas; print('OK', torch.__version__)"
```

## 3. Option B: Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook (or open `notebooks/experiments.ipynb`).
2. Optional: choose **Runtime → Change runtime type → GPU**.
3. Run these cells:

```python
!git clone https://github.com/Raghav-Jhunjhunwala/MUJ-DS-2430010319.git
%cd MUJ-DS-2430010319
!pip -q install -r requirements.txt
!python src/train.py --features geo
```

## 4. Running the project

Always run commands from the **repository root** (the folder that contains `src/`).

| Goal | Command | Approx. time |
|---|---|---|
| Quick test run | `python src/train.py --features geo --fast` | ~2 min |
| Proposed model (full) | `python src/train.py --features geo` | a few minutes |
| Original TabM-mini (needed by compare.py) | `python src/train.py --features base` | a few minutes |
| Ablation study | `python src/ablation.py` | a few minutes |
| Baseline comparison | `python src/compare.py` | a few minutes |
| Reload saved model and re-test | `python src/test.py --features geo` | seconds |
| Build summary tables | `python src/make_tables.py` | seconds |
| Everything in order | `bash code/run_all.sh` | longest |

**Order matters.** `ablation.py`, `compare.py` and `test.py` read files created by `train.py`
(`results/best_params_*.json` and `models/tabm_*.pt`), so run `train.py` first
(both `--features base` and `--features geo` for `compare.py`).

## 5. Where the outputs go

| Folder | Contents |
|---|---|
| `results/` | Metrics and tables (CSV/JSON), `summary.md` |
| `figures/` | Plots (confusion matrix, ROC, training curves, comparison, ablation ...) |
| `models/` | Saved weights (`tabm_geo.pt`, `tabm_base.pt`) |

These folders are created automatically if they do not exist.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| File-not-found errors for `results/` or `models/` | You are not in the repo root. Run `cd MUJ-DS-2430010319` first. |
| `FileNotFoundError: results/best_params_geo.json` | Run `python src/train.py --features geo` before `test.py`, `ablation.py` or `compare.py`. |
| Dataset download fails | Check your internet connection and run again. |
| `xgboost` not installed | It is optional. Run `pip install xgboost`, or `compare.py` will skip XGBoost. |
| Training is slow | Use `--fast`, or switch to a Colab GPU runtime. |
| `pip install -r requirements.txt` fails on Windows | Upgrade pip first: `python -m pip install --upgrade pip`. |

Results can differ from the README by a few thousandths between runs and machines.
All experiments use seed 42.
