"""Run from repo root before submitting:  python resources/check_submission.py"""
import os, re, sys
ok = True
def need(path):
    global ok
    good = os.path.exists(path); ok &= good
    print(("[OK]   " if good else "[MISSING] ") + path)
print("--- required files / folders ---")
for p in ["README.md", "requirements.txt", "assignments", "notebooks", "code", "resources", "presentations", "capstone",
          "src/train.py", "src/model.py", "src/preprocessing.py", "src/test.py", "data/dataset_information.txt"]:
    need(p)
print("--- results produced by the pipeline ---")
for p in ["results/results_geo.csv", "results/confusion_matrix_geo.csv", "results/class_distribution.csv",
          "results/split_distribution.csv", "results/comparison.csv", "results/ablation.csv",
          "figures/confusion_matrix_geo.png", "figures/training_curves_geo.png", "figures/comparison.png",
          "figures/ablation.png", "models/tabm_geo.pt"]:
    need(p)
print("--- unfilled placeholders in README.md ---")
if os.path.exists("README.md"):
    ph = re.findall(r"`<[^`]+>`", open("README.md").read())
    for p in sorted(set(ph)): print("  TODO:", p)
    ok &= not ph
print("\nREADY TO SUBMIT" if ok else "\nNOT READY - fix the items above")
sys.exit(0 if ok else 1)
