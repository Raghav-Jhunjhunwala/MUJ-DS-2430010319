"""Turns results/*.csv into markdown tables (results/summary.md) to paste into README."""
import glob, os, pandas as pd
out = []
for f in sorted(glob.glob("results/*.csv")):
    if "history" in f: continue
    out.append(f"### {os.path.basename(f)}\n\n" + pd.read_csv(f).round(4).to_markdown(index=False) + "\n")
open("results/summary.md", "w").write("\n".join(out)); print("wrote results/summary.md")
