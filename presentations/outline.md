# Presentation outline (10–12 slides, ~8 min + demo)
1. Title / name / reg no
2. Problem & dataset (20,640 districts, binary task, class balance)
3. Preprocessing & split (table)
4. Model: TabM-mini diagram (embedding -> k BatchEnsemble MLPs -> average)
5. Training & tuning (search space, curves)
6. Results (metrics table, confusion matrix, ROC)
7. Bias–variance & CV
8. Comparison with baselines (CO4 table/figure) + references
9. Innovation: geo features (why) 
10. Ablation (CO5) – what helped, what didn't
11. Error analysis (error map), limitations
12. Conclusion, GitHub repo link, live demo: `python src/test.py`
