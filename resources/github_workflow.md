# GitHub workflow checklist (from the Batch F process)
1. Accept org invite; complete profile.
2. Create PUBLIC repo `MUJ-DS-<RegNo>`; add instructor as collaborator (never remove).
3. Push this folder (README.md, assignments/, notebooks/, code/, resources/, presentations/, capstone/ + src/, results/, figures/ ...).
4. Capstone team repo: add all members + instructor.
5. Create issues & assign: Dataset Collection, Preprocessing, Model Development, Testing, Documentation, Deployment.
6. Work on branches -> pull request -> review -> merge. Commit regularly.

## Commands
```bash
git init && git branch -M main
git remote add origin https://github.com/<USERNAME>/MUJ-DS-<RegNo>.git
git add . && git commit -m "Initial project structure and README"
git push -u origin main

# per feature
git checkout -b feature/geo-features
git add src/preprocessing.py && git commit -m "Add geo feature engineering"
git push -u origin feature/geo-features   # open Pull Request on GitHub, review, merge
```
## Suggested commit/PR history (meaningful, small commits)
- feat: data loading + stratified split
- feat: quantile scaling + distribution tables
- feat: TabM-mini model
- feat: training loop with early stopping
- feat: hyperparameter search + CV
- feat: evaluation metrics and plots
- feat: geo feature engineering (PR)
- feat: ablation study (PR)
- feat: baseline comparison (PR)
- docs: README results + references
