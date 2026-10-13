#!/usr/bin/env bash
# One-shot GitHub setup: creates the PUBLIC repo, adds instructor, then creates
# issues + branches + pull requests + merges (so issues/PR/commit history exists).
#
# Needs: git and GitHub CLI (https://cli.github.com), then `gh auth login` once.
# Run from the repo root:  bash resources/github_publish.sh
# Colab/terminal tip: if you don't want gh, do the same steps by hand in the GitHub website.
set -e
read -p "Registration number (e.g. 229301001): " REGNO
read -p "Instructor GitHub username: " INSTR
REPO="MUJ-DS-$REGNO"
ME=$(gh api user -q .login)

git init -q -b main 2>/dev/null || git init -q
git checkout -q -B main
git add README.md requirements.txt .gitignore
git commit -q -m "Initial commit: README, requirements, .gitignore"
if gh repo view "$ME/$REPO" >/dev/null 2>&1; then
  echo "Repo $REPO already exists on GitHub - using it."
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/$ME/$REPO.git"
  git push -u origin main
else
  gh repo create "$REPO" --public --source=. --remote=origin --push
fi
gh api -X PUT "repos/$ME/$REPO/collaborators/$INSTR" -f permission=push >/dev/null \
  && echo "Invited $INSTR as collaborator (they must accept)."

issue() { gh issue create --title "$1" --body "$(printf "%b" "$2")" --assignee "$ME" | grep -o '[0-9]*$'; }

I1=$(issue "Dataset Collection & Preprocessing" "- [ ] Load California Housing\n- [ ] Binary labels (>= median)\n- [ ] Stratified 60/20/20 split\n- [ ] Quantile scaling + geo features")
I2=$(issue "Model Development" "- [ ] PeriodicEmbedding\n- [ ] TabM-mini (k=8 BatchEnsemble)\n- [ ] Training loop with early stopping")
I3=$(issue "Testing & Evaluation" "- [ ] Hyperparameter search + CV\n- [ ] Metrics, confusion matrix, ROC/PR\n- [ ] Bias-variance analysis")
I4=$(issue "Baseline Comparison & Ablation" "- [ ] compare.py (CO4)\n- [ ] ablation.py (CO5)")
I5=$(issue "Documentation" "- [ ] README\n- [ ] References\n- [ ] Notebook, presentation outline")
I6=$(issue "Results & Deployment" "- [ ] Run full pipeline\n- [ ] Commit results, figures, saved model\n- [ ] Demo via src/test.py")

ship() {  # branch, issue#, commit message, paths...
  local br=$1 is=$2 msg=$3; shift 3
  git checkout -q -b "$br"
  git add "$@"
  git commit -q -m "$msg (closes #$is)"
  git push -q -u origin "$br"
  gh pr create --base main --head "$br" --title "$msg" --body "Closes #$is"
  gh pr merge "$br" --merge --delete-branch
  git checkout -q main && git pull -q origin main
}
ship feature/preprocessing   "$I1" "Add data loading, geo features, split and scaling" src/preprocessing.py data
ship feature/model           "$I2" "Add TabM-mini model and training loop"            src/model.py src/trainer.py
ship feature/train-evaluate  "$I3" "Add training pipeline, evaluation and test script" src/train.py src/test.py src/make_tables.py
ship feature/compare-ablate  "$I4" "Add baseline comparison and ablation study"        src/compare.py src/ablation.py
ship docs/documentation      "$I5" "Add documentation, notebook and presentation outline" assignments notebooks code resources presentations capstone models/model_description.txt
echo
echo "Done. Now run the pipeline (bash code/run_all.sh), fill README placeholders, then:"
echo "  bash resources/github_push_results.sh $I6"
echo "Repo: https://github.com/$ME/$REPO"
