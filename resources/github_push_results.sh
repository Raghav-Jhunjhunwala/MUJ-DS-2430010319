#!/usr/bin/env bash
# After running the pipeline: pushes results/figures/models + final README via a PR.
# Usage: bash resources/github_push_results.sh <results-issue-number>
set -e
IS=${1:?give the issue number printed by github_publish.sh}
git checkout -q -b results/final
git add -A
git commit -q -m "Add results, figures, trained model and final README (closes #$IS)"
git push -q -u origin results/final
gh pr create --base main --head results/final --title "Add results and final README" --body "Closes #$IS"
gh pr merge results/final --merge --delete-branch
git checkout -q main && git pull -q origin main
echo "Pushed. Check the repo page, then submit the URL."
