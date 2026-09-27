#!/usr/bin/env bash
# Publish only to an existing authorized repository; never force or change visibility.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
repository="${1:-werg/schnitz}"
[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'Use owner/repo.' >&2; exit 2; }
command -v gh >/dev/null || { echo 'Install GitHub CLI and authenticate locally.' >&2; exit 1; }
gh auth status
git rev-parse --verify HEAD >/dev/null
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit changes first.' >&2; exit 1; }
gh repo view "$repository" --json name >/dev/null
url="https://github.com/$repository.git"
if git remote get-url origin >/dev/null 2>&1; then
  current="$(git remote get-url origin)"
  [[ "$current" == "$url" || "$current" == "git@github.com:$repository.git" ]] || {
    echo 'Unexpected origin; inspect it manually. No remote changed.' >&2; exit 1;
  }
else
  git remote add origin "$url"
fi
git fetch origin
if git show-ref --verify --quiet refs/remotes/origin/main; then
  git merge-base --is-ancestor origin/main HEAD || {
    echo 'Remote is not an ancestor; reconcile histories explicitly. No force push.' >&2; exit 1;
  }
fi
git push -u origin HEAD:main
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
gh repo view "$repository" --json url,isPrivate,defaultBranchRef
