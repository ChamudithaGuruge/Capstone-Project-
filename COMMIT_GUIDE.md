# Commit guide

The capstone guide asks for a public repo with **meaningful commits and
meaningful commit messages from the start**, and a viva may check that the
history shows your own incremental work. Do not dump every file in one commit.
Build the history in the order below, one logical change per commit.

## Message format

```
type(scope): short imperative subject (<= 72 chars)

Body: what changed and WHY, wrapped at ~72 cols. Explain the reasoning
or trade-off, not just the file list. Reference a milestone or decision
if relevant.
```

- **type**: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`.
- **scope**: the area touched, e.g. `config`, `features`, `datagen`, `eval`, `infer`.
- Subject in the **imperative mood** ("add", not "added"/"adds"), no trailing period.
- One logical change per commit. Keep `data/` and `out/` out of git (see `.gitignore`).

## Step-by-step commit sequence

Run these from the repo root after `git init` and setting your remote.

```bash
# 1. scaffold
git add .gitignore requirements.txt
git commit -m "chore: scaffold repo with gitignore and dependencies" \
  -m "Pin Spark 2.4.7 to match the cluster; ignore generated data and spark state."

# 2. configuration
git add config.py
git commit -m "feat(config): add central config for tables, windows and seeds" \
  -m "Single place for source names, non-overlapping feature/label windows, target package codes, calibration tau and fixed seeds, so a run is fully reproducible."

# 3. io helpers
git add common.py
git commit -m "feat(io): add Spark session and table IO helpers" \
  -m "Public build reads/writes parquet; one swap point (load/save) switches to warehouse tables in production."

# 4. synthetic data
git add synthetic_data.py
git commit -m "feat(data): add schema-matched synthetic data generator" \
  -m "A latent per-subscriber trait drives both features and outcome so the pipeline runs end to end on synthetic data with a real signal and no company data."

# 5. base features
git add features.py
git commit -m "feat(features): add 43 base demographic and revenue features" \
  -m "Totals, daily averages, recency, frequency, seasonality and categorical indices from profile and prepaid revenue history."

# 6. tiers (same file, second commit) -- stage only if you split the file; otherwise fold into commit 5
git commit --allow-empty -m "feat(features): add VAS, prior-insurance and balance tiers (59 total)" \
  -m "Three bolt-on tiers that lifted gold AUC; VAS excludes the target packages to prevent label leakage."

# 7. datagen
git add datagen.py
git commit -m "feat(datagen): build label, freeze gold ruler and sample negatives" \
  -m "Billing label for positives; 20% of confirmed outcomes frozen as the gold ruler and anti-joined out of training; confirmed decliners plus random fill to a 1:1 balance."

# 8. evaluation
git add test_mode.py
git commit -m "feat(eval): add RF/GBT training and gold-ruler metrics" \
  -m "Reports gold AUC and per-decile lift on the frozen ruler; MODEL env selects GBT (shipped) or RF (baseline). Flags AUC > 0.70 as likely leakage."

# 9. inference
git add infer_mode.py
git commit -m "feat(infer): add calibrated scoring and top-decile export" \
  -m "Scores the live base, calibrates to the true prior with tau=0.014 (monotonic, rank-preserving), keeps decile 1 as the lead list."

# 10. docs
git add README.md COMMIT_GUIDE.md docs/
git commit -m "docs: add README, architecture diagram and reproducibility notes" \
  -m "End-to-end run instructions, the 59-feature summary, the two-exams leakage note and the reproducibility checklist."

git push -u origin main
```

If you prefer to commit `features.py` once, drop step 6 and fold its message
into step 5. Keep the rest of the order: config and IO before the code that uses
them, data before the pipeline, pipeline before the docs that describe it.

## What a good history looks like to an examiner

- Commits are small, ordered and each builds on the last.
- Messages say **why**, not just **what**.
- No single "add everything" commit, no `data/` or `out/` blobs, no secrets.
- Later fixes are their own commits (`fix(features): correct recency sentinel`),
  so the history shows real iteration.
