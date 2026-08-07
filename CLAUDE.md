# Project: ML Evolution Lab

## What this is
A training instrumentation platform that snapshots a model's behavior at
every epoch (or every N trees, for Random Forest) — not just loss/accuracy,
but per-example predictions, confidence, decision boundary, feature
importance, error clusters, and (in the relevant presets) forgetting
events and subgroup fairness metrics. A Streamlit dashboard scrubs through
training via a slider, with every panel synced to it — watching a model
learn rather than reading a final chart.

## Three presets
- Standard: MLP vs Random Forest on the same synthetic dataset, watch
  both learn side by side.
- Continual: sequential class-phase training on one model (MLP only),
  designed to induce and visualize catastrophic forgetting.
- Fairness: training on a real subgroup-labeled dataset (UCI Adult
  Income), watching per-subgroup accuracy/positive-rate diverge over
  training — "bias emergence".

## Rules for the agent
- Build only what's asked in the current phase. No extra features or
  polish beyond scope unless asked.
- Prefer editing existing files over rewriting from scratch.
- Ask before adding a new dependency not listed below.
- Keep functions small and single-purpose; no premature abstraction.
- Concise code, minimal comments.
- Use config.py for all tunable values (epoch caps, PCA components, grid
  resolution, thresholds, paths). No magic numbers scattered in code.
- No Anthropic/OpenAI API calls anywhere in this project — everything
  runs locally and should cost $0.
- Decision boundaries are computed by projecting features to 2D via PCA
  (fit once per run, fixed across all steps so movement is comparable),
  building a grid in that 2D space, and inverse-transforming grid points
  back to original feature space for model evaluation. This is an
  approximation for any dataset with more than 2 features — the UI must
  say so, not present it as ground truth.
- Continual mode is MLP-only. Sequential warm_start fine-tuning on a
  Random Forest is a murkier story than on a neural net and isn't worth
  the complexity for this project.
- In Fairness mode, the subgroup column (e.g. sex) is excluded from model
  features by default — the interesting finding is whether bias emerges
  even when the model never sees the subgroup directly.
- Keep snapshots small: dense logging early (config.SNAPSHOT_DENSE_STEPS),
  sparser logging after (config.SNAPSHOT_EVERY_N) — this is what keeps
  dashboard scrubbing feel instant instead of janky.

## Tech stack (fixed — do not substitute)
- Python 3.11+
- PyTorch for the MLP path
- scikit-learn for the Random Forest path (warm_start=True), PCA, and
  permutation importance
- SQLite (stdlib sqlite3) for run/snapshot metadata
- Per-step .npz files for raw arrays (predictions, confidences, feature
  importances, boundary grids) — lazy-loaded, not held in memory at once
- Streamlit + Plotly for the dashboard
- pandas for the Fairness preset's dataset (fetched via
  sklearn.datasets.fetch_openml, cached locally after first run)

## Folder structure
ml-evolution-lab/
├── CLAUDE.md
├── requirements.txt
├── config.py
├── main.py                    CLI: train <mode> <model_family>
├── presets/
│   ├── standard.py
│   ├── continual.py
│   └── fairness.py
├── training/
│   ├── mlp_harness.py
│   └── rf_harness.py
├── snapshotting/
│   ├── logger.py
│   └── boundary.py
├── analysis/
│   ├── feature_importance.py
│   ├── error_clusters.py
│   ├── forgetting.py
│   └── fairness_metrics.py
├── storage/
│   └── db.py
├── app.py                     Streamlit dashboard
├── data/                      gitignored: datasets, snapshots/*.npz, ml_lab.db
└── tests/

## Build phases (one phase per session, in order)
1. Training harness + snapshot logger for the MLP path (Standard preset
   dataset), per-epoch predictions/confidence/accuracy
2. Decision boundary + confidence evolution, first scrubber UI
3. Feature importance evolution + Random Forest path — completes
   Standard preset ("watch MLP vs Random Forest learn")
4. Error clusters, tracked and plotted per snapshot step
5. Forgotten-example tracking + Continual preset — completes "watch
   catastrophic forgetting happen"
6. Subgroup fairness metrics + Fairness preset — completes "watch bias
   emerge over time"
7. Unified dashboard across all three presets + diff view between two
   steps or two runs (the "GitHub for model training" moment)

## Data contracts (keep stable across phases)
- Run: {run_id, model_family ("mlp"|"random_forest"), mode
  ("standard"|"continual"|"fairness"), dataset_name, created_at,
  config_snapshot}
- Snapshot: {snapshot_id, run_id, step, train_accuracy, val_accuracy,
  timestamp, npz_path, phase_id}   (phase_id null outside continual mode)
- Snapshot .npz contents: predictions, confidences, feature_importances
  (if computed), boundary_grid (if computed)
- ForgettingEvent: {run_id, example_id, step, transition
  ("learned"|"forgotten")}
- ErrorCluster: {run_id, step, cluster_id, member_example_ids[]}
- SubgroupMetric: {run_id, step, subgroup_value, accuracy,
  avg_confidence, positive_rate}

## Config values (config.py, all tunable)
- MAX_EPOCHS (default 50), MAX_TREES (default 100), TREES_PER_STEP
  (default 5)
- SNAPSHOT_DENSE_STEPS (log every step up to this many, default 20),
  SNAPSHOT_EVERY_N (beyond that, log every Nth step, default 5)
- PCA_COMPONENTS (2), GRID_RESOLUTION (default 100)
- ERROR_CLUSTER_K_RANGE (default 2-6)
- DATA_DIR, DB_PATH
