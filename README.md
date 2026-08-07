# ML Evolution Lab

ML Evolution Lab is a local training instrumentation platform for machine learning models that snapshots the full learning trajectory — not just final metrics. It captures per-step model behavior, including predictions, confidence, decision boundaries, feature importance, error clusters, forgetting events, and subgroup fairness metrics, then presents them in a synchronized Streamlit dashboard.

## Why this matters

This project is designed to make model training explorable and transparent. Instead of only comparing final accuracy, you can watch how a model learns over time, see where it becomes confident or confused, and discover hidden training dynamics such as forgetting and fairness drift.

## Key features

- Stepwise snapshot logging for each training step or epoch
- Decision boundary visualization via PCA-projected 2D approximation
- Feature importance tracking for interpretability
- Error cluster analysis for understanding misclassified regions
- Catastrophic forgetting visualization in continual learning
- Subgroup fairness metrics for bias emergence analysis
- Streamlit dashboard for interactive run inspection

## Presets

- **Standard**: Compare `mlp` and `random_forest` learning on a synthetic classification dataset.
- **Continual**: Train an `mlp` on sequential phases and visualize forgetting across tasks.
- **Fairness**: Train on a real subgroup-labeled dataset and track subgroup accuracy, confidence, and positive rate.

## Tech stack

- Python 3.11+
- PyTorch for the MLP training path
- scikit-learn for Random Forest, PCA, and dataset generation
- SQLite via `sqlite3` for run and snapshot metadata
- NumPy for snapshot storage and array handling
- Streamlit + Plotly for the interactive dashboard
- pandas for fairness dataset handling

## Setup

1. Create a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the `data/` directory exists and is writable. The project stores snapshot files under `data/snapshots` and the SQLite database at `data/ml_lab.db`.

## Training

Use `main.py` to run training for a preset and model family.

```bash
python main.py train standard mlp
python main.py train standard random_forest
python main.py train continual mlp
python main.py train fairness mlp
python main.py train fairness random_forest
```

### Notes

- `standard` supports both `mlp` and `random_forest`
- `continual` only supports `mlp`
- `fairness` supports both `mlp` and `random_forest`

## Dashboard

Start the interactive dashboard with Streamlit:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

## Project layout

```text
ML-Evolution-Lab/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── config.py
├── main.py
├── app.py
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
├── data/
│   ├── openml/  # cached dataset downloads
│   └── snapshots/  # saved snapshot runs
└── tests/
    ├── test_error_clusters.py
    ├── test_feature_importance.py
    ├── test_forgetting.py
    └── test_mlp_harness.py
```

## Configuration

All runtime constants are controlled from `config.py`:

- `MAX_EPOCHS`: maximum MLP epochs
- `MAX_TREES`: maximum Random Forest trees
- `TREES_PER_STEP`: tree growth step size
- `SNAPSHOT_DENSE_STEPS`: dense snapshot logging early in training
- `SNAPSHOT_EVERY_N`: sparser logging later in training
- `PCA_COMPONENTS`: PCA projection dimensions (2)
- `GRID_RESOLUTION`: decision boundary grid resolution
- `ERROR_CLUSTER_K_RANGE`: cluster count search range
- `PERMUTATION_N_REPEATS`: repeats used for permutation importance

## Data storage

- Runs and snapshots are registered in `data/ml_lab.db`
- Snapshot artifacts are stored as `.npz` files in `data/snapshots/<run_id>/`
- OpenML datasets are cached under `data/openml/`

## Recommended workflow

1. Train a run with `python main.py train <preset> <model_family>`
2. Open `streamlit run app.py`
3. Select the desired run and step in the dashboard
4. Explore decision boundaries, confidence, feature importance, and error trends

## Testing

Run the existing test suite with:

```bash
pytest
```

## Contact

This repository is designed as a training visualization tool for model debugging, interpretability, and fairness analysis. If you want to extend it, the code is organized by preset, training harness, snapshotting, and analysis modules.
