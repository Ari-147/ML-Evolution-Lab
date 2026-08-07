import numpy as np
import plotly.graph_objects as go
import streamlit as st

import storage.db as db
import snapshotting.logger as logger

st.set_page_config(page_title="ML Evolution Lab", layout="wide")
st.title("ML Evolution Lab")

# 1. Fetch available runs from DB
runs = db.list_runs()
if not runs:
    st.warning("No runs found yet. Run `python main.py train standard mlp` first.")
    st.stop()

run_labels = {
    f"{mode} / {model_family} — {dataset_name} — {created_at}": run_id
    for run_id, model_family, mode, dataset_name, created_at in runs
}

selected_label = st.selectbox("Run", list(run_labels.keys()))
run_id = run_labels[selected_label]

# 2. Try loading metadata safely
try:
    meta = logger.load_run_meta(run_id)
except FileNotFoundError:
    st.error(f"Snapshot files missing for run ID `{run_id}`. Please select another run or re-run training.")
    st.stop()

# 3. Load snapshots for selected run
snapshots = db.list_snapshots(run_id)
if not snapshots:
    st.warning("This run has no snapshots recorded yet.")
    st.stop()

steps = [row[1] for row in snapshots]
snapshot_by_step = {row[1]: row for row in snapshots}

# Time scrubber slider
step = st.select_slider("Step (epoch)", options=steps, value=steps[0])
snapshot_id, step, train_accuracy, val_accuracy, npz_path, phase_id = snapshot_by_step[step]

try:
    step_data = logger.load_snapshot(npz_path)
except FileNotFoundError:
    st.error(f"Snapshot array file not found at path: `{npz_path}`")
    st.stop()

# Metrics
col_train, col_val = st.columns(2)
col_train.metric("Train accuracy", f"{train_accuracy:.4f}")
col_val.metric("Val accuracy", f"{val_accuracy:.4f}")

col_boundary, col_conf = st.columns(2)

# Panel 1: Decision Boundary
with col_boundary:
    st.subheader("Decision boundary")
    st.caption(
        "⚠️ Approximation: features are projected to 2D via PCA (fit once "
        "for this run and reused across every step). The boundary comes "
        "from evaluating the model on grid points inverse-transformed "
        "back to original feature space — for datasets with more than "
        "2 features this is a comparable slice, not the true boundary."
    )

    xx = meta["grid_xx"]
    yy = meta["grid_yy"]
    boundary_grid = step_data["boundary_grid"]
    val_2d = meta["val_2d"]
    val_true_labels = meta["val_true_labels"]

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=boundary_grid,
            showscale=False,
            opacity=0.5,
            colorscale="RdBu",
            contours=dict(showlines=False),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=val_2d[:, 0],
            y=val_2d[:, 1],
            mode="markers",
            marker=dict(
                color=val_true_labels,
                colorscale="RdBu",
                size=6,
                line=dict(width=0.5, color="black"),
            ),
            name="val points (true label)",
        )
    )
    fig.update_layout(
        xaxis_title="PC1", yaxis_title="PC2", height=500, margin=dict(t=20)
    )
    st.plotly_chart(fig, width="stretch")

# Panel 2: Confidence Distribution
with col_conf:
    st.subheader("Confidence distribution")
    confidences = step_data["confidences"]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=confidences, nbinsx=30))
    fig2.update_layout(
        xaxis_title="confidence (softmax max prob)",
        yaxis_title="count",
        height=500,
        margin=dict(t=20),
    )
    st.plotly_chart(fig2, width="stretch")