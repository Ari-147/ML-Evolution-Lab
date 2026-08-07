import numpy as np
import plotly.graph_objects as go
import streamlit as st

import storage.db as db
import snapshotting.logger as logger
from analysis.error_clusters import compute_error_clusters

st.set_page_config(page_title="ML Evolution Lab", layout="wide")
st.title("ML Evolution Lab")

runs = db.list_runs()
if not runs:
    st.warning("No runs found yet. Run `python main.py train standard mlp` or `python main.py train standard random_forest` first.")
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

step = st.select_slider("Step", options=steps, value=steps[0])
snapshot_id, step, train_accuracy, val_accuracy, npz_path, phase_id = snapshot_by_step[step]

try:
    step_data = logger.load_snapshot(npz_path)
except FileNotFoundError:
    st.error(f"Snapshot array file not found at path: `{npz_path}`")
    st.stop()

col_train, col_val = st.columns(2)
col_train.metric("Train accuracy", f"{train_accuracy:.4f}")
col_val.metric("Val accuracy", f"{val_accuracy:.4f}")

st.subheader("Feature importance")
feature_importances = step_data.get("feature_importances")
if feature_importances is None:
    st.info("This run does not have feature-importance data yet.")
else:
    feature_names = [f"Feature {i + 1}" for i in range(len(feature_importances))]
    fig_importance = go.Figure(go.Bar(x=feature_names, y=feature_importances))
    fig_importance.update_layout(
        xaxis_title="feature",
        yaxis_title="importance",
        height=350,
        margin=dict(t=20),
    )
    st.plotly_chart(fig_importance, width="stretch")

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
    error_cluster_labels = step_data.get("error_cluster_labels")
    if error_cluster_labels is None:
        error_cluster_labels = compute_error_clusters(
            val_2d,
            val_true_labels,
            step_data["predictions"],
        )

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
    error_mask = step_data["predictions"] != val_true_labels
    misclassified_points = val_2d[error_mask]
    misclassified_cluster_ids = error_cluster_labels[error_mask]
    fig.add_trace(
        go.Scatter(
            x=misclassified_points[:, 0],
            y=misclassified_points[:, 1],
            mode="markers",
            marker=dict(
                color=misclassified_cluster_ids,
                colorscale="Viridis",
                size=8,
                line=dict(width=0.5, color="black"),
            ),
            name="misclassified points",
        )
    )
    fig.update_layout(
        xaxis_title="PC1", yaxis_title="PC2", height=500, margin=dict(t=20)
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Error cluster trend")
    cluster_counts = []
    trend_steps = []
    for _, step_value, _, _, npz_path, _ in snapshots:
        try:
            snapshot_values = logger.load_snapshot(npz_path)
        except FileNotFoundError:
            continue
        labels = snapshot_values.get("error_cluster_labels")
        if labels is None:
            continue
        non_negative = labels[labels >= 0]
        cluster_counts.append(len(np.unique(non_negative)))
        trend_steps.append(step_value)
    if trend_steps:
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=trend_steps, y=cluster_counts, mode="lines+markers"))
        trend_fig.update_layout(
            xaxis_title="step",
            yaxis_title="distinct error clusters",
            height=300,
            margin=dict(t=20),
        )
        st.plotly_chart(trend_fig, width="stretch")

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