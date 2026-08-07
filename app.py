import numpy as np
import plotly.graph_objects as go
import streamlit as st

import storage.db as db
import snapshotting.logger as logger
from analysis.error_clusters import compute_error_clusters

st.set_page_config(page_title="ML Evolution Lab", layout="wide")
st.title("ML Evolution Lab")

# 1. Fetch available runs
runs = db.list_runs()
if not runs:
    st.warning("No runs found yet. Run `python main.py train standard mlp`, `python main.py train standard random_forest`, or `python main.py train continual mlp` first.")
    st.stop()

run_labels = {
    f"{mode} / {model_family} — {dataset_name} — {created_at}": run_id
    for run_id, model_family, mode, dataset_name, created_at in runs
}

selected_label = st.selectbox("Run", list(run_labels.keys()))
run_id = run_labels[selected_label]
selected_run = next(row for row in runs if row[0] == run_id)
mode = selected_run[2]

# 2. Load run metadata safely
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

# Metrics Display
col_train, col_val = st.columns(2)
col_train.metric("Train accuracy", f"{train_accuracy:.4f}")
col_val.metric("Val accuracy", f"{val_accuracy:.4f}")

# Feature Importance Panel
feature_importances = step_data.get("feature_importances")
if feature_importances is not None:
    st.subheader("Feature importance")
    feature_names = [f"Feature {i + 1}" for i in range(len(feature_importances))]
    fig_importance = go.Figure(go.Bar(x=feature_names, y=feature_importances))
    fig_importance.update_layout(
        xaxis_title="feature",
        yaxis_title="importance",
        height=300,
        margin=dict(t=20),
    )
    st.plotly_chart(fig_importance, width="stretch")

# Main Visualization Columns
col_boundary, col_conf = st.columns(2)

# Extract and align data safely across all components
val_2d = step_data.get("val_2d", meta.get("val_2d"))
val_true_labels = step_data.get("true_labels", meta.get("val_true_labels"))
predictions = np.asarray(step_data.get("predictions", []))
labels = np.asarray(val_true_labels)

# Safe length alignment across predictions, labels, and 2D features
common_length = min(len(predictions), len(labels), len(val_2d))
predictions = predictions[:common_length]
labels = labels[:common_length]
val_2d = val_2d[:common_length]

# Panel 1: Decision Boundary
with col_boundary:
    st.subheader("Decision boundary")
    st.caption(
        "⚠️ Approximation: features are projected to 2D via PCA (fit once "
        "for this run and reused across every step). The boundary comes "
        "from evaluating the model on grid points inverse-transformed "
        "back to original feature space."
    )

    xx = meta["grid_xx"]
    yy = meta["grid_yy"]
    boundary_grid = step_data["boundary_grid"]

    error_cluster_labels = step_data.get("error_cluster_labels")
    if error_cluster_labels is None:
        error_cluster_labels = compute_error_clusters(val_2d, labels, predictions)
    else:
        error_cluster_labels = error_cluster_labels[:common_length]

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

    # Safe error calculation
    error_mask = predictions != labels
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

st.divider()

# Error Cluster Trend Chart
st.subheader("Error cluster trend")
cluster_counts = []
trend_steps = []
for _, step_value, _, _, npz_path, _ in snapshots:
    try:
        snapshot_values = logger.load_snapshot(npz_path)
    except FileNotFoundError:
        continue
    c_labels = snapshot_values.get("error_cluster_labels")
    if c_labels is None:
        continue
    non_negative = c_labels[c_labels >= 0]
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

# Continual Mode Specific Panels
if mode == "continual":
    st.divider()
    st.header("Continual Learning Analysis")

    col_forget, col_phase = st.columns(2)

    with col_forget:
        st.subheader("Forgetting events")
        forgetting_events = db.list_forgetting_events(run_id)
        if forgetting_events:
            cumulative_counts = []
            cumulative = 0
            steps_seen = []
            for step_key in sorted({event[0] for event in forgetting_events}):
                cumulative += sum(1 for event in forgetting_events if event[0] == step_key)
                cumulative_counts.append(cumulative)
                steps_seen.append(step_key)
            event_fig = go.Figure()
            event_fig.add_trace(go.Scatter(x=steps_seen, y=cumulative_counts, mode="lines+markers"))
            event_fig.update_layout(
                xaxis_title="step", yaxis_title="cumulative forgetting events", height=300, margin=dict(t=20)
            )
            st.plotly_chart(event_fig, width="stretch")
        else:
            st.info("No forgetting events recorded for this run yet.")

    with col_phase:
        st.subheader("Per-phase accuracy")
        phase_series = {}
        for _, step_value, _, _, npz_path, _ in snapshots:
            try:
                snapshot_values = logger.load_snapshot(npz_path)
            except FileNotFoundError:
                continue
            phase_labels = snapshot_values.get("phase_labels")
            phase_accuracy = snapshot_values.get("phase_accuracy")
            if phase_labels is None or phase_accuracy is None:
                continue
            for label, accuracy in zip(phase_labels, phase_accuracy):
                phase_series.setdefault(int(label), []).append((step_value, float(accuracy)))
        if phase_series:
            phase_fig = go.Figure()
            for phase_idx, series in sorted(phase_series.items()):
                x_values = [item[0] for item in series]
                y_values = [item[1] for item in series]
                phase_fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=f"phase {phase_idx + 1}"))
            phase_fig.update_layout(xaxis_title="step", yaxis_title="accuracy", height=300, margin=dict(t=20))
            st.plotly_chart(phase_fig, width="stretch")
        else:
            st.info("No per-phase accuracy data available for this run yet.")

    st.subheader("Phase timeline")
    phase_steps = [row[1] for row in snapshots if row[5] is not None]
    phase_ids = [row[5] for row in snapshots if row[5] is not None]
    if phase_steps and phase_ids:
        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(x=steps, y=[0] * len(steps), mode="markers+lines", showlegend=False))
        for phase_idx in sorted(set(phase_ids)):
            phase_mask = np.array([pid == phase_idx for pid in phase_ids])
            phase_step_values = np.array(phase_steps)[phase_mask]
            if len(phase_step_values) == 0:
                continue
            start = phase_step_values[0]
            end = phase_step_values[-1]
            timeline_fig.add_vrect(
                x0=start, x1=end, line_width=0, fillcolor=f"rgba({(phase_idx - 1) * 80}, 120, 180, 0.15)"
            )
        timeline_fig.update_layout(xaxis_title="step", yaxis_title="phase", height=220, margin=dict(t=20))
        st.plotly_chart(timeline_fig, width="stretch")