import numpy as np
import plotly.graph_objects as go
import streamlit as st

import storage.db as db
import snapshotting.logger as logger
from analysis.error_clusters import compute_error_clusters

st.set_page_config(page_title="ML Evolution Lab", layout="wide")
st.title("ML Evolution Lab")


def build_run_label(run_row):
    run_id, model_family, mode, dataset_name, created_at = run_row
    return f"{mode.title()} / {model_family} — {dataset_name} — {created_at}"


def load_run_snapshot_context(run_id, step):
    meta = logger.load_run_meta(run_id)
    snapshots = db.list_snapshots(run_id)
    snapshot_row = next((row for row in snapshots if row[1] == step), None)
    if snapshot_row is None:
        raise KeyError(f"Step {step} was not found for run {run_id}")
    step_data = logger.load_snapshot(snapshot_row[4])
    return meta, snapshot_row, step_data, snapshots


def align_snapshot_arrays(predictions, labels, val_2d, confidences=None):
    common_length = min(len(predictions), len(labels), len(val_2d))
    predictions = np.asarray(predictions[:common_length], dtype=np.int64)
    labels = np.asarray(labels[:common_length], dtype=np.int64)
    val_2d = np.asarray(val_2d[:common_length], dtype=float)
    if confidences is None:
        return predictions, labels, val_2d
    confidences = np.asarray(confidences[:common_length], dtype=float)
    return predictions, labels, val_2d, confidences


def build_boundary_figure(meta, step_data, predictions, labels, val_2d, highlight=None):
    fig = go.Figure()
    grid_xx = np.asarray(meta.get("grid_xx", []))
    grid_yy = np.asarray(meta.get("grid_yy", []))
    boundary_grid = np.asarray(step_data.get("boundary_grid", []))
    if highlight is not None:
        highlight = np.asarray(highlight, dtype=object)

    if grid_xx.size and grid_yy.size and boundary_grid.size:
        fig.add_trace(
            go.Contour(
                x=grid_xx[0],
                y=grid_yy[:, 0],
                z=boundary_grid,
                showscale=False,
                opacity=0.5,
                colorscale="RdBu",
                contours=dict(showlines=False),
                hoverinfo="skip",
            )
        )

    if highlight is None:
        error_cluster_labels = step_data.get("error_cluster_labels")
        if error_cluster_labels is None:
            error_cluster_labels = compute_error_clusters(val_2d, labels, predictions)
        else:
            error_cluster_labels = np.asarray(error_cluster_labels[: len(labels)])
        error_mask = predictions != labels
        misclassified_points = val_2d[error_mask]
        misclassified_cluster_ids = error_cluster_labels[error_mask]
        if len(misclassified_points):
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
    else:
        status_order = [
            ("became_correct", "#16a34a"),
            ("became_incorrect", "#dc2626"),
            ("stable_correct", "#2563eb"),
            ("stable_incorrect", "#f59e0b"),
        ]
        for status, color in status_order:
            mask = highlight == status
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=val_2d[mask, 0],
                        y=val_2d[mask, 1],
                        mode="markers",
                        marker=dict(color=color, size=8, line=dict(width=0.5, color="black")),
                        name=status.replace("_", " "),
                    )
                )

    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", height=300, margin=dict(t=20))
    return fig


def build_feature_importance_chart(step_data):
    feature_importances = step_data.get("feature_importances")
    if feature_importances is None:
        return None
    feature_names = [f"Feature {i + 1}" for i in range(len(feature_importances))]
    fig = go.Figure(go.Bar(x=feature_names, y=np.asarray(feature_importances, dtype=float)))
    fig.update_layout(xaxis_title="feature", yaxis_title="importance", height=300, margin=dict(t=20))
    return fig


def build_confidence_chart(step_data):
    confidences = np.asarray(step_data.get("confidences", []), dtype=float)
    fig = go.Figure(go.Histogram(x=confidences, nbinsx=30))
    fig.update_layout(xaxis_title="confidence (softmax max prob)", yaxis_title="count", height=300, margin=dict(t=20))
    return fig


def build_flip_statuses(predictions_a, labels_a, predictions_b, labels_b):
    correct_a = predictions_a == labels_a
    correct_b = predictions_b == labels_b
    statuses = np.array(["stable_correct"] * len(labels_a), dtype=object)
    statuses[correct_a & ~correct_b] = "became_incorrect"
    statuses[~correct_a & correct_b] = "became_correct"
    statuses[~correct_a & ~correct_b] = "stable_incorrect"
    return statuses


runs = db.list_runs()
if not runs:
    st.warning("No runs found yet. Run `python main.py train standard mlp`, `python main.py train standard random_forest`, or `python main.py train continual mlp` first.")
    st.stop()

preset_options = ["Standard", "Continual", "Fairness"]
preset_label = st.selectbox("Preset", preset_options, index=0)
mode = preset_label.lower()

mode_runs = [row for row in runs if row[2] == mode]
if not mode_runs:
    st.warning(f"No {preset_label.lower()} runs found yet.")
    st.stop()

if mode == "standard":
    model_family = st.selectbox("Model family", ["mlp", "random_forest"], index=0)
    available_runs = [row for row in mode_runs if row[1] == model_family]
else:
    available_runs = mode_runs
    model_family = None

if not available_runs:
    st.warning(f"No {preset_label.lower()} runs found for the selected family.")
    st.stop()

run_labels = {build_run_label(row): row[0] for row in available_runs}
selected_label = st.selectbox("Run", list(run_labels.keys()))
run_id = run_labels[selected_label]
selected_run = next(row for row in available_runs if row[0] == run_id)

try:
    meta, snapshot_row, step_data, snapshots = load_run_snapshot_context(run_id, None)
except Exception:
    meta = logger.load_run_meta(run_id)
    snapshots = db.list_snapshots(run_id)
    step_data = None

if not snapshots:
    st.warning("This run has no snapshots recorded yet.")
    st.stop()

steps = [row[1] for row in snapshots]
step = st.select_slider("Step", options=steps, value=steps[-1])

snapshot_row = next((row for row in snapshots if row[1] == step), None)
if snapshot_row is None:
    st.error("Selected step was not found in the run snapshots.")
    st.stop()

try:
    meta, snapshot_row, step_data, snapshots = load_run_snapshot_context(run_id, step)
except FileNotFoundError:
    st.error(f"Snapshot files missing for run ID `{run_id}`. Please select another run or re-run training.")
    st.stop()

_, _, train_accuracy, val_accuracy, _, phase_id = snapshot_row

col_train, col_val = st.columns(2)
col_train.metric("Train accuracy", f"{train_accuracy:.4f}")
col_val.metric("Val accuracy", f"{val_accuracy:.4f}")

val_2d = np.asarray(meta.get("val_2d", []), dtype=float)
val_true_labels = np.asarray(meta.get("val_true_labels", []), dtype=np.int64)
predictions = np.asarray(step_data.get("predictions", []), dtype=np.int64)
labels = np.asarray(val_true_labels, dtype=np.int64)
predictions, labels, val_2d = align_snapshot_arrays(predictions, labels, val_2d)

if mode == "standard":
    st.divider()
    st.header("Standard preset")
    st.caption("Boundary, feature importance, and error-cluster views stay tied to the active step slider.")

    feature_fig = build_feature_importance_chart(step_data)
    if feature_fig is not None:
        st.subheader("Feature importance")
        st.plotly_chart(feature_fig, use_container_width=True)

    col_boundary, col_conf = st.columns(2)
    with col_boundary:
        st.subheader("Decision boundary")
        st.caption("Approximation: PCA projects features to 2D and the boundary is evaluated on an inverse-transformed grid.")
        boundary_fig = build_boundary_figure(meta, step_data, predictions, labels, val_2d)
        st.plotly_chart(boundary_fig, use_container_width=True)
    with col_conf:
        st.subheader("Confidence distribution")
        confidence_fig = build_confidence_chart(step_data)
        st.plotly_chart(confidence_fig, use_container_width=True)

    st.divider()
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
        non_negative = np.asarray(c_labels)[np.asarray(c_labels) >= 0]
        cluster_counts.append(len(np.unique(non_negative)))
        trend_steps.append(step_value)
    if trend_steps:
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=trend_steps, y=cluster_counts, mode="lines+markers"))
        trend_fig.update_layout(xaxis_title="step", yaxis_title="distinct error clusters", height=300, margin=dict(t=20))
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("No error-cluster history is available for this run yet.")

elif mode == "continual":
    st.divider()
    st.header("Continual preset")
    st.caption("The step slider remains the single source of truth while the phase and forgetting views summarize the whole trajectory.")

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
            event_fig.update_layout(xaxis_title="step", yaxis_title="cumulative forgetting events", height=300, margin=dict(t=20))
            st.plotly_chart(event_fig, use_container_width=True)
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
                phase_fig.add_trace(go.Scatter(x=[item[0] for item in series], y=[item[1] for item in series], mode="lines+markers", name=f"phase {phase_idx + 1}"))
            phase_fig.update_layout(xaxis_title="step", yaxis_title="accuracy", height=300, margin=dict(t=20))
            st.plotly_chart(phase_fig, use_container_width=True)
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
            timeline_fig.add_vrect(x0=phase_step_values[0], x1=phase_step_values[-1], line_width=0, fillcolor=f"rgba({(phase_idx - 1) * 80}, 120, 180, 0.15)")
        timeline_fig.update_layout(xaxis_title="step", yaxis_title="phase", height=220, margin=dict(t=20))
        st.plotly_chart(timeline_fig, use_container_width=True)

else:
    st.divider()
    st.header("Fairness preset")
    st.caption("The fairness section surfaces how subgroup metrics evolve alongside the active training step.")

    fairness_metrics = []
    for _, step_value, _, _, npz_path, _ in snapshots:
        try:
            snapshot_values = logger.load_snapshot(npz_path)
        except FileNotFoundError:
            continue
        subgroup_values = snapshot_values.get("subgroup_metrics")
        subgroup_accuracy = snapshot_values.get("subgroup_accuracy")
        subgroup_positive_rate = snapshot_values.get("subgroup_positive_rate")
        if subgroup_values is None or subgroup_accuracy is None or subgroup_positive_rate is None:
            continue
        fairness_metrics.append((step_value, subgroup_values, subgroup_accuracy, subgroup_positive_rate))

    if fairness_metrics:
        subgroup_names = sorted({str(value) for _, subgroup_values, _, _ in fairness_metrics for value in subgroup_values})
        accuracy_fig = go.Figure()
        positive_rate_fig = go.Figure()
        for subgroup_name in subgroup_names:
            accuracy_points = []
            positive_points = []
            for step_value, subgroup_values, subgroup_accuracy, subgroup_positive_rate in fairness_metrics:
                subgroup_index = None
                for idx, value in enumerate(subgroup_values):
                    if str(value) == subgroup_name:
                        subgroup_index = idx
                        break
                if subgroup_index is None:
                    continue
                accuracy_points.append((step_value, float(subgroup_accuracy[subgroup_index])))
                positive_points.append((step_value, float(subgroup_positive_rate[subgroup_index])))
            if accuracy_points:
                accuracy_fig.add_trace(go.Scatter(x=[point[0] for point in accuracy_points], y=[point[1] for point in accuracy_points], mode="lines+markers", name=subgroup_name))
            if positive_points:
                positive_rate_fig.add_trace(go.Scatter(x=[point[0] for point in positive_points], y=[point[1] for point in positive_points], mode="lines+markers", name=subgroup_name))
        if len(accuracy_fig.data) > 0:
            accuracy_fig.update_layout(xaxis_title="step", yaxis_title="accuracy", height=300, margin=dict(t=20))
            st.subheader("Per-subgroup accuracy")
            st.plotly_chart(accuracy_fig, use_container_width=True)
        if len(positive_rate_fig.data) > 0:
            positive_rate_fig.update_layout(xaxis_title="step", yaxis_title="positive prediction rate", height=300, margin=dict(t=20))
            st.subheader("Per-subgroup positive rate")
            st.plotly_chart(positive_rate_fig, use_container_width=True)
    else:
        st.info("No subgroup fairness metrics recorded for this run yet.")

st.divider()
st.header("Diff view")
comparison_mode = st.selectbox("Compare", ["Off", "Two steps in this run", "Two runs"], key="diff_mode")
if comparison_mode != "Off":
    if comparison_mode == "Two steps in this run":
        step_a = st.selectbox("Step A", steps, index=0, key="diff_step_a")
        step_b = st.selectbox("Step B", steps, index=len(steps) - 1, key="diff_step_b")
        meta_a, _, step_data_a, _ = load_run_snapshot_context(run_id, step_a)
        meta_b, _, step_data_b, _ = load_run_snapshot_context(run_id, step_b)
        run_a_label = f"{selected_label} · step {step_a}"
        run_b_label = f"{selected_label} · step {step_b}"
    else:
        candidate_runs = [row for row in mode_runs if row[0] != run_id]
        if mode == "standard" and model_family is not None:
            candidate_runs = [row for row in candidate_runs if row[1] == model_family]
        if not candidate_runs:
            st.info("No other runs are available to compare for this preset.")
            st.stop()
        other_run_labels = {build_run_label(row): row[0] for row in candidate_runs}
        other_run_label = st.selectbox("Compare with run", list(other_run_labels.keys()), key="diff_run_choice")
        other_run_id = other_run_labels[other_run_label]
        other_snapshots = db.list_snapshots(other_run_id)
        other_steps = [row[1] for row in other_snapshots]
        if not other_steps:
            st.info("The selected comparison run has no snapshots available.")
            st.stop()
        step_a = st.selectbox("Selected run step", steps, index=len(steps) - 1, key="diff_run_a_step")
        step_b = st.selectbox("Compared run step", other_steps, index=len(other_steps) - 1, key="diff_run_b_step")
        meta_a, _, step_data_a, _ = load_run_snapshot_context(run_id, step_a)
        meta_b, _, step_data_b, _ = load_run_snapshot_context(other_run_id, step_b)
        run_a_label = f"{selected_label} · step {step_a}"
        run_b_label = f"{other_run_label} · step {step_b}"

    val_2d_a = np.asarray(meta_a.get("val_2d", []), dtype=float)
    labels_a = np.asarray(meta_a.get("val_true_labels", []), dtype=np.int64)
    predictions_a = np.asarray(step_data_a.get("predictions", []), dtype=np.int64)
    predictions_a, labels_a, val_2d_a = align_snapshot_arrays(predictions_a, labels_a, val_2d_a)

    val_2d_b = np.asarray(meta_b.get("val_2d", []), dtype=float)
    labels_b = np.asarray(meta_b.get("val_true_labels", []), dtype=np.int64)
    predictions_b = np.asarray(step_data_b.get("predictions", []), dtype=np.int64)
    predictions_b, labels_b, val_2d_b = align_snapshot_arrays(predictions_b, labels_b, val_2d_b)

    common_length = min(len(labels_a), len(labels_b))
    labels_a = labels_a[:common_length]
    predictions_a = predictions_a[:common_length]
    val_2d_a = val_2d_a[:common_length]
    labels_b = labels_b[:common_length]
    predictions_b = predictions_b[:common_length]
    val_2d_b = val_2d_b[:common_length]
    flip_statuses = build_flip_statuses(predictions_a, labels_a, predictions_b, labels_b)

    summary = {
        "became_correct": int(np.sum(flip_statuses == "became_correct")),
        "became_incorrect": int(np.sum(flip_statuses == "became_incorrect")),
        "stable_correct": int(np.sum(flip_statuses == "stable_correct")),
        "stable_incorrect": int(np.sum(flip_statuses == "stable_incorrect")),
    }
    st.write(summary)

    st.subheader("Boundary comparison")
    boundary_col_a, boundary_col_b = st.columns(2)
    with boundary_col_a:
        st.caption(run_a_label)
        st.plotly_chart(build_boundary_figure(meta_a, step_data_a, predictions_a, labels_a, val_2d_a, flip_statuses), use_container_width=True)
    with boundary_col_b:
        st.caption(run_b_label)
        st.plotly_chart(build_boundary_figure(meta_b, step_data_b, predictions_b, labels_b, val_2d_b, flip_statuses), use_container_width=True)

    st.subheader("Feature importance comparison")
    feature_col_a, feature_col_b = st.columns(2)
    with feature_col_a:
        st.caption(run_a_label)
        feature_fig_a = build_feature_importance_chart(step_data_a)
        if feature_fig_a is not None:
            st.plotly_chart(feature_fig_a, use_container_width=True)
        else:
            st.info("No feature importances available for this snapshot.")
    with feature_col_b:
        st.caption(run_b_label)
        feature_fig_b = build_feature_importance_chart(step_data_b)
        if feature_fig_b is not None:
            st.plotly_chart(feature_fig_b, use_container_width=True)
        else:
            st.info("No feature importances available for this snapshot.")

    st.subheader("Confidence comparison")
    conf_col_a, conf_col_b = st.columns(2)
    with conf_col_a:
        st.caption(run_a_label)
        st.plotly_chart(build_confidence_chart(step_data_a), use_container_width=True)
    with conf_col_b:
        st.caption(run_b_label)
        st.plotly_chart(build_confidence_chart(step_data_b), use_container_width=True)
