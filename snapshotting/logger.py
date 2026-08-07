import os
import numpy as np

import config
import storage.db as db


def should_log_step(step: int) -> bool:
    """Dense logging up through SNAPSHOT_DENSE_STEPS, then every
    SNAPSHOT_EVERY_N-th step after that."""
    if step <= config.SNAPSHOT_DENSE_STEPS:
        return True
    return step % config.SNAPSHOT_EVERY_N == 0


def save_snapshot(
    run_id,
    step,
    train_accuracy,
    val_accuracy,
    predictions,
    confidences,
    extra_arrays: dict = None,
    phase_id=None,
):
    """Writes the raw arrays to a per-step .npz (lazy-loaded later by
    the dashboard) and a matching Snapshot row in SQLite."""
    extra_arrays = extra_arrays or {}

    snapshot_dir = os.path.join(config.SNAPSHOT_DIR, run_id)
    os.makedirs(snapshot_dir, exist_ok=True)
    npz_path = os.path.join(snapshot_dir, f"step_{step}.npz")

    # Cast arrays explicitly to numeric dtypes to avoid object-array serialization
    cleaned_extra = {}
    for key, val in extra_arrays.items():
        arr = np.asarray(val)
        if arr.dtype == object:
            # Cast object string arrays to clean Unicode strings
            cleaned_extra[key] = arr.astype(str)
        else:
            cleaned_extra[key] = arr

    np.savez_compressed(
        npz_path,
        predictions=np.asarray(predictions, dtype=np.int64),
        confidences=np.asarray(confidences, dtype=np.float32),
        **cleaned_extra,
    )

    return db.create_snapshot(
        run_id=run_id,
        step=step,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        npz_path=npz_path,
        phase_id=phase_id,
    )


def save_run_meta(run_id, **arrays):
    """Arrays that are fixed for the whole run (PCA grid coordinates,
    the 2D-projected val points, true val labels) — saved once instead
    of duplicated into every per-step .npz."""
    snapshot_dir = os.path.join(config.SNAPSHOT_DIR, run_id)
    os.makedirs(snapshot_dir, exist_ok=True)
    meta_path = os.path.join(snapshot_dir, "meta.npz")

    cleaned_arrays = {}
    for key, val in arrays.items():
        arr = np.asarray(val)
        if arr.dtype == object:
            cleaned_arrays[key] = arr.astype(str)
        else:
            cleaned_arrays[key] = arr

    np.savez_compressed(meta_path, **cleaned_arrays)
    return meta_path


def load_run_meta(run_id):
    """Loads metadata array dictionary. Raises a clean FileNotFoundError
    if the run folder or meta.npz is missing."""
    snapshot_dir = os.path.join(config.SNAPSHOT_DIR, run_id)
    meta_path = os.path.join(snapshot_dir, "meta.npz")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Run metadata file does not exist at: {meta_path}. "
            "Ensure save_run_meta() was executed during training."
        )

    # Returning a dict unlinks the file handle immediately; allow_pickle=True prevents loading crashes
    with np.load(meta_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_snapshot(npz_path):
    """Loads a single step snapshot safely with allow_pickle=True."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Snapshot array not found at: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}