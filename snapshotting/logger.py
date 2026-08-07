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

    np.savez(
        npz_path,
        predictions=np.asarray(predictions),
        confidences=np.asarray(confidences),
        **extra_arrays,
    )

    return db.create_snapshot(
        run_id=run_id,
        step=step,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        npz_path=npz_path,
        phase_id=phase_id,
    )
