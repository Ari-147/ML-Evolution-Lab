import numpy as np

import storage.db as db


def compute_forgetting_events(correctness_sequences):
    """Compute forgetting/learning transitions from ordered correctness arrays."""
    events = []
    previous_correct = None
    for step, correctness in correctness_sequences:
        if previous_correct is None:
            previous_correct = correctness
            continue
        if len(previous_correct) != len(correctness):
            previous_correct = correctness
            continue
        for example_id, (prev, curr) in enumerate(zip(previous_correct, correctness)):
            if prev and not curr:
                events.append((step, example_id, "forgotten"))
            elif not prev and curr:
                events.append((step, example_id, "learned"))
        previous_correct = correctness
    return events


def forgetting_events(run_id):
    """Return forgetting/learning event transitions across ordered snapshots."""
    snapshots = db.list_snapshots(run_id)
    if not snapshots:
        return []

    correctness_sequences = []
    for _, step, _, _, npz_path, _ in snapshots:
        snapshot_data = np.load(npz_path, allow_pickle=True)
        predictions = np.asarray(snapshot_data["predictions"])
        labels = np.asarray(snapshot_data["true_labels"]) if "true_labels" in snapshot_data else None
        if labels is None:
            continue
        common_length = min(len(predictions), len(labels))
        correctness_sequences.append((step, predictions[:common_length] == labels[:common_length]))

    return compute_forgetting_events(correctness_sequences)
