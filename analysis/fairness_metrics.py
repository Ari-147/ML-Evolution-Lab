import numpy as np


def compute_subgroup_metrics(subgroups, predictions, confidences, true_labels):
    """Compute subgroup accuracy, average confidence, and positive-rate metrics."""
    subgroups = np.asarray(subgroups)
    predictions = np.asarray(predictions)
    confidences = np.asarray(confidences)
    true_labels = np.asarray(true_labels)

    common_length = min(len(subgroups), len(predictions), len(confidences), len(true_labels))
    subgroups = subgroups[:common_length]
    predictions = predictions[:common_length]
    confidences = confidences[:common_length]
    true_labels = true_labels[:common_length]

    metrics = []
    for subgroup_value in np.unique(subgroups):
        mask = subgroups == subgroup_value
        subgroup_predictions = predictions[mask]
        subgroup_confidences = confidences[mask]
        subgroup_true = true_labels[mask]
        if len(subgroup_true) == 0:
            continue
        metrics.append(
            {
                "subgroup_value": str(subgroup_value),
                "accuracy": float(np.mean(subgroup_predictions == subgroup_true)),
                "avg_confidence": float(np.mean(subgroup_confidences)),
                "positive_rate": float(np.mean(subgroup_predictions == 1)),
            }
        )
    return metrics
