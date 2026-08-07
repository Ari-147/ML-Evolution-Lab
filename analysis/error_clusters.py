import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config


def compute_error_clusters(val_2d, true_labels, predictions, min_samples=3):
    """Cluster misclassified validation examples in 2D projection space.

    The routine identifies misclassified examples, tries k-values in the
    configured range, and uses silhouette score to select the best
    clustering. If there are too few misclassified examples for the
    minimum k, it returns a vector of -1 for all points.
    """
    val_2d = np.asarray(val_2d, dtype=float)
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)

    if len(true_labels) != len(predictions):
        common_length = min(len(true_labels), len(predictions))
        true_labels = true_labels[:common_length]
        predictions = predictions[:common_length]
        val_2d = val_2d[:common_length]

    error_mask = predictions != true_labels
    labels = np.full(len(true_labels), -1, dtype=int)

    min_k, max_k = config.ERROR_CLUSTER_K_RANGE
    min_samples = max(min_samples, min_k)

    misclassified = val_2d[error_mask]
    if len(misclassified) < min_samples:
        labels[error_mask] = np.zeros(len(misclassified), dtype=int)
        return labels

    candidates = range(min_k, max_k + 1)
    valid_k = [k for k in candidates if len(misclassified) >= k]
    if not valid_k:
        return labels

    best_score = -1.0
    best_clusters = None
    for k in valid_k:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_ids = kmeans.fit_predict(misclassified)
        if len(np.unique(cluster_ids)) < 2:
            continue
        score = silhouette_score(misclassified, cluster_ids)
        if score > best_score:
            best_score = score
            best_clusters = cluster_ids

    if best_clusters is None:
        labels[error_mask] = np.zeros(len(misclassified), dtype=int)
        return labels

    labels[error_mask] = best_clusters
    return labels
