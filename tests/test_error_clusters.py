import unittest

import numpy as np

from analysis.error_clusters import compute_error_clusters


class ErrorClustersTests(unittest.TestCase):
    def test_compute_error_clusters_returns_labels_for_misclassified_examples(self):
        val_2d = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])
        true_labels = np.array([0, 0, 1, 1, 0])
        predictions = np.array([1, 0, 1, 0, 0])

        labels = compute_error_clusters(val_2d, true_labels, predictions)

        self.assertEqual(labels.shape, (len(true_labels),))
        self.assertTrue(np.all(labels[~(predictions != true_labels)] == -1))
        self.assertTrue(np.any(labels[(predictions != true_labels)] >= 0))


if __name__ == "__main__":
    unittest.main()
