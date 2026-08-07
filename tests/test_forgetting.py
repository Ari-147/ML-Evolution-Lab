import unittest

import numpy as np

from analysis.forgetting import compute_forgetting_events
from main import build_continual_snapshot_payload


class ForgettingTests(unittest.TestCase):
    def test_compute_forgetting_events_returns_transitions(self):
        correctness_sequences = [
            (1, [True, False, True]),
            (2, [False, False, True]),
            (3, [False, True, True]),
        ]

        events = compute_forgetting_events(correctness_sequences)

        self.assertEqual(events, [(2, 0, "forgotten"), (3, 1, "learned")])

    def test_build_continual_snapshot_payload_uses_full_validation_labels(self):
        labels = np.array([0, 1, 0, 1])
        payload = build_continual_snapshot_payload(
            phase_holdouts=[(np.array([[0.0], [1.0], [2.0], [3.0]]), labels)],
            model=None,
            predict_fn=lambda X: np.array([0, 1, 0, 1]),
            phase_index=0,
        )

        self.assertIn("true_labels", payload)
        self.assertTrue(np.array_equal(payload["true_labels"], labels))

    def test_build_continual_snapshot_payload_uses_phase_specific_labels_and_projection(self):
        phase_labels = np.array([0, 1, 0])
        phase_projection = np.array([[0.1], [0.2], [0.3]])
        payload = build_continual_snapshot_payload(
            phase_holdouts=[(np.array([[0.0], [1.0], [2.0]]), phase_labels)],
            model=None,
            predict_fn=lambda X: np.array([0, 1, 0]),
            phase_index=0,
            phase_val_y=phase_labels,
            val_2d=phase_projection,
        )

        self.assertTrue(np.array_equal(payload["true_labels"], phase_labels))
        self.assertTrue(np.array_equal(payload["val_2d"], phase_projection))


if __name__ == "__main__":
    unittest.main()
