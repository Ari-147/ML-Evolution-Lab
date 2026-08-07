import unittest

import numpy as np

from training.mlp_harness import train_mlp, predict


class MLPHarnessTests(unittest.TestCase):
    def test_train_mlp_supports_non_zero_based_labels(self):
        X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [3.0, 3.0],
        ], dtype=float)
        y = np.array([0, 0, 0, 0, 5, 5, 5, 5], dtype=int)

        model = train_mlp(X, y, X, y, max_epochs=3, epoch_callback=None)
        preds = predict(model, X)

        self.assertEqual(preds.shape, y.shape)
        self.assertTrue(np.isin(preds, [0, 5]).all())


if __name__ == "__main__":
    unittest.main()
