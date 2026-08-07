import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from analysis.feature_importance import compute_feature_importance


class FeatureImportanceTests(unittest.TestCase):
    def test_compute_feature_importance_returns_expected_shape(self):
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=4,
            n_redundant=0,
            random_state=0,
        )
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

        model = RandomForestClassifier(n_estimators=20, random_state=0)
        model.fit(X_train, y_train)

        importances = compute_feature_importance(
            model=model,
            X=X_val,
            y=y_val,
            n_repeats=3,
            random_state=0,
        )

        self.assertEqual(importances.shape, (X_val.shape[1],))
        self.assertTrue(np.isfinite(importances).all())

    def test_compute_feature_importance_accepts_predict_wrapper(self):
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=4,
            n_redundant=0,
            random_state=1,
        )
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

        model = RandomForestClassifier(n_estimators=20, random_state=1)
        model.fit(X_train, y_train)

        def wrapped_predict(X):
            return model.predict(X)

        importances = compute_feature_importance(
            model=None,
            X=X_val,
            y=y_val,
            n_repeats=2,
            random_state=1,
            predict_fn=wrapped_predict,
        )

        self.assertEqual(importances.shape, (X_val.shape[1],))
        self.assertTrue(np.isfinite(importances).all())


if __name__ == "__main__":
    unittest.main()
