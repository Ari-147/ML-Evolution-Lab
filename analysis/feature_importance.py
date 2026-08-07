import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance

import config


class PredictWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.predict_fn(X)


def compute_feature_importance(
    model=None,
    X=None,
    y=None,
    n_repeats=None,
    random_state=None,
    predict_fn=None,
):
    """Return a 1D feature-importance vector for a trained model.

    RandomForest-style estimators expose feature_importances_ directly.
    For MLP-style models, we wrap a predict function so sklearn's
    permutation_importance can be used without changing the training API.
    """
    if X is None or y is None:
        raise ValueError("X and y are required")

    n_repeats = n_repeats or config.PERMUTATION_N_REPEATS
    random_state = random_state if random_state is not None else 0

    if model is not None and hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)

    estimator = model
    if predict_fn is not None:
        estimator = PredictWrapper(predict_fn)
    elif model is None:
        raise ValueError("Either model or predict_fn must be provided")

    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy",
        n_jobs=1,
    )
    return np.asarray(result.importances_mean, dtype=float)
