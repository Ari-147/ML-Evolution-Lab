import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config


def train_random_forest(X_train, y_train, X_val, y_val, max_trees=None, step_callback=None):
    """Warm-start Random Forest that grows by TREES_PER_STEP each step.

    The step axis is the number of trees in the ensemble, so snapshots are
    taken at 5, 10, 15, ... trees rather than at epochs.
    """
    max_trees = max_trees or config.MAX_TREES

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    model = RandomForestClassifier(
        n_estimators=config.TREES_PER_STEP,
        warm_start=True,
        random_state=42,
        n_jobs=-1,
    )

    for tree_count in range(config.TREES_PER_STEP, max_trees + 1, config.TREES_PER_STEP):
        model.n_estimators = tree_count
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        val_prob = model.predict_proba(X_val)
        val_confidences = val_prob[np.arange(len(val_preds)), val_preds]

        train_accuracy = float(np.mean(train_preds == y_train))
        val_accuracy = float(np.mean(val_preds == y_val))

        if step_callback is not None:
            step_callback(
                step=tree_count,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                val_predictions=val_preds,
                val_confidences=val_confidences,
                model=model,
            )

    return model
