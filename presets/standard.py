from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

DATASET_NAME = "standard_synthetic"


def prepare_dataset(random_state=42):
    """Synthetic binary classification set: a handful of informative
    dims, an equal handful of linear-combination (redundant) dims, and
    plain noise dims. Enough real structure that feature importance
    (phase 3) has something to find and PCA boundary projection
    (phase 2) is a genuine approximation, not a dataset that's already
    2D in disguise.
    """
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=6,
        n_redundant=6,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.02,
        class_sep=1.0,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val
