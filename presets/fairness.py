import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import config

DATASET_NAME = "adult"
SUBGROUP_COLUMN = "sex"


def prepare_fairness_dataset(random_state=42):
    """Load the UCI Adult Income dataset and split it into train/val sets.

    The first fetch requires network access; the dataset is cached under
    config.DATA_DIR for subsequent runs.
    """
    cache_dir = os.path.join(config.DATA_DIR, "openml")
    os.makedirs(cache_dir, exist_ok=True)

    data = fetch_openml("adult", version=2, as_frame=True, cache=True, data_home=cache_dir)
    frame = data.frame

    subgroup_column = SUBGROUP_COLUMN
    if subgroup_column not in frame.columns:
        raise KeyError(f"Expected subgroup column {subgroup_column!r} in Adult dataset")

    target_column = None
    for candidate in ["income", "class"]:
        if candidate in frame.columns:
            target_column = candidate
            break
    if target_column is None:
        raise KeyError("Could not find a target column in the Adult dataset")

    X = frame.drop(columns=[subgroup_column, target_column])
    y = frame[target_column]
    subgroup = frame[subgroup_column].astype(str)

    target = np.where(y == ">50K", 1, 0).astype(int)
    feature_frame = X.copy()
    for column in feature_frame.columns:
        if feature_frame[column].dtype == "object" or feature_frame[column].dtype.name == "category":
            feature_frame[column] = feature_frame[column].astype("category").cat.codes
        else:
            feature_frame[column] = feature_frame[column].fillna(feature_frame[column].median())

    feature_frame = feature_frame.astype(float)

    X_train, X_val, y_train, y_val, subgroup_train, subgroup_val = train_test_split(
        feature_frame.to_numpy(dtype=float),
        target,
        subgroup,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "subgroup_train": subgroup_train,
        "subgroup_val": subgroup_val,
        "subgroup_column": subgroup_column,
    }
