import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

DATASET_NAME = "digits_continual"


def prepare_continual_dataset(random_state=42):
    """Create a 3-phase continual-learning dataset from digits.

    Classes are split into sequential phases so the model can be trained
    phase by phase without replay, which induces catastrophic forgetting.
    """
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target.astype(int)

    phase_slices = [list(range(0, 3)), list(range(3, 6)), list(range(6, 10))]
    phase_data = []
    for phase_classes in phase_slices:
        phase_mask = np.isin(y, phase_classes)
        X_phase = X[phase_mask]
        y_phase = y[phase_mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_phase,
            y_phase,
            test_size=0.25,
            random_state=random_state,
            stratify=y_phase,
        )
        phase_data.append((X_train, X_test, y_train, y_test, phase_classes))

    return phase_data
