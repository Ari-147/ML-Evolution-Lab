import numpy as np
from sklearn.decomposition import PCA

import config


def fit_projection(X_val, random_state=42):
    """Fit PCA once per run on the validation set. Reused across every
    step so boundary movement between steps is directly comparable."""
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=random_state)
    X_val_2d = pca.fit_transform(X_val)
    return pca, X_val_2d


def build_grid(X_val_2d, resolution=None):
    """A resolution x resolution grid over the 2D projected space, padded
    a bit past the val points' extent."""
    resolution = resolution or config.GRID_RESOLUTION
    x_min, x_max = X_val_2d[:, 0].min(), X_val_2d[:, 0].max()
    y_min, y_max = X_val_2d[:, 1].min(), X_val_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.1 or 1.0
    y_pad = (y_max - y_min) * 0.1 or 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, resolution),
        np.linspace(y_min - y_pad, y_max + y_pad, resolution),
    )
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid_2d


def compute_boundary(pca, xx, grid_2d, predict_fn):
    """Inverse-transforms grid points from the fixed 2D PCA space back to
    original feature space, evaluates predict_fn on them, and reshapes
    the predicted classes back to the grid's shape. This is an
    approximation for any dataset with more than PCA_COMPONENTS features
    -- not the true high-dimensional boundary."""
    grid_original = pca.inverse_transform(grid_2d)
    preds = np.asarray(predict_fn(grid_original))
    return preds.reshape(xx.shape)
