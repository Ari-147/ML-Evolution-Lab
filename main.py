import argparse
import uuid

import numpy as np

import config
import storage.db as db
import snapshotting.logger as logger
import snapshotting.boundary as boundary
from analysis.error_clusters import compute_error_clusters
from analysis.feature_importance import compute_feature_importance
from presets import standard
from training.mlp_harness import train_mlp, predict as mlp_predict
from training.rf_harness import train_random_forest


def run_standard_training(model_family):
    db.init_db()

    run_id = str(uuid.uuid4())
    X_train, X_val, y_train, y_val = standard.prepare_dataset()

    db.create_run(
        run_id=run_id,
        model_family=model_family,
        mode="standard",
        dataset_name=standard.DATASET_NAME,
        config_snapshot={
            "MAX_EPOCHS": config.MAX_EPOCHS,
            "MAX_TREES": config.MAX_TREES,
            "TREES_PER_STEP": config.TREES_PER_STEP,
            "SNAPSHOT_DENSE_STEPS": config.SNAPSHOT_DENSE_STEPS,
            "SNAPSHOT_EVERY_N": config.SNAPSHOT_EVERY_N,
            "PCA_COMPONENTS": config.PCA_COMPONENTS,
            "GRID_RESOLUTION": config.GRID_RESOLUTION,
        },
    )

    pca, X_val_2d = boundary.fit_projection(X_val)
    xx, yy, grid_2d = boundary.build_grid(X_val_2d)
    logger.save_run_meta(
        run_id,
        grid_xx=xx,
        grid_yy=yy,
        val_2d=X_val_2d,
        val_true_labels=y_val,
    )

    def predict_for_model(model, X):
        if model_family == "mlp":
            return mlp_predict(model, X)
        return model.predict(X)

    def on_step(step=None, epoch=None, train_accuracy=None, val_accuracy=None, val_predictions=None, val_confidences=None, model=None):
        current_step = step if step is not None else epoch
        print(f"step {current_step:3d}  train_acc={train_accuracy:.4f}  val_acc={val_accuracy:.4f}")
        if logger.should_log_step(current_step):
            boundary_grid = boundary.compute_boundary(
                pca, xx, grid_2d, lambda X: predict_for_model(model, X)
            )
            feature_importances = compute_feature_importance(
                model=model,
                X=X_val,
                y=y_val,
                n_repeats=config.PERMUTATION_N_REPEATS,
                random_state=42,
                predict_fn=(lambda X: mlp_predict(model, X)) if model_family == "mlp" else None,
            )
            error_cluster_labels = compute_error_clusters(
                X_val_2d,
                y_val,
                val_predictions,
            )
            cluster_groups = {}
            for cluster_id in np.unique(error_cluster_labels[error_cluster_labels >= 0]):
                cluster_groups[int(cluster_id)] = [
                    int(idx) for idx, label in enumerate(error_cluster_labels) if label == int(cluster_id)
                ]
            db.save_error_clusters(run_id, current_step, cluster_groups)
            logger.save_snapshot(
                run_id=run_id,
                step=current_step,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                predictions=val_predictions,
                confidences=val_confidences,
                extra_arrays={
                    "boundary_grid": boundary_grid,
                    "feature_importances": feature_importances,
                    "error_cluster_labels": error_cluster_labels,
                },
            )

    if model_family == "mlp":
        train_mlp(X_train, y_train, X_val, y_val, epoch_callback=on_step)
    else:
        train_random_forest(X_train, y_train, X_val, y_val, step_callback=on_step)

    print(f"done. run_id={run_id}")


def main():
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("mode", choices=["standard", "continual", "fairness"])
    train_parser.add_argument("model_family", choices=["mlp", "random_forest"])

    args = parser.parse_args()

    if args.command == "train":
        if args.mode == "standard" and args.model_family in {"mlp", "random_forest"}:
            run_standard_training(args.model_family)
        else:
            raise NotImplementedError(
                f"train {args.mode} {args.model_family} isn't implemented yet"
            )


if __name__ == "__main__":
    main()
