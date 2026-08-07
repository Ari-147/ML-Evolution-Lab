import argparse
import uuid

import numpy as np

import config
import storage.db as db
import snapshotting.logger as logger
import snapshotting.boundary as boundary
from analysis.error_clusters import compute_error_clusters
from analysis.feature_importance import compute_feature_importance
from analysis.forgetting import forgetting_events
from presets import standard, continual, fairness
from analysis.fairness_metrics import compute_subgroup_metrics
from training.mlp_harness import train_mlp, predict as mlp_predict
from training.rf_harness import train_random_forest


def build_continual_snapshot_payload(
    phase_holdouts,
    model,
    predict_fn,
    phase_index,
    pca=None,
    xx=None,
    grid_2d=None,
    val_2d=None,
    phase_val_y=None,
):
    per_phase_accuracy = []
    phase_labels = []
    true_labels = []
    for phase_idx, (phase_val_X, phase_val_y) in enumerate(phase_holdouts):
        phase_preds = predict_fn(phase_val_X)
        per_phase_accuracy.append(float(np.mean(phase_preds == phase_val_y)))
        phase_labels.append(phase_idx)
        true_labels.append(np.asarray(phase_val_y))

    payload = {
        "phase_accuracy": np.asarray(per_phase_accuracy, dtype=float),
        "phase_labels": np.asarray(phase_labels, dtype=int),
        "true_labels": (
            np.asarray(phase_val_y, dtype=int)
            if phase_val_y is not None
            else np.concatenate(true_labels) if true_labels else np.asarray([], dtype=int)
        ),
    }

    if pca is not None and xx is not None and grid_2d is not None and model is not None:
        payload["boundary_grid"] = boundary.compute_boundary(
            pca,
            xx,
            grid_2d,
            lambda X: predict_fn(X),
        )

    if val_2d is not None:
        payload["val_2d"] = val_2d

    return payload


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
                    "true_labels": y_val,
                    "val_2d": X_val_2d,
                },
            )

    if model_family == "mlp":
        train_mlp(X_train, y_train, X_val, y_val, epoch_callback=on_step)
    else:
        train_random_forest(X_train, y_train, X_val, y_val, step_callback=on_step)

    print(f"done. run_id={run_id}")


def run_fairness_training(model_family):
    db.init_db()

    run_id = str(uuid.uuid4())
    dataset = fairness.prepare_fairness_dataset()

    db.create_run(
        run_id=run_id,
        model_family=model_family,
        mode="fairness",
        dataset_name=fairness.DATASET_NAME,
        config_snapshot={
            "MAX_EPOCHS": config.MAX_EPOCHS,
            "SNAPSHOT_DENSE_STEPS": config.SNAPSHOT_DENSE_STEPS,
            "SNAPSHOT_EVERY_N": config.SNAPSHOT_EVERY_N,
            "PCA_COMPONENTS": config.PCA_COMPONENTS,
            "GRID_RESOLUTION": config.GRID_RESOLUTION,
        },
    )

    pca, X_val_2d = boundary.fit_projection(dataset["X_val"])
    xx, yy, grid_2d = boundary.build_grid(X_val_2d)
    logger.save_run_meta(
        run_id,
        grid_xx=xx,
        grid_yy=yy,
        val_2d=X_val_2d,
        val_true_labels=dataset["y_val"],
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
            subgroup_metrics = compute_subgroup_metrics(
                dataset["subgroup_val"],
                val_predictions,
                val_confidences,
                dataset["y_val"],
            )
            db.save_subgroup_metrics(run_id, current_step, subgroup_metrics)
            logger.save_snapshot(
                run_id=run_id,
                step=current_step,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                predictions=val_predictions,
                confidences=val_confidences,
                extra_arrays={
                    "boundary_grid": boundary_grid,
                    "true_labels": dataset["y_val"],
                    "val_2d": X_val_2d,
                    "subgroup_metrics": np.asarray([metric["subgroup_value"] for metric in subgroup_metrics], dtype=object),
                    "subgroup_accuracy": np.asarray([metric["accuracy"] for metric in subgroup_metrics], dtype=float),
                    "subgroup_positive_rate": np.asarray([metric["positive_rate"] for metric in subgroup_metrics], dtype=float),
                    "subgroup_avg_confidence": np.asarray([metric["avg_confidence"] for metric in subgroup_metrics], dtype=float),
                },
            )

    if model_family == "mlp":
        train_mlp(dataset["X_train"], dataset["y_train"], dataset["X_val"], dataset["y_val"], epoch_callback=on_step)
    else:
        train_random_forest(dataset["X_train"], dataset["y_train"], dataset["X_val"], dataset["y_val"], step_callback=on_step)

    print(f"done. run_id={run_id}")


def run_continual_training():
    db.init_db()

    run_id = str(uuid.uuid4())
    phase_data = continual.prepare_continual_dataset()

    db.create_run(
        run_id=run_id,
        model_family="mlp",
        mode="continual",
        dataset_name=continual.DATASET_NAME,
        config_snapshot={
            "MAX_EPOCHS": config.MAX_EPOCHS,
            "SNAPSHOT_DENSE_STEPS": config.SNAPSHOT_DENSE_STEPS,
            "SNAPSHOT_EVERY_N": config.SNAPSHOT_EVERY_N,
            "PCA_COMPONENTS": config.PCA_COMPONENTS,
            "GRID_RESOLUTION": config.GRID_RESOLUTION,
        },
    )

    pca, X_val_2d = boundary.fit_projection(np.concatenate([phase[1] for phase in phase_data]))
    xx, yy, grid_2d = boundary.build_grid(X_val_2d)
    logger.save_run_meta(
        run_id,
        grid_xx=xx,
        grid_yy=yy,
        val_2d=X_val_2d,
        val_true_labels=np.concatenate([phase[3] for phase in phase_data]),
    )

    all_phase_val_X = np.concatenate([phase[1] for phase in phase_data])
    phase_holdouts = []
    for phase in phase_data:
        phase_holdouts.append((phase[1], phase[3]))

    def on_step(
        step=None,
        epoch=None,
        train_accuracy=None,
        val_accuracy=None,
        val_predictions=None,
        val_confidences=None,
        model=None,
        phase_id=None,
        phase_val_X=None,
        phase_val_y=None,
    ):
        current_step = step if step is not None else epoch
        print(f"step {current_step:3d}  train_acc={train_accuracy:.4f}  val_acc={val_accuracy:.4f}")
        if logger.should_log_step(current_step):
            payload = build_continual_snapshot_payload(
                phase_holdouts=phase_holdouts,
                model=model,
                predict_fn=lambda X: mlp_predict(model, X),
                phase_index=phase_id - 1 if phase_id is not None else 0,
                pca=pca,
                xx=xx,
                grid_2d=grid_2d,
                val_2d=pca.transform(phase_val_X) if phase_val_X is not None else None,
                phase_val_y=phase_val_y,
            )
            logger.save_snapshot(
                run_id=run_id,
                step=current_step,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                predictions=val_predictions,
                confidences=val_confidences,
                extra_arrays=payload,
                phase_id=phase_id,
            )

    phase_counter = 0
    for phase_index, (X_train, X_val, y_train, y_val, phase_classes) in enumerate(phase_data):
        phase_counter += 1
        phase_epochs = max(1, config.MAX_EPOCHS // len(phase_data))

        def phase_callback(
            epoch,
            train_accuracy,
            val_accuracy,
            val_predictions,
            val_confidences,
            model,
            phase_index=phase_index,
            phase_val_X=X_val,
            phase_val_y=y_val,
        ):
            current_step = (phase_index * phase_epochs) + epoch
            on_step(
                step=current_step,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                val_predictions=val_predictions,
                val_confidences=val_confidences,
                model=model,
                phase_id=phase_index + 1,
                phase_val_X=phase_val_X,
                phase_val_y=phase_val_y,
            )

        train_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            max_epochs=phase_epochs,
            epoch_callback=phase_callback,
        )

    events = forgetting_events(run_id)
    db.save_forgetting_events(run_id, events)
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
        elif args.mode == "continual" and args.model_family == "mlp":
            run_continual_training()
        elif args.mode == "fairness" and args.model_family in {"mlp", "random_forest"}:
            run_fairness_training(args.model_family)
        else:
            raise NotImplementedError(
                f"train {args.mode} {args.model_family} isn't implemented yet"
            )


if __name__ == "__main__":
    main()
