import argparse
import uuid

import config
import storage.db as db
import snapshotting.logger as logger
from presets import standard
from training.mlp_harness import train_mlp


def run_standard_mlp():
    db.init_db()

    run_id = str(uuid.uuid4())
    X_train, X_val, y_train, y_val = standard.prepare_dataset()

    db.create_run(
        run_id=run_id,
        model_family="mlp",
        mode="standard",
        dataset_name=standard.DATASET_NAME,
        config_snapshot={
            "MAX_EPOCHS": config.MAX_EPOCHS,
            "SNAPSHOT_DENSE_STEPS": config.SNAPSHOT_DENSE_STEPS,
            "SNAPSHOT_EVERY_N": config.SNAPSHOT_EVERY_N,
        },
    )

    def on_epoch(epoch, train_accuracy, val_accuracy, val_predictions, val_confidences):
        print(f"epoch {epoch:3d}  train_acc={train_accuracy:.4f}  val_acc={val_accuracy:.4f}")
        if logger.should_log_step(epoch):
            logger.save_snapshot(
                run_id=run_id,
                step=epoch,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                predictions=val_predictions,
                confidences=val_confidences,
            )

    train_mlp(X_train, y_train, X_val, y_val, epoch_callback=on_epoch)
    print(f"done. run_id={run_id}")


def main():
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("mode", choices=["standard", "continual", "fairness"])
    train_parser.add_argument("model_family", choices=["mlp", "random_forest"])

    args = parser.parse_args()

    if args.command == "train":
        if args.mode == "standard" and args.model_family == "mlp":
            run_standard_mlp()
        else:
            raise NotImplementedError(
                f"train {args.mode} {args.model_family} isn't implemented yet "
                "(phase 1 only covers standard mlp)"
            )


if __name__ == "__main__":
    main()
