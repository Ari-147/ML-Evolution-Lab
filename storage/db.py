import json
import os
import sqlite3
from datetime import datetime, timezone

import config


def get_connection():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            model_family TEXT NOT NULL,
            mode TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            config_snapshot TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            train_accuracy REAL,
            val_accuracy REAL,
            timestamp TEXT NOT NULL,
            npz_path TEXT NOT NULL,
            phase_id INTEGER,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS error_clusters (
            run_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            member_example_ids TEXT NOT NULL,
            PRIMARY KEY (run_id, step, cluster_id)
        )
        """
    )
    conn.commit()
    conn.close()


def create_run(run_id, model_family, mode, dataset_name, config_snapshot):
    conn = get_connection()
    conn.execute(
        "INSERT INTO runs (run_id, model_family, mode, dataset_name, created_at, config_snapshot) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            run_id,
            model_family,
            mode,
            dataset_name,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(config_snapshot),
        ),
    )
    conn.commit()
    conn.close()


def list_runs():
    conn = get_connection()
    rows = conn.execute(
        "SELECT run_id, model_family, mode, dataset_name, created_at "
        "FROM runs ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows


def list_snapshots(run_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT snapshot_id, step, train_accuracy, val_accuracy, npz_path, phase_id "
        "FROM snapshots WHERE run_id = ? ORDER BY step",
        (run_id,),
    ).fetchall()
    conn.close()
    return rows


def create_snapshot(run_id, step, train_accuracy, val_accuracy, npz_path, phase_id=None):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO snapshots (run_id, step, train_accuracy, val_accuracy, timestamp, npz_path, phase_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            step,
            train_accuracy,
            val_accuracy,
            datetime.now(timezone.utc).isoformat(),
            npz_path,
            phase_id,
        ),
    )
    conn.commit()
    snapshot_id = cur.lastrowid
    conn.close()
    return snapshot_id


def save_error_clusters(run_id, step, clusters):
    conn = get_connection()
    conn.execute("DELETE FROM error_clusters WHERE run_id = ? AND step = ?", (run_id, step))
    for cluster_id, members in clusters.items():
        conn.execute(
            "INSERT INTO error_clusters (run_id, step, cluster_id, member_example_ids) VALUES (?, ?, ?, ?)",
            (run_id, step, cluster_id, json.dumps(members)),
        )
    conn.commit()
    conn.close()


def list_error_clusters(run_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT step, cluster_id, member_example_ids FROM error_clusters WHERE run_id = ? ORDER BY step, cluster_id",
        (run_id,),
    ).fetchall()
    conn.close()
    return rows
