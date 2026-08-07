import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "ml_lab.db")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")

MAX_EPOCHS = 50
MAX_TREES = 100
TREES_PER_STEP = 5

SNAPSHOT_DENSE_STEPS = 20
SNAPSHOT_EVERY_N = 5

PCA_COMPONENTS = 2
GRID_RESOLUTION = 100

ERROR_CLUSTER_K_RANGE = (2, 6)

PERMUTATION_N_REPEATS = 5