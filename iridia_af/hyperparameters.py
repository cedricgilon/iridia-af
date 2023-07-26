import multiprocessing
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.resolve()
DATA_PATH = Path(ROOT_PATH, "data")
DATASET_PATH = Path(DATA_PATH, "datasets")

LOG_DL_PATH = Path(ROOT_PATH, "examples", "dl", "log")

RECORDS_PATH = Path(DATA_PATH, "records")
assert RECORDS_PATH.exists(), f"RECORDS_PATH does not exist: {RECORDS_PATH}"
METADATA_PATH = Path(DATA_PATH, "metadata.csv")
assert METADATA_PATH.exists(), f"METADATA_PATH does not exist: {METADATA_PATH}"

NUM_PROC = multiprocessing.cpu_count() - 2

SAMPLE_RATE = 200
