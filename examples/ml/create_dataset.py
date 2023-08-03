import multiprocessing
from itertools import repeat
from pathlib import Path

import hrvanalysis as hrv
import numpy as np
import pandas as pd

import iridia_af.hyperparameters as hp
import config as cfg
from iridia_af.record import Record


def main():
    metadata_df = pd.read_csv(hp.METADATA_PATH)
    list_record_path = metadata_df["record_id"].values
    with multiprocessing.Pool(hp.NUM_PROC) as pool:
        all_windows = pool.starmap(get_record_windows, zip(list_record_path, repeat(metadata_df)))

    new_all_windows = []
    for windows in all_windows:
        new_all_windows.extend(windows)

    df_features = pd.DataFrame(new_all_windows)
    new_df_path = Path(hp.DATASET_PATH, f"dataset_hrv_{cfg.WINDOW_SIZE}_{cfg.TRAINING_STEP}.csv")
    df_features.to_csv(new_df_path, index=False)
    print(f"Saved dataset to {new_df_path}")


def get_record_windows(record_id, metadata_df):
    metadata_record = metadata_df[metadata_df["record_id"] == record_id]
    metadata_record = metadata_record.values[0]
    record_path = Path(hp.RECORDS_PATH, record_id)
    record = Record(record_path, metadata_record)
    record.load_rr()
    record_windows = []
    for day_index in range(record.num_days):
        for i in range(0, len(record.rr[day_index]) - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
            rr_window = record.rr[day_index][i:i + cfg.WINDOW_SIZE]
            label_window = record.rr_labels[day_index][i:i + cfg.WINDOW_SIZE]
            if np.sum(label_window) == 0:
                label = 0
            else:
                label = 1
            features = get_hrv_metrics(rr_window)
            features["patient"] = record.metadata.patient_id
            features["record"] = record.metadata.record_id
            features["label"] = label
            record_windows.append(features)
    print(f"Finished record {record_id}")
    return record_windows


def get_hrv_metrics(rr_window: np.ndarray):
    time_domain_features = hrv.get_time_domain_features(rr_window)
    frequency_domain_features = hrv.get_frequency_domain_features(rr_window)
    geometrical_features = hrv.get_geometrical_features(rr_window)
    poincare_features = hrv.get_poincare_plot_features(rr_window)

    all_features = {**time_domain_features, **frequency_domain_features, **geometrical_features, **poincare_features}
    # remove tinn
    del all_features["tinn"]
    return all_features


if __name__ == '__main__':
    main()
