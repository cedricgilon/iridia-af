from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import config as cfg
import iridia_af.hyperparameters as hp
from iridia_af.record import create_record


def create_dataset_csv():
    metadata_df = pd.read_csv(hp.METADATA_PATH)
    list_windows = []
    for record_id in tqdm(metadata_df["record_id"].unique()):
        record = create_record(record_id, metadata_df, hp.RECORDS_PATH)
        record.load_ecg()
        for day_index in range(record.metadata.record_n_files):
            len_day = record.ecg[day_index].shape[0]
            for i in range(0, len_day - cfg.WINDOW_SIZE, cfg.TRAINING_STEP):
                label = 1 if np.sum(record.ecg_labels[day_index][i:i + cfg.WINDOW_SIZE]) > 0 else 0
                detection_window = {
                    "patient_id": record.metadata.patient_id,
                    "file": record.ecg_files[day_index],
                    "start_index": i,
                    "end_index": i + cfg.WINDOW_SIZE,
                    "label": label
                }
                list_windows.append(detection_window)

    new_df = pd.DataFrame(list_windows)
    new_df_path = Path(hp.DATASET_PATH, f"dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv")
    new_df.to_csv(Path(new_df_path, index=False))
    print(f"Saved dataset to {Path(hp.DATASET_PATH, f'dataset_detection_ecg_{cfg.WINDOW_SIZE}.csv')}")


class DetectionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dw = self.df.iloc[idx]
        with h5py.File(dw.file, "r") as f:
            key = list(f.keys())[0]
            ecg_data = f[key][dw.start_index:dw.end_index, 0]
        ecg_data = torch.tensor(ecg_data, dtype=torch.float32)
        ecg_data = ecg_data.unsqueeze(0)
        label = torch.tensor(dw.label, dtype=torch.float32)
        label = label.unsqueeze(0)
        return ecg_data, label


if __name__ == "__main__":
    create_dataset_csv()
