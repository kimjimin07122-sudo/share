import os
import glob
import numpy as np
import pandas as pd
from config import Config

# torch is needed only by DroneSensorDataset/get_dataloaders, not by the
# scalers. The onboard streaming detector needs the scalers and runs
# inference through ONNX Runtime, so importing this module must not drag in
# torch -- installing it on a Jetson means matching the JetPack build.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except ImportError:  # inference-only environment
    _TORCH_OK = False
    Dataset = object


class MinMaxScaler:
    def fit(self, data):
        data = np.asarray(data, dtype=np.float32)
        self.data_min_ = np.min(data, axis=0)
        self.data_max_ = np.max(data, axis=0)
        data_range = self.data_max_ - self.data_min_
        safe_range = np.where(data_range == 0, 1.0, data_range)
        self.scale_ = 1.0 / safe_range
        self.min_ = -self.data_min_ * self.scale_
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, data):
        data = np.asarray(data, dtype=np.float32)
        return data * self.scale_ + self.min_

    def fit_transform(self, data):
        return self.fit(data).transform(data)


class StandardScaler:
    def fit(self, data):
        data = np.asarray(data, dtype=np.float32)
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, data):
        data = np.asarray(data, dtype=np.float32)
        return (data - self.mean_) / self.std_

    def fit_transform(self, data):
        return self.fit(data).transform(data)


def build_scaler():
    method = Config.SCALING_METHOD.lower()
    if method in {"minmax", "min_max", "normalization"}:
        return MinMaxScaler()
    if method in {"standard", "standardization", "zscore", "z_score"}:
        return StandardScaler()
    raise ValueError(f"Unsupported scaling method: {Config.SCALING_METHOD}")


class DroneSensorDataset(Dataset):
    def __init__(self, data_source, scaler=None, is_train=True):
        if not _TORCH_OK:
            raise ImportError("DroneSensorDataset needs torch; the scalers in this "
                              "module do not, so inference-only installs can skip it.")
        self.seq_len = Config.SAMPLE_LENGTH
        self.stride = Config.STRIDE
        self.is_train = is_train

        if os.path.isdir(data_source):
            raw_sequences = self._load_csv_directory(data_source)
        else:
            raw_sequences = [self._process_single_csv(data_source)]

        if not raw_sequences:
            raise ValueError(f"No usable CSV data found in {data_source}")

        if self.is_train:
            self.scaler = build_scaler()
            self.scaler.fit(np.vstack(raw_sequences))
        else:
            if scaler is None:
                raise ValueError("A fitted scaler is required for evaluation data.")
            self.scaler = scaler

        scaled_sequences = [self.scaler.transform(data) for data in raw_sequences]
        self.samples = self._create_sequences(scaled_sequences)

    def _process_single_csv(self, filepath):
        df = pd.read_csv(filepath, comment="#", header=None)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

        if df.shape[1] > Config.NUM_CHANNELS:
            df = df.iloc[:, -Config.NUM_CHANNELS:]

        if df.shape[1] != Config.NUM_CHANNELS:
            raise ValueError(
                f"{filepath} has {df.shape[1]} usable columns; "
                f"expected {Config.NUM_CHANNELS} sensor channels."
            )

        return df.to_numpy(dtype=np.float32)

    def _load_csv_directory(self, data_dir):
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        return [self._process_single_csv(f) for f in csv_files]

    def _create_sequences(self, sequences):
        samples = []
        for data in sequences:
            for i in range(0, len(data) - self.seq_len + 1, self.stride):
                samples.append(data[i : i + self.seq_len])

        if not samples:
            raise ValueError(
                f"No sequences could be created. Need at least {self.seq_len} rows."
            )

        return np.asarray(samples, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].T
        return torch.tensor(sample, dtype=torch.float32)


def get_dataloaders():
    train_dataset = DroneSensorDataset(Config.TRAIN_DATA_PATH, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    test_loaders = {}
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    for file in test_files:
        test_dataset = DroneSensorDataset(file, scaler=train_dataset.scaler, is_train=False)
        test_loaders[os.path.basename(file)] = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
        )

    return train_loader, test_loaders, train_dataset.scaler
