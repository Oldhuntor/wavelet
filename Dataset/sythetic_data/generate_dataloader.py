import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_length, stride=1):
        """
        Args:
            df: Single pandas DataFrame from GutenTAG
            seq_length: Length of each sequence window
            stride: Step size between consecutive windows (default=1)
        """
        self.seq_length = seq_length
        self.stride = stride
        self.sequences = []
        self.labels = []

        values = df['value-0'].values
        anomalies = df['is_anomaly'].values

        # Create sliding windows
        for i in range(0, len(values) - seq_length + 1, stride):
            seq = values[i:i + seq_length]
            anomaly_window = anomalies[i:i + seq_length]

            # If any point in the sequence is an anomaly, label the whole sequence as anomaly
            label = 1 if np.any(anomaly_window) else 0

            self.sequences.append(seq)
            self.labels.append(label)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)  # Changed to int64 for Long

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Shape: (seq_length,) -> (1, seq_length) for (channel, seq_length)
        sequence = torch.tensor(self.sequences[idx]).unsqueeze(0)
        # Shape: scalar (will become (batch_size,) after batching)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


def create_dataloaders(dataframes, seq_length, batch_size, train_ratio=0.8, stride=1, shuffle=True):
    """
    Create train and test DataLoaders for each dataframe.

    Args:
        dataframes: List of pandas DataFrames from GutenTAG
        seq_length: Length of each sequence window
        batch_size: Number of sequences per batch
        train_ratio: Ratio of data to use for training (default=0.8)
        stride: Step size between consecutive windows (default=1)
        shuffle: Whether to shuffle training data (default=True)

    Returns:
        train_loaders: List of training DataLoaders
        test_loaders: List of testing DataLoaders
    """
    train_loaders = []
    test_loaders = []

    for df in dataframes:
        df = df.timeseries
        # Calculate split point
        split_idx = int(len(df) * train_ratio)

        # Split dataframe
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        # Create datasets
        train_dataset = TimeSeriesDataset(train_df, seq_length=seq_length, stride=stride)
        test_dataset = TimeSeriesDataset(test_df, seq_length=seq_length, stride=stride)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


def get_gutenTAG_loaders(dataframes, seq_length, batch_size, train_ratio=0.8, shuffle=True, stride=10):
    train_loaders, test_loaders = create_dataloaders(
        dataframes,
        seq_length=seq_length,
        batch_size=batch_size,
        train_ratio=train_ratio,
        stride=stride,
        shuffle=shuffle
    )
    return train_loaders, test_loaders


if __name__ == '__main__':

    # Usage example
    dataframes = DATASETS  # Your list of GutenTAG dataframes
    seq_length = 100
    batch_size = 32
    train_ratio = 0.8

    # Create dataloaders
    train_loaders, test_loaders = create_dataloaders(
        dataframes,
        seq_length=seq_length,
        batch_size=batch_size,
        train_ratio=train_ratio,
        stride=1,
        shuffle=True
    )

    # Usage: iterate through each dataloader
    for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        print(f"\nDataset {i}:")
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Example: iterate through training data
        for sequences, labels in train_loader:
            print(f"Train batch shape: {sequences.shape}")  # (batch_size, seq_length, 1)
            print(f"Train labels shape: {labels.shape}")  # (batch_size, seq_length)
            break

        # Example: iterate through test data
        for sequences, labels in test_loader:
            print(f"Test batch shape: {sequences.shape}")
            print(f"Test labels shape: {labels.shape}")
            break