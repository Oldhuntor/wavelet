from torch.utils.data import Dataset, DataLoader
import torch


class MorletDataset(Dataset):
    def __init__(self, amplitude, phase, labels):
        self.amplitude = amplitude
        self.phase = phase
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        amp = self.amplitude[idx]     # (5, 576)
        pha = self.phase[idx]         # (5, 576)
        lab = self.labels[idx]
        return amp, pha, lab


