import numpy as np
import torch
from torch.utils.data import Dataset

class ISLDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = torch.tensor(np.load(X_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(y_path), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]