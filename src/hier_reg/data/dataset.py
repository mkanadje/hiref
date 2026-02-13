"""
Pytorch dataset class for hierarchical forecasting model.
Receives pre-processed data from DataPreprocessor and provides samples for the DataLoader
"""

import torch
from torch.utils.data import Dataset


class HierarchicalDataset(Dataset):
    def __init__(self, hierarchy_ids, features, targets, keys):
        """
        Args:
            hierarchy_ids: dict of {col_name: numpy array of hierarchical id}
            features: numpy array of shape (n_samples, n_features)
            targets: numpy array of targets (n_samples,)
            keys: numpy array of SKU keys
        """
        super().__init__()
        self.hierarchy_ids = {
            col: torch.tensor(ids, dtype=torch.long)
            for col, ids in hierarchy_ids.items()
        }
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.keys = keys

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        hierarchy_ids = {
            col: self.hierarchy_ids[col][idx] for col in self.hierarchy_ids.keys()
        }
        features = self.features[idx]
        target = self.targets[idx]
        key = self.keys[idx]
        return hierarchy_ids, features, target, key
