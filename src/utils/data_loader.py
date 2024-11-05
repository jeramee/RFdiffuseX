# src/utils/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class PPIDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq1 = self.data.iloc[idx, 0]
        seq2 = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        return seq1, seq2, label

def load_ppi_data(csv_file, batch_size=32):
    dataset = PPIDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
