# src/training/train_ppi.py

import torch
import torch.optim as optim
from src.models.PPIModel import PPIModel
from src.utils.data_loader import load_ppi_data

def train_ppi():
    model = PPIModel()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    data_loader = load_ppi_data('data/ppi_data.csv')

    for epoch in range(10):
        for seq1, seq2, labels in data_loader:
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(seq1, seq2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train_ppi()
