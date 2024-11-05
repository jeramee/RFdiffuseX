# RFdiffuseX
## Project Overview

**RFdiffuseX** is an enhanced version of RFdiffusion that integrates advanced protein-protein interaction (PPI) prediction functionalities inspired by RoseTTAFold2-PPI while retaining RFdiffusion's strengths in protein structure generation. This expanded tool offers comprehensive applications in protein design, such as motif scaffolding, symmetric protein design, and PPI analysis for binding interactions—all achieved without copying or directly using code from RoseTTAFold2-PPI.

## Key Features

- **Protein-Protein Interaction Prediction**: New PPI modules predict and analyze protein interaction interfaces, adding a layer of functionality ideal for studying binding networks.
- **Enhanced Motif Scaffolding and Symmetry**: Builds on RFdiffusion’s capabilities to scaffold motifs, generate symmetric designs, and allow for guided diversification around specific motifs or structures.
- **Data Compatibility**: Expanded to process PPI-focused datasets alongside RFdiffusion’s data formats, supporting complex structural analysis and interaction predictions.

## Planned Fork

This expansion will be developed as an official fork of RFdiffusion, ensuring compatibility with the original tool and enabling future updates from the community. Released under the BSD license, RFdiffuseX will support broad adoption and contributions while maintaining ethical alignment with open-source standards.

## File Structure

```yml
RFdiffuseX/
├── examples/
│   ├── expected_output/           # Test outputs
│   ├── test.list                  # Test file for PPI and diffusion
│   ├── test_msas/                 # Test multiple sequence alignments
├── src/
│   ├── data/                      # Datasets for PPI and RFdiffusion
│   ├── models/                    # Model architectures
│   │   ├── RFdiffusionModel.py    # Original diffusion model
│   │   └── PPIModel.py            # New PPI prediction model
│   ├── utils/                     # Helper functions
│   │   ├── preprocess.py          # Data preprocessing
│   │   ├── data_loader.py         # Updated data loader
│   │   └── evaluation.py          # PPI evaluation metrics
│   ├── training/                  # Training scripts
│   │   ├── train_diffusion.py     # RFdiffusion training script
│   │   ├── train_ppi.py           # PPI prediction training script
│   │   └── loss_functions.py      # Loss functions for both tasks
│   ├── config/                    # Configuration files
│   │   ├── diffusion_config.yaml  # Config for RFdiffusion tasks
│   │   └── ppi_config.yaml        # Config for PPI tasks
│   └── main.py                    # Main entry point
├── LICENSE
├── README.md                      # Documentation
└── requirements.txt               # Dependencies
```

## Example Code Snippets
### PPI Model (PPIModel.py)

This model uses attention mechanisms to focus on protein interaction interfaces, enabling binary classification for interactions.

```python
# src/models/PPIModel.py
import torch
import torch.nn as nn
from transformers import BertModel

class PPIModel(nn.Module):
    def __init__(self):
        super(PPIModel, self).__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.fc = nn.Linear(768, 1)

    def forward(self, seq1, seq2):
        emb1 = self.bert(seq1)['last_hidden_state']
        emb2 = self.bert(seq2)['last_hidden_state']
        combined_emb = torch.cat((emb1, emb2), dim=1)
        attention_output, _ = self.attention(combined_emb, combined_emb, combined_emb)
        return torch.sigmoid(self.fc(attention_output.mean(dim=1)))
```

### Data Loader (data_loader.py)

This data loader supports both RFdiffusion and PPI data formats, loading protein pairs for interaction analysis.

```python
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
```

### Training Script for PPI (train_ppi.py)

Trains the PPI model using binary classification to predict interactions.

```python
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
```

### Main Script (main.py)

Allows users to run both diffusion and PPI prediction tasks.

```python
# src/main.py
import argparse
from src.training.train_diffusion import train_diffusion
from src.training.train_ppi import train_ppi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFdiffuseX: Expanded RFdiffusion with PPI Prediction")
    parser.add_argument("--task", choices=["diffusion", "ppi"], required=True, help="Task to perform")
    args = parser.parse_args()

    if args.task == "diffusion":
        train_diffusion()
    elif args.task == "ppi":
        train_ppi()
```

## Expanded RFdiffusion Functionalities

- **Motif Scaffolding**: Scaffold motifs within proteins for applications in structural biology.
- **Unconditional Protein Generation**: Generate proteins of specific lengths with no structural constraints.
- **Symmetric Protein Design**: Includes cyclic, dihedral, and tetrahedral symmetry configurations.
- **Binder Design and Partial Diffusion**: Control protein interface design for tailored binding characteristics.

## Setup and Usage

1. **Clone the Repository**:
```bash
git clone https://github.com/RosettaCommons/RFdiffuseX.git
cd RFdiffuseX
```

2. **Install Dependencies**:

```bash

conda create -f env/SE3nv.yml
conda activate SE3nv
pip install -r requirements.txt
```

3. **Download Model Weights**:

```bash

mkdir models && cd models
# Download weights using provided URLs
```

4. **Run Tasks**:

```bash

    # Run diffusion tasks
    python src/main.py --task diffusion
    # Run PPI tasks
    python src/main.py --task ppi
```

For further instructions and examples, refer to the README.md file.

RFdiffuseX is a powerful, open-source tool for protein design and interaction analysis, intended for diverse research in structural biology and drug discovery.
