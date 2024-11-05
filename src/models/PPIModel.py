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
