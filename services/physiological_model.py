"""V2 physiological architecture, shared with the included WESAD training script."""

import torch
from torch import nn


class PhysiologicalLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=1,
                 bidirectional=False, num_classes=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        state = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        return self.fc(state)
