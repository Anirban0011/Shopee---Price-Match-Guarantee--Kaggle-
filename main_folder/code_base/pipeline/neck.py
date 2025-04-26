import timm
import torch.nn as nn
import torch.nn.functional as F

class Neck1:
    def __init__(self, in_features, channel_size, dropout=0.5):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc1 = nn.Linear(in_features * 16 * 16 , channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn2(x)
        return x

