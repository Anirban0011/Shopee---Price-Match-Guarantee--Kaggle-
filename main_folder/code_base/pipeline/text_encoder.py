import torch.nn as nn
import torch.nn.functional as F
from code_base.pipeline.ArcFace import ArcMarginProduct


class TextEncoder(nn.Module):
    def __init__(self, out_features, Tokenizer, backbone):
        super().__init__()
        self.tokenizer = Tokenizer
        self.backbone = backbone
        self.out_features = out_features
        self.in_features = self.backbone.pooler.dense.in_features
        self.arcface = ArcMarginProduct(
            in_features=self.in_features,
            out_features=self.out_features
        )
        self.linear = nn.Linear(in_features=self.in_features, out_features=1024)

    def forward(self, x, labels=None):
        features = self.tokenizer(x, return_tensors="pt")
        features = self.backbone(**features).pooler_output
        features = self.linear(features)
        features = F.normalize(features)
        if labels is not None:
            return self.arcface(features, labels)
        return features
