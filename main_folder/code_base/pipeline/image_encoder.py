import torch.nn as nn
import torch.nn.functional as F
from code_base.pipeline.ArcFace import ArcMarginProduct


class ImgEncoder(nn.Module):
    def __init__(self, out_features, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)
        self.out_features = out_features
        self.arcface = ArcMarginProduct(
            in_features=self.backbone.num_features,
            out_features=self.out_features
        )
        self.prelu = nn.PReLU()

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = F.normalize(features)
        if labels is not None:
            return self.arcface(features, labels)
        return features


