import timm
import torch.nn as nn
import torch.nn.functional as F
from code_base.pipeline.ArcFace import ArcMarginProduct


class ImgEncoder(nn.Module):
    def __init__(
        self,
        channel_size,
        out_features,
        dropout=0.5,
        backbone="densenet121",
        pretrained= True
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.channel_size = channel_size
        self.in_features = self.backbone.classifier.in_features
        self.out_features = out_features
        self.arcface = ArcMarginProduct(
            in_features=self.channel_size,
            out_features=self.out_features
        )
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 16 * 16 , self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):
        features = self.backbone.features(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)
        if labels is not None:
            return self.arcface(features, labels)
        return features
