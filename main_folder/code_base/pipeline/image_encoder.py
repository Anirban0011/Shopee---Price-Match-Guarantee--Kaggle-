import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import ScaledStdConv2d, ScaledStdConv2dSame, BatchNormAct2d
from code_base.utils import ArcMarginProduct, ArcModule

class ImgEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_size=1792,
        backbone=None,
        pretrained=True,
        arcmargin=0.5,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embed_size = embed_size  # embedding size
        self.out_features = num_classes  # num classes
        self.margin = arcmargin

        self.arcface = ArcMarginProduct(
            in_features=self.backbone.num_features,
            out_features=self.out_features,
            m=self.margin,
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(self.backbone.num_features)

    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        features = F.normalize(features)
        if labels is not None:
            features = self.arcface(features, labels)
        return features
