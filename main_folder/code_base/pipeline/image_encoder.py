import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import ScaledStdConv2d, ScaledStdConv2dSame, BatchNormAct2d
from code_base.utils import ArcMarginProduct, CurricularFace


class ImgEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_size=1792,
        backbone=None,
        pretrained=True,
        scale=30.0,
        margin=0.5,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embed_size = embed_size  # embedding size
        self.num_classes = num_classes  # num classes
        self.margin = margin
        self.scale = scale

        self.fc = nn.Linear(self.backbone.num_features, self.embed_size)

        self.final = CurricularFace(
            in_features=self.embed_size,
            out_features=self.num_classes,
            s=self.scale,
            m=self.margin,
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm1d(self.embed_size)

    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        features = F.normalize(features)
        if labels is not None:
            features = self.final(features, labels)
        return features
