import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import ScaledStdConv2d, ScaledStdConv2dSame, BatchNormAct2d
from code_base.utils import ArcMarginProduct, CurricularFace
from .gempool import GeM


class ImgEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_size=1792,
        backbone=None,
        pretrained=True,
        dropout=0.5,
        scale=30.0,
        margin=0.5,
        alpha = 0.0,
        final_layer="arcface",
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embed_size = embed_size  # embedding size
        self.num_classes = num_classes  # num classes
        self.margin = margin
        self.scale = scale

        self.final_conv = nn.Conv2d(
            self.backbone.num_features,
            self.embed_size,
            kernel_size=1,
        )

        self.fc1 = nn.Linear(self.backbone.num_features, self.embed_size)

        if final_layer == "arcface":
            self.final = ArcMarginProduct(
                in_features=self.embed_size,
                out_features=self.num_classes,
                s=self.scale,
                m=self.margin,
                alpha=alpha,
            )

        if final_layer == "currface":
            self.final = CurricularFace(
                in_features=self.embed_size,
                out_features=self.num_classes,
                s=self.scale,
                m=self.margin,
                alpha=alpha,
            )

        self.gem = GeM()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm2d(self.backbone.num_features)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.bn2 = nn.BatchNorm1d(self.embed_size)

    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        features = self.final_conv(features)
        # features = self.gem(features)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        features = self.bn2(features)
        features = F.normalize(features)
        if labels is not None:
            return self.final(features, labels)
        return features
