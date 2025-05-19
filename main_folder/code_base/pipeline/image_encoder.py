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
        scale=30.0,
        margin=0.5,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embed_size = embed_size  # embedding size
        self.num_classes = num_classes  # num classes
        self.margin = margin
        self.scale = scale

        self.final_conv = nn.Conv2d(self.backbone.num_features,
                                    self.embed_size,
                                    kernel_size=1,
                                    )

        self.fc1 = nn.Linear(self.backbone.num_features, self.embed_size)

        self.final = CurricularFace(
            in_features=self.embed_size,
            out_features=self.num_classes,
            s=self.scale,
            m=self.margin,
        )
        self.gem = GeM()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.bn = nn.BatchNorm1d(self.embed_size)
        self.prelu = nn.PReLU()

    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        features = self.final_conv(features)
        features = self.gem(features)
        features = features.view(features.size(0), -1)
        # features = self.layernorm(features)
        # features = self.fc1(features)
        features = self.bn(features)
        # features = self.prelu(features)
        # features = F.normalize(features)
        if labels is not None:
            # return feat with and without margin
            features, _ = self.final(features, labels)
        return features, _
