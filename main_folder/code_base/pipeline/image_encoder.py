import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import ScaledStdConv2d, ScaledStdConv2dSame, BatchNormAct2d
from code_base.utils import ArcMarginProduct


class ImgEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_size=1792,
        backbone=None,
        pretrained=True,
        arcmargin=0.5,
        alpha=1e-4,
        use_dynamic_margin=False,
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embed_size = embed_size  # embedding size
        self.out_features = num_classes  # num classes
        self.margin = arcmargin
        self.use_dynamic_margin = use_dynamic_margin
        self.alpha = alpha

        if "nfnet" in backbone:
            self.backbone._modules["final_conv"] = nn.Conv2d(
                self.backbone._modules["conv_head"].in_channels,
                self.embed_size,
                kernel_size=(1, 1),
                stride=(1, 1),
            )
        #     self.backbone._modules["final_conv"] = ScaledStdConv2dSame(
        #         self.backbone._modules["final_conv"].in_channels,
        #         self.embed_size,
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     )
        # elif "nfnet_l0" in backbone or "nfnet_l1" in backbone:
        #     self.backbone._modules["final_conv"] = ScaledStdConv2d(
        #         self.backbone._modules["final_conv"].in_channels,
        #         self.embed_size,
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     )
        elif any(x in backbone for x in ["b5", "b6", "b7"]):
            self.backbone._modules["conv_head"] = nn.Conv2d(
                self.backbone._modules["conv_head"].in_channels,
                self.embed_size,
                kernel_size=(1, 1),
                stride=(1, 1),
            )
            self.backbone._modules["bn2"] = BatchNormAct2d(
                self.embed_size,
                eps=self.backbone._modules["bn2"].eps,
                affine=self.backbone._modules["bn2"].affine,
                track_running_stats=self.backbone._modules["bn2"].track_running_stats,
                drop_layer=type(self.backbone._modules["bn2"].drop),
                act_layer=type(self.backbone._modules["bn2"].act),
            )

        self.arcface = ArcMarginProduct(
            in_features=self.embed_size,
            out_features=self.out_features,
            m=self.margin,
            use_dynamic_margin=self.use_dynamic_margin,
            alpha=self.alpha,
        )
        self.bn = nn.BatchNorm1d(self.embed_size)

    def forward(self, x, labels=None, epoch=0):
        features = self.backbone.forward_features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        features = F.normalize(features)
        if labels is not None:
            features = self.arcface(features, labels, epoch)
        return features
