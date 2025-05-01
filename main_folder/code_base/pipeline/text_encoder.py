import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from code_base.utils import ArcMarginProduct


class TextEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_size=1024,
        max_seq_length=35,
        backbone=None,
        arcmargin=0.5,
        alpha=1e-4,
        use_dynamic_margin=False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        self.out_features = num_classes
        self.embed_size = embed_size
        self.margin = arcmargin
        self.alpha = alpha
        self.use_dynamic_margin = use_dynamic_margin
        self.arcface = ArcMarginProduct(
            in_features=self.embed_size,
            out_features=self.out_features,
            m=self.margin,
            use_dynamic_margin=self.use_dynamic_margin,
            alpha=self.alpha,
        )
        self.pool = nn.AvgPool1d(kernel_size=max_seq_length)
        self.bn = nn.BatchNorm1d(self.embed_size)

    def forward(self, input_ids, attention_mask, labels=None, epoch=0):
        if any(x in self.backbone_name for x in ["bert-base", "roberta-base"]):
            features = self.backbone(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            ).hidden_states[-2:]
            features = torch.cat([features[-1], features[-2]], dim=-1)
            features = features[:, :, : self.embed_size]
        else:
            features = self.backbone(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state
        features = features.transpose(1, 2)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        features = self.bn(features)
        features = F.normalize(features)
        if labels is not None:
            return self.arcface(features, labels, epoch)
        return features
