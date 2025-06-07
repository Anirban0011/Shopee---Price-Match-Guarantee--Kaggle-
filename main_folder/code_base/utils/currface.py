import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class CurricularFace(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        m=0.5,
        s=64.0,
        alpha=0.0,
    ):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.alpha = alpha
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer("t", torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def update_margin(self, epoch):
        self.m = self.m + (self.alpha*(epoch+1))
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        print(f"margin updated to : {self.m}")
        return None

    def forward(self, embbedings, label):
        cos_theta = F.linear(embbedings, F.normalize(self.kernel))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        # with torch.no_grad():
        #     origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (
            target_logit * self.cos_m - sin_theta * self.sin_m
        )  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(
            target_logit > self.threshold, cos_theta_m, target_logit - self.mm
        )

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output

    # , origin_cos * self.s
