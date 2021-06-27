import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


@HEADS.register_module
class CoronaryHeatmapHead(nn.Module):
    def __init__(self, in_channels, scale_factor=1):
        super(CoronaryHeatmapHead, self).__init__()
        # TODO: 定制Head模型
        self.conv_cv = self._get_deep_head(in_channels, 1, depth=1)
        self.scale_factor = scale_factor
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    def _get_deep_head(self, channels, pred_count, depth=3):
        return nn.Sequential(*([SingleConv(channels, channels)] * depth + [nn.Conv3d(channels, pred_count, 1)]))

    def forward(self, inputs):
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="trilinear")

        cv_pred = self.conv_cv(inputs)
        return cv_pred

    def loss(self, inputs, target):
        # TODO: 定制Loss，CoronarySeg_Network中调用

        cv_pred = inputs
        cor_mask, vein_mask, sk_seg = target
        with torch.no_grad():
            restore_thresh1 = 0.3
            restore_thresh2 = 0.6
            cor_mask = (cor_mask > restore_thresh1) * 1.0
            vein_mask = (vein_mask > restore_thresh1) * 1.0
            sk_seg = (sk_seg > restore_thresh2) * 1.0

            cor_count = cor_mask.sum()
            vein_count = vein_mask.sum()

            cor_weight = (sk_seg * cor_mask) * 5 + cor_mask
            vein_weight = (sk_seg * vein_mask) * 5 + vein_mask

        loss_cv = self.loss_func(cv_pred, cor_mask)
        loss_cor = (cor_mask * loss_cv).sum() / (cor_count + 5)
        loss_vein = (vein_mask * loss_cv).sum() / (vein_count + 5)
        return {"loss_cor": loss_cor, "loss_vein": loss_vein}

    def _soft_label_loss(self, input, target):
        log_likelihood = -F.log_softmax(input, dim=1)
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
        return loss
