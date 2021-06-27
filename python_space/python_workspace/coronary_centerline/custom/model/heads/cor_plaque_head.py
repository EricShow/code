import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss


@HEADS.register_module
class CoronaryPlaqueHead(nn.Module):

    def __init__(self, in_channels, scale_factor=1):
        super(CoronaryPlaqueHead, self).__init__()
        # TODO: 定制Head模型
        self.conv1 = nn.Conv3d(in_channels, 1, 1)
        self.scale_factor = scale_factor
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)


    def forward(self, inputs):
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode='trilinear')
        binary_pre = self.conv1(inputs)
        return binary_pre

    def loss(self, inputs, target):
        # TODO: 定制Loss，CoronarySeg_Network中调用

        plaque_pred = inputs
        plaque_seg, seg = target
        with torch.no_grad():
            restore_thresh = 0.4
            seg = (seg > restore_thresh) * 1.0
            tp_seg = (plaque_seg > restore_thresh) * 1.0

            #affect_range = 5
            #mask_affect_zone = torch.max_pool3d(tp_seg, kernel_size=affect_range, stride=1, padding=affect_range // 2)

            #fp_seg = seg * (1 - mask_affect_zone)
            fp_seg = seg * (1 - tp_seg)

            tp_count = tp_seg.sum()
            fp_count = fp_seg.sum()
            if fp_count < 10 or tp_count < 10:
                tp_count = 1
                fp_count = 1
                tp_seg = 0.0 * tp_seg
                fp_seg = 0.0 * fp_seg
            #
            # affect_range = 15
            # mask_affect_zone = torch.max_pool3d(mask, kernel_size=affect_range, stride=1, padding=affect_range // 2)[:,
            #                    0]
            # # mask_affect_zone_count = mask_affect_zone.sum()
            # mask_weight = mask_weight * mask_affect_zone * 4 + 1

        loss_pla = self.loss_func(plaque_pred, tp_seg)
        loss_pla = (tp_seg * loss_pla).sum() / tp_count  + (fp_seg * loss_pla).sum() / fp_count * 10
        return {'loss_pla': loss_pla}

