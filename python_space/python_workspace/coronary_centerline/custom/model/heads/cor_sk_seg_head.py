import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss


@HEADS.register_module
class CoronarySkSegHead(nn.Module):

    def __init__(self, in_channels, scale_factor=1):
        super(CoronarySkSegHead, self).__init__()
        # TODO: 定制Head模型
        self.conv1 = nn.Conv3d(in_channels, 1, 1)
        self.scale_factor = scale_factor
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)


    def forward(self, inputs):
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode='trilinear')
        binary_pre = self.conv1(inputs)
        return binary_pre

    def loss(self, inputs, mask):
        # TODO: 定制Loss，CoronarySeg_Network中调用
        with torch.no_grad():
            mask = (mask > 0.4) * 1.0
            mask_weight = mask * 2 + 1.0
            mask_weight = mask_weight[:, 0]

            affect_range = 15
            mask_affect_zone = torch.max_pool3d(mask, kernel_size=affect_range, stride=1, padding=affect_range//2)[:, 0]
            #mask_affect_zone_count = mask_affect_zone.sum()
            mask_weight = mask_weight * mask_affect_zone * 4 + 1

        loss = self.loss_func(inputs, mask)
        #loss = self._soft_label_loss(inputs, mask)
        loss = (loss * mask_weight).mean()
        return {'loss': loss * 100}

    def _soft_label_loss(self, input, target):
        log_likelihood = -F.log_softmax(input, dim=1)
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
        return loss
