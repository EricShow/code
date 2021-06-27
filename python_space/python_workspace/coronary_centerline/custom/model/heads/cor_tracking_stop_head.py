import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss


@HEADS.register_module
class CoronaryTrackingStop_Head(nn.Module):

    def __init__(self):
        super(CoronaryTrackingStop_Head, self).__init__()
        self.loss_func = torch.nn.BCEWithLogitsLoss()
    def forward(self, inputs):
        # TODO: 定制forward网络
        return inputs

    def loss(self, input, target):
        # TODO: 定制Loss，CoronarySeg_Network中调用
        end_head = input

        loss_end = self.loss_func(end_head, target)
        loss_end = loss_end.mean()
        return {'loss_end': loss_end * 3}
