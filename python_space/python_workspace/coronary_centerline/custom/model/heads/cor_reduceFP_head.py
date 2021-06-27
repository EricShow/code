import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss


@HEADS.register_module
class CoronaryReducefpHead(nn.Module):

    def __init__(self):
        super(CoronaryReducefpHead, self).__init__()
        # TODO: 定制Head模型
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        # TODO: 定制forward网络
        return inputs

    def loss(self, input, target):
        # TODO: 定制Loss，CoronarySeg_Network中调用
        loss_fp = self._soft_label_loss(input, target)
        loss_fp = loss_fp.mean()
        return {'loss_fp': loss_fp}

    def _soft_label_loss(self, input, target):
        #print(input)
        log_likelihood = -F.log_softmax(input, dim=1)
        #print(log_likelihood.shape)
        #print(target.shape)
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
        return loss
