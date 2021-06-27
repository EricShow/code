import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss


@HEADS.register_module
class CoronaryDirectionHead(nn.Module):

    def __init__(self):
        super(CoronaryDirectionHead, self).__init__()
        # TODO: 定制Head模型
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        # TODO: 定制forward网络
        return inputs

    def loss(self, inputs, targets):
        # TODO: 定制Loss，CoronarySeg_Network中调用
        dirction_head, joint_head = inputs
        direct_target, joint_target = targets
        loss_direct = self._soft_label_loss(dirction_head, direct_target)
        loss_direct = loss_direct.mean()
        loss_joint = self.loss_func(joint_head, joint_target)
        loss_joint = loss_joint.mean()
        return {'loss_joint': loss_joint * 3, 'loss_direct': loss_direct * 0.3}

    def _soft_label_loss(self, input, target):
        #print(input)
        log_likelihood = -F.log_softmax(input, dim=1)
        #print(log_likelihood.shape)
        #print(target.shape)
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)
        return loss
