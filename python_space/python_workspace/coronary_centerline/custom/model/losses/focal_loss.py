import torch
from starship.umtf.common.model import LOSSES
from torch import nn


@LOSSES.register_module
class Binary_Focal_Loss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean', epsilon=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, input, target):
        pt = torch.sigmoid(input)
        pt = torch.clamp(pt, self.epsilon, 1 - self.epsilon)
        alpha = self.alpha
        loss = -alpha * (1 - pt)**self.gamma * target * torch.log(pt) - (1 - alpha) * pt**self.gamma * (
            1 - target
        ) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
