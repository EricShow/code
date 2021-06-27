import math
import os
import random

import torch
import torch.nn as nn
import torchgeometry as tgm
import numpy as np
from starship.umtf.common.dataset import PIPELINES


@PIPELINES.register_module
class AssignTarget(nn.Module):

    def __init__(self, patch_size: tuple, predict_radius: int):
        super(AssignTarget, self).__init__()
        self._patch_size = patch_size
        self._predict_radius = predict_radius
        # self._predict_cube_edge = self._get_predict_cube_edge()

    def _get_predict_cube_edge(self):
        cube = np.zeros(self._patch_size, dtype=np.float32)
        patch_size = np.array(self._patch_size)
        patch_center = patch_size // 2

        p_s = patch_center - self._predict_radius
        p_e = patch_center + self._predict_radius + 1
        cube[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = 1

        p_s = patch_center - self._predict_radius + 1
        p_e = patch_center + self._predict_radius
        cube[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = 0
        cube = torch.from_numpy(cube)
        return cube

    def forward(self, data):
        img = data['img']
        batch_size = img.size(0)

        mask = img[:, 1:2]
        sk_seg = img[:, 2:3]
        img = img[:, 0:1]

        img = (img - 128.0) / 128.0
        mask = (mask > 0.5) * 1.0
        sk_seg = (sk_seg > 0.5) * 1.0
        cube_edge = self._predict_cube_edge.cuda(img.device)

        batch_cube_edge = torch.cat([cube_edge[None, None]] * batch_size, dim=0)
        img = torch.cat([img, batch_cube_edge], dim=1)
        return img.detach(), mask.detach(), sk_seg.detach(), cube_edge.detach()
