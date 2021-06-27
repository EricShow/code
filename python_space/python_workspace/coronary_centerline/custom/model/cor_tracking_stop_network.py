import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common import build_pipelines
from starship.umtf.common.model import NETWORKS, build_backbone, build_head


@NETWORKS.register_module
class CoronaryTrackingStop_Network(nn.Module):

    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(CoronaryTrackingStop_Network, self).__init__()

        # TODO: 定制网络
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()
        self._show_count = 0

    @torch.jit.ignore
    def forward(self, img0, img1, img2, label):
        # save_dir = './trail_data'
        # import os
        # import SimpleITK as sitk
        # show_img = img0
        # for idx in range(show_img.size(0)):
        #     save_prefix = os.path.join(save_dir, '%04d' % self._show_count)
        #     self._show_count += 1
        #
        #     vol_p0 = (show_img[idx, 0].cpu().numpy() * 255).astype(np.uint8)
        #
        #     img = sitk.GetImageFromArray(vol_p0)
        #     sitk.WriteImage(img, save_prefix + '-vol.nii.gz')

        outs = self.backbone(img0, img1, img2)
        head_outs = self.head(outs)
        loss = self.head.loss(head_outs[1], label)
        return loss

    @torch.jit.export
    def forward_test(self, img0, img1, img2):
        # TODO: 根据需求适配，python3.7 custom/utils/save_torchscript.py保存静态图时使用

        outs = self.backbone(img0, img1, img2)
        # head_outs = self.head(outs)
        _, ts_head = outs
        ts_head = torch.sigmoid(ts_head)
        return ts_head

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
