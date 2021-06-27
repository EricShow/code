import numpy as np
import torch
import torch.nn as nn
from starship.umtf.common import build_pipelines
from starship.umtf.common.model import NETWORKS, build_backbone, build_head


@NETWORKS.register_module
class CorPlaqueNetwork(nn.Module):

    def __init__(
            self,
            backbone,
            head,
            apply_sync_batchnorm=False,
            pipeline=[],
            train_cfg=None,
            test_cfg=None
    ):
        super(CorPlaqueNetwork, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._show_count = 0
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, img, mask):
        with torch.no_grad():
            data = {'img': img, 'mask': mask}
            data = self._pipeline(data)
            img, mask = data['img'], data['mask']
            seg = img[:, 1]
            img = img.detach()

            plaque_seg = mask.detach()

            mask = (plaque_seg, seg)

            # save_dir = './trail_data'
            # import os
            # import SimpleITK as sitk
            # for idx in range(img.size(0)):
            #     save_prefix = os.path.join(save_dir, '%04d' % self._show_count)
            #     self._show_count += 1
            #
            #     si = img[idx, 0].cpu().numpy() * 255
            #     si = si.astype(np.uint8)
            #     si = sitk.GetImageFromArray(si)
            #     sitk.WriteImage(si, save_prefix + '-vol.nii.gz')
            #
            #     si = img[idx, 1].cpu().numpy()
            #     si = (si > 0.4)
            #     ss = si.astype(np.uint8)
            #
            #     si = plaque_seg[idx, 0].cpu().numpy()
            #     si = (si >= 0.4)
            #     si =  ss #si.astype(np.uint8) # + ss
            #     si = sitk.GetImageFromArray(si)
            #     sitk.WriteImage(si, save_prefix + '-seg.nii.gz')

        outs = self.backbone(img)
        head_outs = self.head(outs)
        loss = self.head.loss(head_outs, mask)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        outs = self.backbone(img)
        head_outs = self.head(outs)
        head_outs = torch.sigmoid(head_outs)
        return head_outs

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
