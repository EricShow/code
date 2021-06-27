import torch
import torch.nn as nn
from starship.umtf.common import build_pipelines
from starship.umtf.common.model import NETWORKS, build_backbone, build_head


@NETWORKS.register_module
class HeatmapNetwork(nn.Module):
    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(HeatmapNetwork, self).__init__()

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

        self._init_module(self.backbone)
        self._init_module(self.head)

    @torch.jit.ignore
    def forward(self, img):
        with torch.no_grad():
            # 数据pipeline(augmentation)处理
            result = self._pipeline(dict(img=img))
            # result = {"img": img}
            img = result["img"][:, 0:3, :]
            cor_mask, vein_mask, sk_seg = result["img"][:, 3:4, :], result["img"][:, 4:5, :], result["img"][:, 5:6, :]
            target = [cor_mask, vein_mask, sk_seg]
            # thresh = 0.13
            # img[:, 1:3, :] = torch.max_pool3d((img[:, 1:3, :] > thresh).float(), kernel_size=3, stride=1, padding=1)

        # import SimpleITK as sitk
        # import time

        # name = time.time()
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(img[0, 0].detach().cpu().float().numpy()),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_img0.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(img[0, 1].detach().cpu().float().numpy()),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_heart.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(img[0, 2].detach().cpu().float().numpy()),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_artery.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray((cor_mask[0, 0] > 0.2).detach().cpu().numpy().astype("uint8")),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_cor.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray((vein_mask[0, 0] > 0.2).detach().cpu().numpy().astype("uint8")),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_vein.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray((sk_seg[0, 0] > 0.2).detach().cpu().numpy().astype("uint8")),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_sk.nii.gz",
        # )

        outs = self.backbone(img)
        outs = self.head(outs)
        loss = self.head.loss(outs, target)

        return {
            "loss": loss["loss_cor"] + loss["loss_vein"],
            "cor": loss["loss_cor"].detach(),
            "vein": loss["loss_vein"].detach(),
        }

    @torch.jit.export
    def forward_test(self, img):
        # thresh = 0.13
        # img[:, 1:3, :] = torch.max_pool3d((img[:, 1:3, :] > thresh).float(), kernel_size=3, stride=1, padding=1)

        # import SimpleITK as sitk
        # import time

        # name = time.time()
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(img[0, 0].detach().cpu().float().numpy()),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_img.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray((img[0, 1] > 0.5).detach().cpu().numpy().astype("uint8")),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_seg.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray((img[0, 2] > 0.5).detach().cpu().numpy().astype("uint8")),
        #     f"/media/d/tx_data/tmp/heart/tmp/{name}_seg_check.nii.gz",
        # )

        outs = self.backbone(img)
        outs = self.head(outs)
        outs = torch.sigmoid(outs)
        return outs

    def _apply_sync_batchnorm(self):
        print("apply sync batch norm")
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

    def _init_module(self, module: nn.Module):
        # use xavier init parameter first
        for m in module.children():
            if len(list(m.children())) > 0:
                self._init_module(m)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

