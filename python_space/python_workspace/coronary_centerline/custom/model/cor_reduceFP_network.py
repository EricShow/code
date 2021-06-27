import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common import build_pipelines
from starship.umtf.common.model import NETWORKS, build_backbone, build_head


@NETWORKS.register_module
class CoronaryReducefpNetwork(nn.Module):

    def __init__(self, backbone, head, apply_sync_batchnorm=False, pipeline=[], train_cfg=None, test_cfg=None):
        super(CoronaryReducefpNetwork, self).__init__()

        # TODO: 定制网络
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

        self._pipeline = build_pipelines(pipeline)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        #self.t_h = torch.zeros(2, 128 , 1, 1, 1)
        #self.t_c = torch.zeros(2, 128 , 1, 1, 1)
        

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()
        self._sphere_coord_array = self._get_sphere_direction()
        self._show_count = 0

    def _get_sphere_direction(self):
        radius = 1.5 / 0.3
        sample_point_count = 1000
        sphere_coord_array = np.zeros(shape=(sample_point_count, 3), dtype=np.float32)

        offset = 2.0 / sample_point_count
        increment = math.pi * (3.0 - math.sqrt(5.0))

        for idx in range(sample_point_count):
            z = ((idx * offset) - 1.0) + (offset / 2.0)
            r = math.sqrt(1.0 - pow(z, 2.0))

            phi = ((idx + 1) % sample_point_count) * increment
            x = math.cos(phi) * r
            y = math.sin(phi) * r

            sphere_coord_array[idx] = np.array([radius * z, radius * y, radius * x])
        return sphere_coord_array

    @torch.jit.ignore
    def forward(self,sequence,label):

        # save_dir = './trail_data'
        # import os
        # import SimpleITK as sitk
        # show_img = img0
        # for idx in range(show_img.size(0)):
        #     save_prefix = os.path.join(save_dir, '%04d' % self._show_count)
        #     self._show_count += 1
        #
        #     vol_p0 = (show_img[idx, 0].cpu().numpy() * 255).astype(np.uint8)
        #     shape = np.array(vol_p0.shape)
        #     half_shape = shape // 2
        #
        #     dl = direction_label[idx].cpu().numpy()
        #     dl_max = np.max(np.unique(dl))
        #     dl_indexes = np.argwhere(dl == dl_max)
        #     dl_array = np.zeros(shape=tuple(shape), dtype=np.uint8)
        #
        #     is_joint = 1 if joint_label[idx, 0] == 1 else 2
        #     p_s = half_shape - 1
        #     p_e = half_shape + 2
        #     dl_array[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = 3
        #     for idx_direction in dl_indexes:
        #         idx_direction = idx_direction[0]
        #         direct = self._sphere_coord_array[idx_direction]
        #         move = direct + half_shape
        #         move = np.round(move).astype(np.int)
        #         p_s = move - 1
        #         p_e = move + 2
        #         dl_array[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = is_joint
        #
        #     img = sitk.GetImageFromArray(vol_p0)
        #     sitk.WriteImage(img, save_prefix + '-vol.nii.gz')
        #     img = sitk.GetImageFromArray(dl_array)
        #     sitk.WriteImage(img, save_prefix + '-direction.nii.gz')
        i = 0

        # print (sequence.size())
        # raise
        device = sequence.device 
        #print(device)
        #ls = [1,2,3]
        #len_ls = ls.__len__()
        batchsize = sequence.size(0)
        self.t_h = torch.zeros(batchsize, 128 , 1, 1, 1)
        self.t_c = torch.zeros(batchsize, 128 , 1, 1, 1)
        self.t_h = self.t_h.cuda(device)
        self.t_c = self.t_c.cuda(device)
        #print(self.t_h.device)
        while i < sequence.size(1):
            #print (sequence[:,i,0:1].size())
            #fp_predict, self.t_h, self.t_c = self.backbone(sequence[:,i,0:1],sequence[:,i,1:2],sequence[:,i,2:3], self.t_h, self.t_c)
            fp_predict, self.t_h, self.t_c = self.backbone(sequence[:,i,0:1],sequence[:,i,1:2],sequence[:,i,2:3], self.t_h, self.t_c)
            #d_predicts.append(d_predict)
            #c_predicts.append(c_predict)
            i = i+1
        #print("d_predict",d_predict)
        #print("c_predict",c_predict)
        #print(outs.shape)
        head_outs = self.head(fp_predict)
        loss = self.head.loss(head_outs, label)
        return loss

    @torch.jit.export
    def forward_test(self, img0, img1, img2):
        # TODO: 根据需求适配，python3.7 custom/utils/save_torchscript.py保存静态图时使用
        i = 0
        #len_list = input_list.__len__()
        # i循环次数根据我们采样的序列的patch数量来决定
        while i < 100:
            d_predict, c_predict, self.t_h, self.t_c = self.backbone(img0, img1, img2, self.t_h, self.t_c)
            i = i+1
        #head_outs = self.head(outs)
        #print("d_predict",d_predict)
        #print("c_predict",c_predict)

        dirction_head = d_predict
        # print(dirction_head)
        joint_head = c_predict
        dirction_head = F.softmax(dirction_head, dim=1)
        joint_head = torch.sigmoid(joint_head)
        return dirction_head, joint_head

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
