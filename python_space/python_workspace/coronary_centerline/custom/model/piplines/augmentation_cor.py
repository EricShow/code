import math
import os
import random

import torch
import torch.nn as nn
import torchgeometry as tgm
from starship.umtf.common.dataset import PIPELINES


@PIPELINES.register_module
class AugmentationCor(nn.Module):

    def __init__(self, aug_parameters, size_o, intermedia_size):
        super(AugmentationCor, self).__init__()

        self.rot_range_x = aug_parameters['rot_range_x']
        self.rot_range_y = aug_parameters['rot_range_y']
        self.rot_range_z = aug_parameters['rot_range_z']
        self.scale_range_x = aug_parameters['scale_range_x']
        self.scale_range_y = aug_parameters['scale_range_y']
        self.scale_range_z = aug_parameters['scale_range_z']
        self.scale_range_xyz = aug_parameters['scale_range_xyz']
        self.shift_range_x = aug_parameters['shift_range_x']
        self.shift_range_y = aug_parameters['shift_range_y']
        self.shift_range_z = aug_parameters['shift_range_z']
        self.contrast = aug_parameters['contrast']
        self.gray_shift = aug_parameters['gray_shift']
        self.flip_x = aug_parameters['flip_x']
        self.flip_y = aug_parameters['flip_y']
        self.flip_z = aug_parameters['flip_z']
        self.elastic_alpha = aug_parameters['elastic_alpha']
        self.size_o = size_o
        self._intermedia_size = intermedia_size

    def _mixup_arteries(self, vol_orig, smooth_orig, vol_aug, smooth_aug, try_num=5):

        def is_overlap(mask_0, mask_1, threshold=0.7):
            overlap = (mask_0 > threshold) & (mask_1 > threshold)
            if torch.max(overlap) > 0:
                return True
            else:
                return False

        def mixup_single_artery(vol_orig, vol_aug, mask_orig, rand_gray_range=(-0.2, 0.1)):
            rand_gray_shift = random.uniform(rand_gray_range[0], rand_gray_range[1])
            vol_mixup = vol_aug * (1 - mask_orig) + (vol_orig + rand_gray_shift) * mask_orig
            vol_mixup = vol_mixup.clamp(0.0, 1.0)
            return vol_mixup

        def rand_crop(vol, mask):
            size_src = self._intermedia_size
            size_tgt = self.size_o
            rand_shift = [random.randint(0, size_src[i] - size_tgt[i] - 1) for i in range(len(size_src))]
            vol_crop = vol[:, rand_shift[0]:rand_shift[0] + size_tgt[0], rand_shift[1]:rand_shift[1] + size_tgt[1],
                           rand_shift[2]:rand_shift[2] + size_tgt[2]]
            mask_crop = mask[:, rand_shift[0]:rand_shift[0] + size_tgt[0], rand_shift[1]:rand_shift[1] + size_tgt[1],
                             rand_shift[2]:rand_shift[2] + size_tgt[2]]
            return vol_crop, mask_crop

        bs = vol_orig.size(0)
        half_bs = bs // 2
        vol_mixup_batch = vol_aug.clone()
        for i in range(half_bs):
            vol_mixup = None
            for _ in range(try_num):
                vol_crop, smooth_crop = rand_crop(vol_orig[i + half_bs], smooth_orig[i + half_bs])
                if not is_overlap(smooth_crop, smooth_aug[i]):
                    vol_mixup = mixup_single_artery(vol_crop, vol_aug[i], smooth_crop)
                    break
            if vol_mixup is not None:
                vol_mixup_batch[i] = vol_mixup
        return vol_mixup_batch

    def forward(self, data):
        vol, mask, smooth, seed = data['vol'], data['mask'], data['smooth'], data['seed']

        vol_aug, mask_aug, smooth_aug, seed_aug = self._data_aug(vol, mask, smooth, seed)
        vol_aug, mask_aug, smooth_aug, seed_aug = self._crop(vol_aug, mask_aug, smooth_aug, seed_aug)
        seed_aug[seed_aug > 0.5] = 1.0

        vol_mixup = self._mixup_arteries(vol, smooth, vol_aug, smooth_aug)
        vol_aug = torch.cat((vol_mixup, seed_aug), dim=1)

        data['vol'] = vol_aug
        data['mask'] = mask_aug
        return data

    def _crop(self, vol, mask, smooth, seed):
        center = [vol.size(2) // 2, vol.size(3) // 2, vol.size(4) // 2]
        z_s = center[0] - self.size_o[0] // 2
        z_e = z_s + self.size_o[0]
        y_s = center[1] - self.size_o[1] // 2
        y_e = y_s + self.size_o[1]
        x_s = center[2] - self.size_o[2] // 2
        x_e = x_s + self.size_o[2]
        vol_crop = vol[:, :, z_s:z_e, y_s:y_e, x_s:x_e]
        mask_crop = mask[:, :, z_s:z_e, y_s:y_e, x_s:x_e]
        smooth_crop = smooth[:, :, z_s:z_e, y_s:y_e, x_s:x_e]
        seed_crop = seed[:, :, z_s:z_e, y_s:y_e, x_s:x_e]

        return vol_crop, mask_crop, smooth_crop, seed_crop

    def _affine_transform_3d_gpu(self, data, rot, scale, shift, mode, padding_mode='zeros'):
        '''
        :param data: N * C * D * H * W, float32 cuda tensor, value belongs [0, 1]
        :param rot: N * 3, range is [-pi/2, pi/2]
        :param scale: N * 3, range is [0, 2]
        :param shift: N * 3, range is [-1.0, 1.0]
        :return:
        '''
        aff_matrix = tgm.angle_axis_to_rotation_matrix(rot)  # Nx4x4
        aff_matrix[:, 0, 3] = shift[:, 0]  # * data.size(4)
        aff_matrix[:, 1, 3] = shift[:, 1]  # * data.size(3)
        aff_matrix[:, 2, 3] = shift[:, 2]  # * data.size(2)
        # if scale:
        aff_matrix[:, 0, 0] *= scale[:, 0]
        aff_matrix[:, 1, 0] *= scale[:, 0]
        aff_matrix[:, 2, 0] *= scale[:, 0]
        aff_matrix[:, 0, 1] *= scale[:, 1]
        aff_matrix[:, 1, 1] *= scale[:, 1]
        aff_matrix[:, 2, 1] *= scale[:, 1]
        aff_matrix[:, 0, 2] *= scale[:, 2]
        aff_matrix[:, 1, 2] *= scale[:, 2]
        aff_matrix[:, 2, 2] *= scale[:, 2]

        aff_matrix = aff_matrix[:, 0:3, :]

        grid = torch.nn.functional.affine_grid(aff_matrix, data.size()).to(data.device)
        trans_data = torch.nn.functional.grid_sample(data, grid, mode, padding_mode=padding_mode)
        return trans_data

    def _affine_elastic_transform_3d_gpu(
        self, vol_list, rot, scale, shift, mode_list, alpha=2.0, smooth_num=4, win=[5, 5, 5], field_size=[30, 30, 15]
    ):
        aff_matrix = tgm.angle_axis_to_rotation_matrix(rot)  # Nx4x4
        aff_matrix[:, 0, 3] = shift[:, 0]  # * data.size(4)
        aff_matrix[:, 1, 3] = shift[:, 1]  # * data.size(3)
        aff_matrix[:, 2, 3] = shift[:, 2]  # * data.size(2)
        # if scale:
        aff_matrix[:, 0, 0] *= scale[:, 0]
        aff_matrix[:, 1, 0] *= scale[:, 0]
        aff_matrix[:, 2, 0] *= scale[:, 0]
        aff_matrix[:, 0, 1] *= scale[:, 1]
        aff_matrix[:, 1, 1] *= scale[:, 1]
        aff_matrix[:, 2, 1] *= scale[:, 1]
        aff_matrix[:, 0, 2] *= scale[:, 2]
        aff_matrix[:, 1, 2] *= scale[:, 2]
        aff_matrix[:, 2, 2] *= scale[:, 2]

        aff_matrix = aff_matrix[:, 0:3, :]

        grid = torch.nn.functional.affine_grid(aff_matrix, vol_list[0].size()).cuda()

        pad = [win[i] // 2 for i in range(3)]
        fs = field_size
        dz = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2).cuda()
        dy = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2).cuda()
        dx = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2).cuda()
        dz = (dz - 0.5) * 2.0 * alpha
        dy = (dy - 0.5) * 2.0 * alpha
        dx = (dx - 0.5) * 2.0 * alpha

        for _ in range(smooth_num):
            dz = self._smooth_3d(dz, win)
            dy = self._smooth_3d(dy, win)
            dx = self._smooth_3d(dx, win)

        dz = dz[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]
        dy = dy[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]
        dx = dx[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]

        size_3d = [vol_list[0].size(2), vol_list[0].size(3), vol_list[0].size(4)]
        batch_size = vol_list[0].size(0)
        dz = self._resize(dz, size_3d).repeat(batch_size, 1, 1, 1, 1)
        dy = self._resize(dy, size_3d).repeat(batch_size, 1, 1, 1, 1)
        dx = self._resize(dx, size_3d).repeat(batch_size, 1, 1, 1, 1)

        grid[:, :, :, :, 0] += dz[:, 0, :, :, :]
        grid[:, :, :, :, 1] += dy[:, 0, :, :, :]
        grid[:, :, :, :, 2] += dx[:, 0, :, :, :]

        vol_o_list = []
        for i, vol in enumerate(vol_list):
            vol_o = torch.nn.functional.grid_sample(vol, grid, mode_list[i])
            vol_o_list.append(vol_o)
        return vol_o_list

    def _smooth_3d(self, vol, win):
        kernel = torch.ones([1, vol.size(1), win[0], win[1], win[2]]).cuda()
        pad_size = [
            (int)((win[2] - 1) / 2), (int)((win[2] - 1) / 2), (int)((win[1] - 1) / 2), (int)((win[1] - 1) / 2),
            (int)((win[0] - 1) / 2), (int)((win[0] - 1) / 2)
        ]
        vol = torch.nn.functional.pad(vol, pad_size, 'replicate')
        vol_s = torch.nn.functional.conv3d(vol, kernel, stride=(1, 1, 1)) / torch.sum(kernel)
        return vol_s

    def _resize(self, vol, size_tgt, mode='trilinear'):
        vol_t = nn.functional.interpolate(vol, size=size_tgt, mode=mode, align_corners=False)

        return vol_t

    def _data_aug(self, vol, mask, smooth, seed):
        N = vol.size(0)
        rand_rot_x = (
            torch.rand(N, 1) * (self.rot_range_x[1] - self.rot_range_x[0]) + self.rot_range_x[0]
        ) / 180 * math.pi
        rand_rot_y = (
            torch.rand(N, 1) * (self.rot_range_y[1] - self.rot_range_y[0]) + self.rot_range_y[0]
        ) / 180 * math.pi
        rand_rot_z = (
            torch.rand(N, 1) * (self.rot_range_z[1] - self.rot_range_z[0]) + self.rot_range_z[0]
        ) / 180 * math.pi
        rand_rot = torch.cat([rand_rot_x, rand_rot_y, rand_rot_z], dim=1).cuda()

        rand_scale_x = torch.rand(N, 1) * (self.scale_range_x[1] - self.scale_range_x[0]) + self.scale_range_x[0]
        rand_scale_y = torch.rand(N, 1) * (self.scale_range_y[1] - self.scale_range_y[0]) + self.scale_range_y[0]
        rand_scale_z = torch.rand(N, 1) * (self.scale_range_z[1] - self.scale_range_z[0]) + self.scale_range_z[0]
        rand_scale_xyz = random.uniform(self.scale_range_xyz[0], self.scale_range_xyz[1])
        rand_scale = torch.cat([rand_scale_x, rand_scale_y, rand_scale_z], dim=1).cuda() * rand_scale_xyz

        rand_shift_x = torch.rand(N, 1) * (self.shift_range_x[1] - self.shift_range_x[0]) + self.shift_range_x[0]
        rand_shift_y = torch.rand(N, 1) * (self.shift_range_y[1] - self.shift_range_y[0]) + self.shift_range_y[0]
        rand_shift_z = torch.rand(N, 1) * (self.shift_range_z[1] - self.shift_range_z[0]) + self.shift_range_z[0]
        rand_shift = torch.cat([rand_shift_x, rand_shift_y, rand_shift_z], dim=1).cuda()

        # rotation and shift
        # vol_aug = self.__affine_transform_3d_gpu__(vol, rand_rot, rand_scale, rand_shift, "bilinear")
        # mask_aug = self.__affine_transform_3d_gpu__(mask.float(), rand_rot, rand_scale, rand_shift, "nearest")
        # mask_aug = mask_aug.long()
        # weight_aug = self.__affine_transform_3d_gpu__(weight, rand_rot, rand_scale, rand_shift, "bilinear")

        vol_aug, mask_aug, smooth_aug, seed_aug = self._affine_elastic_transform_3d_gpu(
            [vol, mask.float(), smooth, seed.float()],
            rand_rot,
            rand_scale,
            rand_shift, ['bilinear', 'bilinear', 'bilinear', 'bilinear'],
            alpha=self.elastic_alpha
        )
        # mask_aug = mask_aug.long()
        seed_aug[seed_aug >= 0.4] = 1.0
        seed_aug[seed_aug <= 0.4] = 0.0

        # contrast
        rand_contrast = random.uniform(self.contrast[0], self.contrast[1])
        rand_gray_shift = random.uniform(self.gray_shift[0], self.gray_shift[1])
        vol_aug = vol_aug * rand_contrast + rand_gray_shift
        vol_aug = torch.clamp(vol_aug, 0.0, 1.0)

        # flip
        if self.flip_x and random.randint(0, 1) == 0:
            vol_aug = torch.flip(vol_aug, dims=[4])
            mask_aug = torch.flip(mask_aug, dims=[4])
            smooth_aug = torch.flip(smooth_aug, dims=[4])
            seed_aug = torch.flip(seed_aug, dims=[4])

        if self.flip_y and random.randint(0, 1) == 0:
            vol_aug = torch.flip(vol_aug, dims=[3])
            mask_aug = torch.flip(mask_aug, dims=[3])
            smooth_aug = torch.flip(smooth_aug, dims=[3])
            seed_aug = torch.flip(seed_aug, dims=[3])

        if self.flip_z and random.randint(0, 1) == 0:
            vol_aug = torch.flip(vol_aug, dims=[2])
            mask_aug = torch.flip(mask_aug, dims=[2])
            smooth_aug = torch.flip(smooth_aug, dims=[2])
            seed_aug = torch.flip(seed_aug, dims=[2])

        return vol_aug.detach(), mask_aug.detach(), smooth_aug.detach(), seed_aug.detach()
