import math
import os
import random
import traceback
from typing import List, Union

import numpy as np
import scipy
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from starship.umtf.common import build_pipelines
from starship.umtf.common.dataset import DATASETS
from starship.umtf.service.component import CustomDataset, DefaultSampleDataset

import cc3d
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm
from collections import defaultdict

from ..utils.data.data_io import load_nii
from ..utils.data.data_process import vol_add, vol_crop


@DATASETS.register_module
class Cor_Heatmap_ysy_Sample_Dataset(DefaultSampleDataset):
    def __init__(
        self,
        dst_list_file,
        win_level,
        win_width,
        crop_size,
        random_chose_center_prob,
        random_drop_vein_prob,
        sample_frequent,
    ):
        self._win_level = win_level
        self._win_width = win_width
        self._sample_frequent = sample_frequent
        self._crop_size = crop_size
        self._data_file_list = self._load_file_list(dst_list_file)
        self._random_chose_center_prob = random_chose_center_prob
        self._random_drop_vein_prob = random_drop_vein_prob
        # self._check_data_av()

    def _check_data_av(self):
        new_lst = []
        print("check all file ----------------------------")
        for file_name in tqdm(self._data_file_list):
            try:
                self._load_source_data(file_name)
                new_lst.append(file_name)
            except Exception:
                print(f"load file erro {file_name}")

        self._data_file_list = new_lst

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not os.path.exists(line):
                    print(f"{line} not exist")
                    continue
                data_file_list.append(line)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        result = {}
        vol = torch.from_numpy(data["hu_vol"].astype(np.float32))
        vol = self._window_array(vol)

        shape = data["hu_vol"].shape
        sk_color = data["sk_line"]
        heart_seg = data["heart_seg"]
        artery_seg = data["artery_seg"]

        joint_points = np.argwhere(sk_color == 3)
        all_points = np.argwhere(sk_color > 0)

        if random.random() < self._random_drop_vein_prob:
            cc_vein, cc_vein_num = cc3d.connected_components(sk_color == 2, return_N=True)
            cc_vein_num_list = list(range(1, cc_vein_num + 1))
            random.shuffle(cc_vein_num_list)

            delete_ratio = random.randint(3, 5)
            for i in cc_vein_num_list[: -len(cc_vein_num_list) // delete_ratio]:
                cur_cc = cc_vein == i
                sk_color[cur_cc] = 0

        point_line_map, line_point_map = defaultdict(list), defaultdict(list)

        # cc_cor, cc_cor_num = cc3d.connected_components(sk_color == 1, return_N=True)
        # for i in range(1, cc_cor_num + 1):
        #     cur_cc = cc_cor == i
        #     if np.sum(cur_cc) < 4:
        #         sk_color[cur_cc] = 0
        # cc_cor, cc_cor_num = cc3d.connected_components(sk_color == 1, return_N=True)

        sk_seg = binary_dilation(sk_color > 0, np.ones([3, 3, 3]))
        cor_sk_seg = binary_dilation(sk_color == 1, np.ones([3, 3, 3]))
        vein_sk_seg = binary_dilation(sk_color == 2, np.ones([3, 3, 3]))

        # make sure cor continue
        vein_sk_seg[cor_sk_seg > 0] = 0

        # import SimpleITK as sitk
        # import time

        # name = time.time()
        # sitk.WriteImage(
        #     sitk.GetImageFromArray(data["hu_vol"]), f"/media/d/tx_data/tmp/heart/tmp/{name}_img.nii.gz",
        # )
        # sitk.WriteImage(
        #     sitk.GetImageFromArray((sk_seg).astype("uint8")), f"/media/d/tx_data/tmp/heart/tmp/{name}_sk.nii.gz",
        # )

        compose_data = [
            torch.from_numpy(sk_seg).float(),
            torch.from_numpy(heart_seg).float(),
            torch.from_numpy(artery_seg).float(),
            torch.from_numpy(cor_sk_seg).float(),
            torch.from_numpy(vein_sk_seg).float(),
            torch.from_numpy(sk_seg).float(),
        ]
        compose_data = [v[None] for v in compose_data]
        compose_data = torch.cat(compose_data, dim=0).detach()

        result["vol"] = vol
        result["sk_color"] = sk_color
        result["joint_points"] = joint_points
        result["all_points"] = all_points
        # result["cc_cor"] = cc_cor
        # result["cc_vein"] = cc_vein
        result["compose_data"] = compose_data

        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, sk_color, joint_points, all_points, compose_data = (
            _source_data["vol"],
            _source_data["sk_color"],
            _source_data["joint_points"],
            _source_data["all_points"],
            # _source_data["cc_cor"],
            # _source_data["cc_vein"],
            _source_data["compose_data"],
        )

        if random.random() > self._random_chose_center_prob:
            cen_p_idx = random.randint(0, len(joint_points) - 1)
            cen_p = joint_points[cen_p_idx]
        else:
            cen_p_idx = random.randint(0, len(all_points) - 1)
            cen_p = all_points[cen_p_idx]

        p_offset = np.random.randint(-10, 10, [3])
        p_offset[0] //= 2
        cen_p += p_offset

        compose_data = self._crop_data(compose_data, cen_p, list(compose_data.size()[1:]))
        # results = {"img": compose_data.detach(), "label": torch.tensor([label]).float().detach()}
        results = {"img": compose_data.detach()}
        return results

    def _crop_data(self, compose_data, cen_zyx, shape):
        xmin, ymin, zmin, xmax, ymax, zmax = (
            cen_zyx[2] - self._crop_size[2] // 2,
            cen_zyx[1] - self._crop_size[1] // 2,
            cen_zyx[0] - self._crop_size[0] // 2,
            cen_zyx[2] + self._crop_size[2] // 2,
            cen_zyx[1] + self._crop_size[1] // 2,
            cen_zyx[0] + self._crop_size[0] // 2,
        )

        want_box = [xmin, ymin, zmin, xmax, ymax, zmax]
        real_box = [
            max(xmin, 0),
            max(ymin, 0),
            max(zmin, 0),
            min(xmax, shape[2]),
            min(ymax, shape[1]),
            min(zmax, shape[0]),
        ]

        crop_data = torch.zeros([compose_data.shape[0], *self._crop_size])
        crop_data[
            :,
            real_box[2] - want_box[2] : real_box[5] - want_box[5] + self._crop_size[0],
            real_box[1] - want_box[1] : real_box[4] - want_box[4] + self._crop_size[1],
            real_box[0] - want_box[0] : real_box[3] - want_box[3] + self._crop_size[2],
        ] = compose_data[:, real_box[2] : real_box[5], real_box[1] : real_box[4], real_box[0] : real_box[3]]

        return crop_data

    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode="trilinear")

    def _window_array(self, vol):
        win = [self._win_level - self._win_width / 2, self._win_level + self._win_width / 2]
        vol = torch.clamp(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        return vol

    def __getitem__(self, idx):
        source_data = self._load_source_data(self._data_file_list[idx])
        return [None, source_data]

    @property
    def sampled_data_count(self):
        # TODO: sample后数据总数量
        return self._sample_frequent * self.source_data_count

    @property
    def source_data_count(self):
        # TODO: 原始数据总数量
        return len(self._data_file_list)

    def __len__(self):
        return self.source_data_count

    def sample_source_data(self, idx, source_data):
        sample = None
        if idx < self._sample_frequent:
            sample = self._sample_source_data(idx, source_data)
        return sample

    def _normalization(self, vol_np):
        win = [self._win_level - self._win_width / 2, self._win_level + self._win_width / 2]
        vol = vol_np.astype("float32")
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        return vol
