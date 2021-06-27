import os
import random

import numpy as np
from torch.utils import data
from tqdm import tqdm
from utils.data.data_io import load_nii, save_nii
from utils.data.data_process import vol_add, vol_crop


class Artery_Dataset(data.Dataset):

    def __init__(self, train_root, plaque_root, win_level, win_width, patch_size, aug_parameters):
        self.win_level = win_level
        self.win_width = win_width
        self.small_pos_plaque_num_range = aug_parameters['small_pos_plaque_num']
        self.large_pos_plaque_num_range = aug_parameters['large_pos_plaque_num']
        self.large_neg_plaque_num_range = aug_parameters['large_neg_plaque_num']
        self.pos_gray_range = aug_parameters['pos_plaque_gray_range']
        self.neg_gray_range = aug_parameters['neg_plaque_gray_range']
        self.plaque_prob = aug_parameters['plaque_prob']

        self.vol_list, self.mask_list, self.smooth_list, self.seed_list = self.__load_data__(train_root)
        self.large_plaque_list, self.small_plaque_list = self.__load_plaque__(plaque_root)
        self.patch_size = patch_size
        self.seed_patch = self.__get_seed_patch__()

    def __len__(self):
        """get number of data."""
        return len(self.vol_list) * 100

    def __getitem__(self, index):
        data_num = len(self.vol_list)
        data_idx = random.randint(0, data_num - 1)
        seed_idx = random.randint(0, self.seed_list[data_idx].shape[0] - 1)
        seed = self.seed_list[data_idx][seed_idx]
        z_s = seed[0] - self.patch_size[0] // 2
        y_s = seed[1] - self.patch_size[1] // 2
        x_s = seed[2] - self.patch_size[2] // 2
        z_e = z_s + self.patch_size[0]
        y_e = y_s + self.patch_size[1]
        x_e = x_s + self.patch_size[2]

        vol_patch = vol_crop(self.vol_list[data_idx], (z_s, y_s, x_s), (z_e, y_e, x_e), pad_value=-2048)
        mask_patch = vol_crop(self.mask_list[data_idx], (z_s, y_s, x_s), (z_e, y_e, x_e), pad_value=0)
        smooth_patch = vol_crop(self.smooth_list[data_idx], (z_s, y_s, x_s), (z_e, y_e, x_e), pad_value=0)

        vol_patch = self.__normalization__(vol_patch)
        smooth_patch = smooth_patch.astype('float32') / 255.0
        plaques_patch = np.zeros(self.patch_size, dtype='float32')

        if random.random() < self.plaque_prob:
            vol_patch, plaques_patch = self.__add_plaque__(vol_patch, mask_patch, smooth_patch, plaques_patch)

        vol_patch = np.expand_dims(vol_patch, 0)
        mask_patch = np.expand_dims(mask_patch, 0)
        smooth_patch = np.expand_dims(smooth_patch, 0)
        seed_patch = np.expand_dims(self.seed_patch, 0)
        plaques_patch = np.expand_dims(plaques_patch, 0)

        return vol_patch, mask_patch, smooth_patch, seed_patch, plaques_patch

    def __normalization__(self, vol_np):
        win = [self.win_level - self.win_width / 2, self.win_level + self.win_width / 2]
        vol = vol_np.astype('float32')
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= self.win_width
        return vol

    def __load_data__(self, root):
        vol_list = []
        mask_list = []
        smooth_list = []
        seed_list = []
        for file_name in tqdm(os.listdir(root)):
            if ('seg' in file_name) or ('mask' in file_name):
                continue
            if not file_name.endswith('nii.gz'):
                continue

            # # debug
            # if len(vol_list) > 10:
            #     break

            mask_file_name = file_name.replace('.nii.gz', '') + '-seg.nii.gz'
            smooth_file_name = file_name.replace('.nii.gz', '') + '-mask.nii.gz'
            vol = load_nii(os.path.join(root, file_name))
            mask = load_nii(os.path.join(root, mask_file_name))
            mask_bin = (mask > 1).astype('uint8')
            smooth = load_nii(os.path.join(root, smooth_file_name))
            seed = np.argwhere(smooth > 120)
            vol_list.append(vol)
            mask_list.append(mask_bin)
            smooth_list.append(smooth)
            seed_list.append(seed)
        return vol_list, mask_list, smooth_list, seed_list

    def __load_plaque__(self, root):
        large_root = os.path.join(root, 'large')
        small_root = os.path.join(root, 'small')
        large_list = []
        small_list = []
        for file_name in os.listdir(large_root):
            plaque = load_nii(os.path.join(large_root, file_name))
            plaque = plaque.astype('float32') / 255.0
            large_list.append(plaque)
        for file_name in os.listdir(small_root):
            plaque = load_nii(os.path.join(small_root, file_name))
            plaque = plaque.astype('float32') / 255.0
            small_list.append(plaque)

        return large_list, small_list

    def __get_seed_patch__(self):
        seed_patch = np.zeros(self.patch_size, dtype='uint8')
        seed_patch[self.patch_size[0] // 2 - 1:self.patch_size[0] // 2 + 2,
                   self.patch_size[1] // 2 - 1:self.patch_size[1] // 2 + 2,
                   self.patch_size[2] // 2 - 1:self.patch_size[2] // 2 + 2] = 1
        return seed_patch

    def __add_plaque__(self, vol, mask, smooth, plaques):
        seed_list = np.argwhere((smooth < 0.6) & (smooth > 0.2))  # edge of vessel

        patch_center = [16, 16, 16]

        large_pos_plaque_num = random.randint(self.large_pos_plaque_num_range[0], self.large_pos_plaque_num_range[1])
        large_neg_plaque_num = random.randint(self.large_neg_plaque_num_range[0], self.large_neg_plaque_num_range[1])
        small_pos_plaque_num = random.randint(self.small_pos_plaque_num_range[0], self.small_pos_plaque_num_range[1])

        all_small_plaque_num = len(self.small_plaque_list)
        all_large_plaque_num = len(self.large_plaque_list)

        for _ in range(small_pos_plaque_num):
            plaque_idx = random.randint(0, all_small_plaque_num - 1)
            plaque_temp = self.small_plaque_list[plaque_idx]
            pos_gray = random.uniform(self.pos_gray_range[0], self.pos_gray_range[1])
            plaque_temp *= pos_gray
            seed_idx = random.randint(0, seed_list.shape[0] - 1)
            seed = seed_list[seed_idx]
            seed_start = [seed[i] - patch_center[i] for i in range(3)]
            vol_add(plaques, plaque_temp, seed_start)

        for _ in range(large_pos_plaque_num):
            plaque_idx = random.randint(0, all_large_plaque_num - 1)
            plaque_temp = self.large_plaque_list[plaque_idx]
            pos_gray = random.uniform(self.pos_gray_range[0], self.pos_gray_range[1])
            plaque_temp *= pos_gray
            seed_idx = random.randint(0, seed_list.shape[0] - 1)
            seed = seed_list[seed_idx]
            seed_start = [seed[i] - patch_center[i] for i in range(3)]
            vol_add(plaques, plaque_temp, seed_start)

        for _ in range(large_neg_plaque_num):
            plaque_idx = random.randint(0, all_large_plaque_num - 1)
            plaque_temp = self.large_plaque_list[plaque_idx]
            neg_gray = random.uniform(self.neg_gray_range[0], self.neg_gray_range[1])
            plaque_temp *= neg_gray
            seed_idx = random.randint(0, seed_list.shape[0] - 1)
            seed = seed_list[seed_idx]
            seed_start = [seed[i] - patch_center[i] for i in range(3)]
            vol_add(plaques, plaque_temp, seed_start)

        smooth_temp = smooth.copy()
        smooth_temp[smooth_temp > 0.3] = 1.0
        plaques *= smooth_temp
        vol += plaques
        vol = np.clip(vol, 0.0, 1.0)

        return vol, plaques
