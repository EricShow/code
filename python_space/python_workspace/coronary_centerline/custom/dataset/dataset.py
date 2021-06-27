"""data loader."""

import os
import random
import math
from typing import List, Union
from tqdm import tqdm
import traceback

import numpy as np
import scipy
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
import torch
from starship.umtf.common import build_pipelines
from starship.umtf.common.dataset import DATASETS
from starship.umtf.service.component import CustomDataset, DefaultSampleDataset

from ..utils.data.data_io import load_nii
from ..utils.data.data_process import vol_add, vol_crop

@DATASETS.register_module
class CoronarySeg_Squence_Sample_Dataset(DefaultSampleDataset):
    def __init__(self, dst_list_file, win_level, win_width, patch_size, isotropy_spacing, data_pyramid_level,
                 data_pyramid_step, sample_frequent):
        self._win_level = win_level
        self._win_width = win_width
        self._sample_frequent = sample_frequent
        self._isotropy_spacing = np.array([isotropy_spacing] * 3)
        assert patch_size % 2 == 1, f'patch_size:{patch_size} should be odd!'
        self._patch_size = np.array((patch_size,) * 3)
        self._data_pyramid_level = data_pyramid_level
        self._data_pyramid_step = data_pyramid_step
        self._data_file_list = self._load_file_list(dst_list_file)
        self.draw_idx = 1
        # self._check_data_av()

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not os.path.exists(line):
                    print(f'{line} not exist')
                    continue
                data_file_list.append(line)
        assert len(data_file_list) != 0, 'has no avilable file in dst_list_file'
        return data_file_list

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        result = {}
        for k in data.keys():
            result[k] = data[k]
        result['ep_info'] = dict(result['ep_info'].item())
        result['joint_info'] = dict(result['joint_info'].item())
        result['sk_info'] = dict(result['sk_info'].item())
        result['smooth_seg'] = torch.from_numpy(result['smooth_seg'].astype(np.float32))[None, None] / 255.0
        result['vol'] = torch.from_numpy(result['vol'].astype(np.float32))[None, None]
        # result['vol'] = torch.from_numpy(result['seg'].astype(np.float32))[None, None]  # for debug
        return result
    
    
    def _sample_single_line(self,select_id, adjs_info, degree_info,points):
        line_list = [select_id]
        while True:
            if select_id not in adjs_info:
                break
            select_id = adjs_info[select_id][0]
            degree = degree_info[select_id]
            line_list.append(select_id)
            if degree > 2:
                break
        points_list = []
        for idx in line_list:
            points_list.append(points[idx])
        return points_list
                 

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg, smooth_seg, spacing, ep_info, joint_info, sk_info = _source_data['vol'], _source_data[
            'seg'], _source_data['smooth_seg'], _source_data['spacing'], _source_data['ep_info'], _source_data[
                                                                          'joint_info'], _source_data[
                                                                          'sk_info']
        shape_array = np.array(vol.shape[2:])
        phyical_shape = shape_array * spacing
        points = sk_info['points']
        radius_info = sk_info['radius_info']
        adjs_info = sk_info['adjs_info']   
        degree_info = sk_info["degree_info"]
        select_range = np.argwhere(degree_info==3)
        select_range = select_range.reshape(-1)
        
        adjs_dict = {}
        for adj in adjs_info:
            if len(adj)>1:
                if adj[0] not in adjs_dict:
                    adjs_dict[adj[0]]=[adj[1]]
                else:
                    adjs_dict[adj[0]].append(adj[1])
        while True:
            select_id = int(np.random.choice(select_range))
            points_list = self._sample_single_line(select_id,adjs_dict,degree_info,points)
            if len(points_list) > 20:
                break
        
        sequence_list = []
        
        max_length = 20
        for idx,c_p in enumerate(points_list):
            if idx%4==0:
                c_t_pixel = c_p / spacing
                c_t_pixel = np.round(c_t_pixel).astype(np.int)
                data_pyramid = self._get_pyramid_data(vol, c_p, spacing, shape_array)
                data_pyramid = torch.cat(data_pyramid,0)

                sequence_list.append(data_pyramid)
                if len(sequence_list)==20:
                    break
        if len(sequence_list)<20:
            for _ in range(20-len(sequence_list)):
                sequence_list.append(torch.zeros_like(sequence_list[0]))
    
        # raise
        sequence_tensor = torch.stack(tuple(sequence_list),0)

        results = {}
        results['sequence']=sequence_tensor
        results['label']=0

        return results

    def _get_sample_grid(self, center_point, src_spacing, src_shape, tgt_spacing, rot_mat=None):
        ps_half = self._patch_size[0] // 2
        grid = []
        for cent_px, ts in zip(center_point, tgt_spacing):
            p_s = cent_px - ps_half * ts
            p_e = cent_px + (ps_half + 1) * ts - (ts / 2)
            grid.append(np.arange(p_s, p_e, ts))
        grid = np.meshgrid(*grid)
        grid = [g[:, :, :, None] for g in grid]  # shape (h,d,w,(zxy))
        grid = np.concatenate(grid, axis=-1)
        grid = np.transpose(grid, axes=(1, 0, 2, 3))  # shape (d,h,w,(zxy))

        if rot_mat is not None:
            grid -= center_point[None, None, None, :]
            grid = np.matmul(grid, np.linalg.inv(rot_mat))
            grid += center_point[None, None, None, :]
        grid *= 2
        grid /= src_spacing[None, None, None, :]
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1
        # change z,y,x to x,y,z
        grid = np.array(grid[:, :, :, ::-1], dtype=np.float32)
        return torch.from_numpy(grid)[None]

    def _get_pyramid_data(self, vol, c_t, src_spacing, vol_shape_array, rot_mat=None):
        data_pyramid = []
        for level in range(self._data_pyramid_level):
            tgt_spacing = self._isotropy_spacing * (level * self._data_pyramid_step + 1)
            grid = self._get_sample_grid(c_t, src_spacing, vol_shape_array, tgt_spacing, rot_mat)
            data = torch.nn.functional.grid_sample(vol, grid, align_corners=True)[0]
            data = self._window_array(data)
            data_pyramid.append(data)
        return data_pyramid

    def _window_array(self, vol):
        win = [self._win_level - self._win_width / 2, self._win_level + self._win_width / 2]
        vol = torch.clamp(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        # if np.random.uniform() < self._gaussian_noise_prob:
        #     vol = self._augment_gaussian_noise(vol)
        #     vol = vol.clamp(0.0, 1.0)
        return vol

    def _augment_gaussian_noise(self, data, noise_variance=(0, 0.1)):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        data = data + torch.normal(0.0, variance, size=data.size())
        return data

    def _get_soft_direction_label(self, tgt_dir):
        label = np.zeros((self._direction_sphere_sample_count, 1), dtype=np.float32)
        cos_len = np.matmul(self._sphere_coord_array, tgt_dir.transpose())
        cos_len_idx = np.argmax(cos_len, axis=0)
        cos_len_idx = tuple(np.unique(cos_len_idx))
        tgt_degree = len(cos_len_idx)
        label[cos_len_idx, :] = 1 / len(cos_len_idx)
        label = label[:, 0]
        return label, tgt_degree

    def _get_direction_points(self, center_point, adjs_info, points):

        distance = np.linalg.norm(points - center_point[None, :], axis=1)
        distace_av = distance <= self._direction_sphere_radius
        pos_sample = np.argwhere(distace_av)
        pos_sample = [v[0] for v in pos_sample]
        expect_degree = 0
        surface_coord_idx = []
        for idx_pos in pos_sample:
            adj_p = adjs_info[idx_pos]
            if not np.all(distace_av[adj_p]):
                expect_degree += 1
                surface_coord_idx.append(idx_pos)
        if expect_degree == 0 or expect_degree == 1:
            return None
        # tgt_dir = points[surface_coord_idx, :] - center_point[None, :]
        # tgt_dir = tgt_dir / np.linalg.norm(tgt_dir, axis=1, keepdims=True)
        return points[surface_coord_idx, :].copy()

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
        vol = vol_np.astype('float32')
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        return vol
@DATASETS.register_module
class CoronarySeg_direction_Sample_Dataset(DefaultSampleDataset):

    def __init__(self, dst_list_file, win_level, win_width, patch_size, isotropy_spacing, data_pyramid_level,
                 data_pyramid_step,
                 joint_sample_prob, translation_prob, pseudo_ratio, rotation_prob, rot_range, mixup_prob,
                 gaussian_noise_prob,
                 direction_sphere_radius,
                 direction_sphere_sample_count,
                 sample_frequent):
        self._win_level = win_level
        self._win_width = win_width
        self._sample_frequent = sample_frequent
        self._isotropy_spacing = np.array([isotropy_spacing] * 3)
        assert patch_size % 2 == 1, f'patch_size:{patch_size} should be odd!'
        self._patch_size = np.array((patch_size,) * 3)
        self._data_pyramid_level = data_pyramid_level
        self._data_pyramid_step = data_pyramid_step
        self._joint_sample_prob = joint_sample_prob
        self._translation_prob = translation_prob
        self._pseudo_ratio = pseudo_ratio
        self._rotation_prob = rotation_prob
        rot_range = [math.radians(rr) for rr in rot_range]
        self._rot_range = rot_range
        self._mixup_prob = mixup_prob
        self._gaussian_noise_prob = gaussian_noise_prob
        self._direction_sphere_radius = direction_sphere_radius
        self._direction_sphere_sample_count = direction_sphere_sample_count
        self._sphere_coord_array = self._get_sphere_direction()
        self._data_file_list = self._load_file_list(dst_list_file)
        self.draw_idx = 1
        # self._check_data_av()

    def _check_data_av(self):
        new_lst = []
        print('check all file ----------------------------')
        for file_name in tqdm(self._data_file_list):
            #print(file_name)
            try:
                self._load_source_data(file_name)
                new_lst.append(file_name)
            except Exception:
                print(f'load file erro {file_name}')

        self._data_file_list = new_lst

    def _get_sphere_direction(self):
        radius = self._direction_sphere_radius
        sample_point_count = self._direction_sphere_sample_count
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

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, 'r') as f:
            for line in f:
                line = line.strip()   #移除末尾符号
                #print(line)
                if not os.path.exists(line):
                    print(f'{line} not exist')
                    continue
                data_file_list.append(line)
                
        assert len(data_file_list) != 0, 'has no avilable file in dst_list_file'
        return data_file_list

    def _load_source_data(self, file_name):
        try:
            data = np.load(file_name, allow_pickle=True)
        except:
            print (file_name)
            raise
        result = {}
        for k in data.keys():
            result[k] = data[k]
        result['ep_info'] = dict(result['ep_info'].item())
        result['joint_info'] = dict(result['joint_info'].item())
        result['sk_info'] = dict(result['sk_info'].item())
        result['smooth_seg'] = torch.from_numpy(result['smooth_seg'].astype(np.float32))[None, None] / 255.0
        result['vol'] = torch.from_numpy(result['vol'].astype(np.float32))[None, None]
        # result['vol'] = torch.from_numpy(result['seg'].astype(np.float32))[None, None]  # for debug
        return result

    # def _get_rotate_mat(self, axis, angle):
    #     rot_matrix = scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * angle))
    #     return rot_matrix

    def _mixup_arteries(self, points_full, point_indices_other, point_need, vol_need, rot_mat_need, smooth_full,
                        vol_full, src_spacing,
                        shape_array, radius_info, try_num=100):

        def is_overlap(mask_0, mask_1, threshold):
            overlap = (mask_0 > threshold) & (mask_1 > threshold)
            if torch.max(overlap) > 0:
                return True
            else:
                return False

        def mixup_single_artery(v_a, v_n, s_a, rand_gray_range=(-0.2, 0.0)):
            rand_gray_shift = random.uniform(rand_gray_range[0], rand_gray_range[1])
            vol_mixup = v_n * (1 - s_a) + (v_a + rand_gray_shift) * s_a
            vol_mixup = vol_mixup.clamp(0.0, 1.0)
            # vol_mixup = v_a * 0.3 + v_n * 0.7
            return vol_mixup

        def _interplot_crop_data(data, p, tgt_spacing, rot_mat):
            grid = self._get_sample_grid(p, src_spacing, shape_array, tgt_spacing, rot_mat)
            data = torch.nn.functional.grid_sample(data, grid, align_corners=True)[0]
            return data

        smooth_need = _interplot_crop_data(smooth_full, point_need, self._isotropy_spacing, rot_mat_need)
        rot_range = np.array([360] * 3)
        t_count = 0
        while t_count < try_num:
            p_add = random.choice(point_indices_other)
            radius = radius_info[p_add]
            if radius > 3:
                continue

            p_add = points_full[p_add]
            delta = np.random.uniform(1, 2, (3,)) * (np.random.randint(0, 2, (3,)) * 2 - 1)
            p_add = p_add + delta

            p_add_pixel = p_add / src_spacing
            p_add_pixel = np.round(p_add_pixel).astype(np.int)

            if np.any(p_add_pixel >= shape_array):
                continue
            if smooth_full[0, 0, p_add_pixel[0], p_add_pixel[1], p_add_pixel[2]] > 0.4:
                continue

            _, rot_mat_other = self._rotate_point(p_add[None,], p_add, shape_array, rot_range)
            smooth_add = _interplot_crop_data(smooth_full, p_add, self._isotropy_spacing, rot_mat_other)
            if not is_overlap(smooth_need, smooth_add, 0.5):
                vol_add = self._get_pyramid_data(vol_full, p_add, src_spacing, shape_array, rot_mat_other)
                vol_need = [mixup_single_artery(v_a, v_n, smooth_add, (-0.25, -0.05)) for v_a, v_n in
                            zip(vol_add, vol_need)]
                break
            t_count += 1
        return vol_need

    def _get_rotate_mat(self, z_angle, y_angle, x_angle):
        def _create_matrix_rotation_z_3d(angle, matrix=None):
            rotation_x = np.array([[1, 0, 0],
                                   [0, np.cos(angle), -np.sin(angle)],
                                   [0, np.sin(angle), np.cos(angle)]])
            if matrix is None:
                return rotation_x
            return np.dot(matrix, rotation_x)

        def _create_matrix_rotation_y_3d(angle, matrix=None):
            rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]])
            if matrix is None:
                return rotation_y

            return np.dot(matrix, rotation_y)

        def _create_matrix_rotation_x_3d(angle, matrix=None):
            rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0],
                                   [0, 0, 1]])
            if matrix is None:
                return rotation_z

            return np.dot(matrix, rotation_z)

        rot_matrix = np.identity(3)
        rot_matrix = _create_matrix_rotation_z_3d(z_angle, rot_matrix)
        rot_matrix = _create_matrix_rotation_y_3d(y_angle, rot_matrix)
        rot_matrix = _create_matrix_rotation_x_3d(x_angle, rot_matrix)
        return rot_matrix

    def _rotate_point(self, point_array, center_point, shape_constrian, rot_range):
        center_point = center_point[None,]
        trans_point = point_array - center_point
        try_num = 100
        idx_try = 0
        flag = False
        while idx_try < try_num:
            idx_try += 1
            z_angle = (np.random.random() * 2 - 1) * rot_range[0]
            y_angle = (np.random.random() * 2 - 1) * rot_range[1]
            x_angle = (np.random.random() * 2 - 1) * rot_range[2]

            rot_mat = self._get_rotate_mat(z_angle, y_angle, x_angle)

            ret_point = np.matmul(trans_point, rot_mat)
            ret_point = ret_point + center_point
            if np.any(ret_point < 0) or np.any(ret_point >= shape_constrian):
                continue
            else:
                flag = True
                break
        if flag:
            return ret_point, rot_mat
        else:
            return point_array, None

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg, smooth_seg, spacing, ep_info, joint_info, sk_info = _source_data['vol'], _source_data[
            'seg'], _source_data['smooth_seg'], _source_data['spacing'], _source_data['ep_info'], _source_data[
                                                                          'joint_info'], _source_data[
                                                                          'sk_info']
        shape_array = np.array(vol.shape[2:])
        phyical_shape = shape_array * spacing
        points = sk_info['points']
        radius_info = sk_info['radius_info']
        adjs_info = sk_info['adjs_info']

        joint_flag = random.random() < self._joint_sample_prob
        translation_flag = random.random() < self._translation_prob
        rotation_flag = random.random() < self._rotation_prob
        mixup_flag = random.random() < self._mixup_prob
        mixup_flag = mixup_flag and (not joint_flag)

        mixup_av_set = list(joint_info['no_joint'])

        if joint_flag:
            av_point_set = list(joint_info['joint'])
        else:
            av_point_set = list(joint_info['no_joint'])
        if len(av_point_set) == 0:
            av_point_set = list(joint_info['no_joint'])
            joint_flag = False
        delta_t_try_num = 100
        while True:
            try:
                idx_c_p = random.choice(av_point_set)
                c_p = points[idx_c_p]
                #print(c_p.shape)
                radius = radius_info[idx_c_p]
                if not translation_flag:   #translation_flag
                    c_pseudo = c_p
                    c_t = c_p
                else:
                    idx_try = 0
                    flag = False
                    while idx_try < delta_t_try_num:
                        idx_try += 1

                        delta_t = np.random.uniform(-1, 1, (3,)) * min(radius, 5)
                        c_t = c_p + delta_t
                        c_t_pixel = c_t / spacing
                        c_t_pixel = np.round(c_t_pixel).astype(np.int)
                        if np.any(c_t_pixel < 0) or np.any(c_t_pixel >= shape_array):
                            continue
                        if seg[c_t_pixel[0], c_t_pixel[1], c_t_pixel[2]] != 1:
                            continue
                        else:
                            tmp_distance = np.linalg.norm(points - c_t[None, :], axis=1)
                            tmp_idx_c_p = np.argmin(tmp_distance)
                            tmp_c_p = points[tmp_idx_c_p]
                            tmp_radius = radius_info[tmp_idx_c_p]

                            if tmp_distance[tmp_idx_c_p] > min(tmp_radius, 5):
                                continue
                            else:
                                idx_c_p = tmp_idx_c_p
                                c_p = tmp_c_p
                                delta_t = (c_t - c_p)
                                radius = tmp_radius
                                flag = True
                                break
                    if not flag:
                        c_t = c_p
                        delta_t = 0.0
                    c_pseudo = c_p + delta_t * self._pseudo_ratio

                tgt_points = self._get_direction_points(c_pseudo, adjs_info, points)
                if tgt_points is None or tgt_points.shape[0] == 1:
                    continue
                if rotation_flag:
                    need_rot_points = [c_pseudo[None, :], tgt_points]
                    need_rot_points = np.concatenate(need_rot_points, axis=0)
                    roted_points, rot_mat = self._rotate_point(need_rot_points, c_t, phyical_shape, self._rot_range)
                    c_pseudo = roted_points[0]
                    tgt_points = roted_points[1:, :]
                    data_pyramid = self._get_pyramid_data(vol, c_t, spacing, shape_array, rot_mat)
                else:
                    rot_mat = None
                    data_pyramid = self._get_pyramid_data(vol, c_t, spacing, shape_array)
                if not data_pyramid:
                    continue

                tgt_dir = tgt_points - c_pseudo[None, :]

                # save_dir = './trail_data'
                # import os
                # import SimpleITK as sitk
                # show_img = data_pyramid[0][0]
                # save_prefix = os.path.join(save_dir, '%04d' % self.draw_idx)
                # self.draw_idx += 1
                #
                # vol_p0 = (show_img.cpu().numpy() * 255).astype(np.uint8)
                # shape = np.array(vol_p0.shape)
                # half_shape = shape // 2
                #
                # dl_vectors = tgt_dir
                # # dl_vectors[:, 0] = -dl_vectors[:, 0]
                # dl_array = np.zeros(shape=tuple(shape), dtype=np.uint8)
                #
                # is_joint = 1 if joint_flag else 2
                # p_s = half_shape - 1
                # p_e = half_shape + 2
                # dl_array[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = 3
                # for direct in dl_vectors:
                #     move = direct / 0.3 + half_shape
                #     move = np.round(move).astype(np.int)
                #     p_s = move - 1
                #     p_e = move + 2
                #     dl_array[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = is_joint
                #
                # img = sitk.GetImageFromArray(vol_p0)
                # sitk.WriteImage(img, save_prefix + '-vol.nii.gz')
                # img = sitk.GetImageFromArray(dl_array)
                # sitk.WriteImage(img, save_prefix + '-direction.nii.gz')

                tgt_dir = tgt_dir / np.linalg.norm(tgt_dir, axis=1, keepdims=True)
                direction_label, tgt_degree = self._get_soft_direction_label(tgt_dir)
                if tgt_degree < 2:
                    continue
                if mixup_flag and radius < 3:
                    data_pyramid = self._mixup_arteries(points, mixup_av_set, c_t, data_pyramid, rot_mat,
                                                        smooth_seg, vol, spacing, shape_array, radius_info, try_num=500)
                is_joint = 1 if tgt_degree >= 3 else 0
                p_type = np.array([is_joint, ], dtype=np.float32)
                break
            except:
                traceback.print_exc()
                pass

        results = {'img%d' % idx: v for idx, v in enumerate(data_pyramid)}
        results['joint_label'] = p_type
        results['direction_label'] = direction_label
        return results

    def _get_sample_grid(self, center_point, src_spacing, src_shape, tgt_spacing, rot_mat=None):  #采样网格
        ps_half = self._patch_size[0] // 2
        grid = []
        for cent_px, ts in zip(center_point, tgt_spacing):
            p_s = cent_px - ps_half * ts
            p_e = cent_px + (ps_half + 1) * ts - (ts / 2)
            grid.append(np.arange(p_s, p_e, ts))
        grid = np.meshgrid(*grid)
        grid = [g[:, :, :, None] for g in grid]  # shape (h,d,w,(zxy))
        grid = np.concatenate(grid, axis=-1)
        grid = np.transpose(grid, axes=(1, 0, 2, 3))  # shape (d,h,w,(zxy))

        if rot_mat is not None:
            grid -= center_point[None, None, None, :]
            grid = np.matmul(grid, np.linalg.inv(rot_mat))
            grid += center_point[None, None, None, :]
        grid *= 2
        grid /= src_spacing[None, None, None, :]
        grid /= (src_shape - 1)[None, None, None, :]
        grid -= 1
        # change z,y,x to x,y,z
        grid = np.array(grid[:, :, :, ::-1], dtype=np.float32)
        return torch.from_numpy(grid)[None]

    def _get_pyramid_data(self, vol, c_t, src_spacing, vol_shape_array, rot_mat=None):
        data_pyramid = []
        for level in range(self._data_pyramid_level):
            tgt_spacing = self._isotropy_spacing * (level * self._data_pyramid_step + 1)
            grid = self._get_sample_grid(c_t, src_spacing, vol_shape_array, tgt_spacing, rot_mat)
            data = torch.nn.functional.grid_sample(vol, grid, align_corners=True)[0]
            data = self._window_array(data)
            data_pyramid.append(data)
        return data_pyramid

    def _window_array(self, vol):
        win = [self._win_level - self._win_width / 2, self._win_level + self._win_width / 2]
        vol = torch.clamp(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        if np.random.uniform() < self._gaussian_noise_prob:
            vol = self._augment_gaussian_noise(vol)
            vol = vol.clamp(0.0, 1.0)

        # vol = (vol > 0.5).float()

        # vol = vol.numpy()
        # upper_bound = np.percentile(vol, 99.5)
        # lower_bound = np.percentile(vol, 00.5)
        # data = np.clip(vol, lower_bound, upper_bound)
        # #if np.random.uniform() < self._gaussian_noise_prob:
        # #    data = self._augment_gaussian_noise(data)
        # mean_intensity = np.mean(vol)
        # std_intensity = np.std(vol)
        # data = (vol - mean_intensity) / (std_intensity + 1e-9)
        # vol = torch.from_numpy(vol)
        return vol

    def _augment_gaussian_noise(self, data, noise_variance=(0, 0.1)):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        data = data + torch.normal(0.0, variance, size=data.size())
        return data

    def _get_soft_direction_label(self, tgt_dir):
        label = np.zeros((self._direction_sphere_sample_count, 1), dtype=np.float32)
        cos_len = np.matmul(self._sphere_coord_array, tgt_dir.transpose())
        cos_len_idx = np.argmax(cos_len, axis=0)
        cos_len_idx = tuple(np.unique(cos_len_idx))
        tgt_degree = len(cos_len_idx)
        label[cos_len_idx, :] = 1 / len(cos_len_idx)
        label = label[:, 0]
        return label, tgt_degree

    def _get_direction_points(self, center_point, adjs_info, points):

        distance = np.linalg.norm(points - center_point[None, :], axis=1)
        distace_av = distance <= self._direction_sphere_radius
        pos_sample = np.argwhere(distace_av)
        pos_sample = [v[0] for v in pos_sample]
        expect_degree = 0
        surface_coord_idx = []
        for idx_pos in pos_sample:
            adj_p = adjs_info[idx_pos]
            if not np.all(distace_av[adj_p]):
                expect_degree += 1
                surface_coord_idx.append(idx_pos)
        if expect_degree == 0 or expect_degree == 1:
            return None
        # tgt_dir = points[surface_coord_idx, :] - center_point[None, :]
        # tgt_dir = tgt_dir / np.linalg.norm(tgt_dir, axis=1, keepdims=True)
        return points[surface_coord_idx, :].copy()

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
        vol = vol_np.astype('float32')
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        return vol


@DATASETS.register_module
class CorTrackingStop_Sample_Dataset(CoronarySeg_direction_Sample_Dataset):
    def __init__(self, dst_list_file, win_level, win_width, patch_size, isotropy_spacing, data_pyramid_level,
                 data_pyramid_step,
                 joint_sample_prob, translation_prob, pseudo_ratio, rotation_prob, rot_range, gaussian_noise_prob,
                 direction_sphere_radius,
                 direction_sphere_sample_count, end_choice_prob, fp_area_choice_prob, sample_frequent):
        super(CorTrackingStop_Sample_Dataset, self).__init__(dst_list_file, win_level, win_width, patch_size,
                                                             isotropy_spacing,
                                                             data_pyramid_level, data_pyramid_step,
                                                             joint_sample_prob, translation_prob, pseudo_ratio,
                                                             rotation_prob, rot_range, 0.0,
                                                             gaussian_noise_prob, direction_sphere_radius,
                                                             direction_sphere_sample_count,
                                                             sample_frequent)
        self._end_choice_prob = end_choice_prob
        self._fp_area_choice_prob = fp_area_choice_prob

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg, smooth_seg, spacing, ep_info, joint_info, sk_info = _source_data['vol'], _source_data[
            'seg'], _source_data['smooth_seg'], _source_data['spacing'], _source_data['ep_info'], _source_data[
                                                                          'joint_info'], _source_data[
                                                                          'sk_info']
        shape_array = np.array(vol.shape[2:])
        points = sk_info['points']
        radius_info = sk_info['radius_info']
        adjs_info = sk_info['adjs_info']

        translation_flag = random.random() < self._translation_prob
        rotation_flag = random.random() < self._rotation_prob

        end_choice_flag = random.random() < self._end_choice_prob
        fp_area_flag = random.random() < self._fp_area_choice_prob
        fp_area_flag = end_choice_flag and fp_area_flag
        if end_choice_flag:
            av_point_set = ep_info['ep']
            if fp_area_flag:
                av_point_set = av_point_set['outer']
            else:
                av_point_set = list(av_point_set['inner'])
        else:
            av_point_set = list(ep_info['no_ep'])

        if len(av_point_set) == 0:
            if end_choice_flag:
                fp_area_flag = True
                av_point_set = list(ep_info['ep']['outer'])
            if len(av_point_set) == 0:
                end_choice_flag = False
                fp_area_flag = False
                av_point_set = list(ep_info['no_ep'])

        shape_array = np.array(vol.shape[2:])
        phyical_shape = shape_array * spacing
        while True:
            try:
                if fp_area_flag:
                    c_p = random.choice(av_point_set)
                    c_t = c_p
                else:
                    idx_c_p = random.choice(av_point_set)
                    c_p = points[idx_c_p]
                    if not translation_flag:
                        c_t = c_p
                    else:
                        radius = radius_info[idx_c_p]
                        delta_t = np.random.uniform(-1, 1, (3,)) * min(radius, 1.5)

                        c_t = c_p + delta_t

                if rotation_flag:
                    need_rot_points = c_p[None, :]
                    _, rot_mat = self._rotate_point(need_rot_points, c_t, phyical_shape, self._rot_range)
                    data_pyramid = self._get_pyramid_data(vol, c_t, spacing, shape_array, rot_mat)
                else:
                    data_pyramid = self._get_pyramid_data(vol, c_t, spacing, shape_array)

                is_end = 1 if end_choice_flag else 0
                p_type = np.array([is_end, ], dtype=np.float32)
                break
            except:
                traceback.print_exc()
                pass

        results = {'img%d' % idx: v for idx, v in enumerate(data_pyramid)}
        results['label'] = p_type
        return results


@DATASETS.register_module
class CorSkSeg_Sample_Dataset(DefaultSampleDataset):
    def __init__(self, dst_list_file, win_level, win_width, patch_size, constant_shift, shift_range, sk_noise_prob,
                 joint_noise_ratio,
                 joint_noise_range, joint_noise_count, other_noise_ratio,
                 other_noise_range, sample_frequent):
        self._win_level = win_level
        self._win_width = win_width
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._constant_shift = constant_shift
        self._shift_range = shift_range
        self._sk_noise_prob = sk_noise_prob
        self._joint_noise_ratio = joint_noise_ratio
        self._joint_noise_range = joint_noise_range
        self._joint_noise_count = joint_noise_count
        self._other_noise_ratio = other_noise_ratio
        self._other_noise_range = other_noise_range
        self._data_file_list = self._load_file_list(dst_list_file)
        self.draw_idx = 1
        # self._check_data_av()

    def _check_data_av(self):
        new_lst = []
        print('check all file ----------------------------')
        for file_name in tqdm(self._data_file_list):
            try:
                self._load_source_data(file_name)
                new_lst.append(file_name)
            except Exception:
                print(f'load file erro {file_name}')

        self._data_file_list = new_lst

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not os.path.exists(line):
                    print(f'{line} not exist')
                    continue
                data_file_list.append(line)
        assert len(data_file_list) != 0, 'has no avilable file in dst_list_file'
        return data_file_list

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        result = {}
        result['spacing'] = data['spacing']
        result['vol'] = torch.from_numpy(data['vol'].astype(np.float32))
        result['vol'] = self._window_array(result['vol'])
        result['seg'] = torch.from_numpy(data['seg'].astype(np.float32))

        points = data['sk_info'].item()['points']
        spacing = result['spacing']
        joint_points = list(data['joint_info'].item()['joint'])
        result['sk_seg'], result['points_pixel'], result['sk_range'] = self._gen_sk_seg(points, spacing,
                                                                                        result['vol'].size())
        joint_points = points[joint_points, :]
        joint_points_pixel = joint_points / spacing[None,]
        joint_points_pixel = np.round(joint_points_pixel).astype(np.int)
        joint_points_pixel = np.unique(joint_points_pixel, axis=0)
        result['joint_points_pixel'] = joint_points_pixel
        # result['vol'] = torch.from_numpy(result['seg'].astype(np.float32))[None, None]  # for debug

        return result

    def _gen_sk_seg(self, points, spacing, shape):
        sk_seg = np.zeros(shape=shape, dtype=np.float32)
        points_pixel = points / spacing[None,]
        points_pixel = np.round(points_pixel).astype(np.int)
        points_pixel = np.unique(points_pixel, axis=0)
        zmin, zmax = np.min(points_pixel[:, 0]), np.max(points_pixel[:, 0])
        ymin, ymax = np.min(points_pixel[:, 1]), np.max(points_pixel[:, 1])
        xmin, xmax = np.min(points_pixel[:, 2]), np.max(points_pixel[:, 2])
        for p in points_pixel:
            sk_seg[p[0], p[1], p[2]] = 1

        zmin, ymin, xmin = (max(0, min(v - self._constant_shift, shape[idx % 3])) for idx, v in (zmin, ymin, xmin))
        zmax, ymax, xmax = (max(0, min(v + self._constant_shift, shape[idx % 3])) for idx, v in (zmax, ymax, xmax))
        sk_range = (zmin, ymin, xmin, zmax, ymax, xmax)
        return sk_seg, points_pixel, sk_range

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        vol, seg, sk_seg, sk_range, points_pixel, joint_points_pixel = _source_data['vol'], _source_data[
            'seg'], _source_data['sk_seg'], _source_data['sk_range'], _source_data['points_pixel'], _source_data[
                                                                           'joint_points_pixel']

        sk_seg = self._add_noise_in_sk_seg(sk_seg, points_pixel, joint_points_pixel)
        vol, sk_seg, seg = self._crop_vol(vol, sk_seg, seg, sk_range, vol.size())
        data = self._compose_data(vol, sk_seg)
        img, mask = [self._resize_torch(v[None], self._patch_size)[0] for v in [data, seg[None]]]
        results = {'img': img.detach(), 'mask': mask.detach()}
        return results

    def _crop_vol(self, vol, sk_seg, seg, sk_range, shape):
        zmin, ymin, xmin, zmax, ymax, xmax = sk_range
        zmin, ymin, xmin = [max(0, v + random.randint(-self._shift_range, self._shift_range)) for v in
                            [zmin, ymin, xmin]]
        zmax, ymax, xmax = [min(sc, v + random.randint(-self._shift_range, self._shift_range)) for v, sc in
                            zip([zmax, ymax, xmax], shape)]
        ret = [v[zmin:zmax, ymin:ymax, xmin:xmax] for v in [vol, sk_seg, seg]]
        return ret

    def _compose_data(self, vol, sk_seg):
        sk_seg = gaussian_filter(sk_seg, 1.0)
        sk_seg = torch.from_numpy(sk_seg)[None]
        vol = vol[None]
        data = torch.cat([vol, sk_seg], dim=0)
        return data

    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode='trilinear')

    def _add_noise_in_sk_seg(self, sk_seg, other_points, joint_points):
        if self._other_noise_ratio > 0 or self._joint_noise_ratio > 0:
            sk_seg = sk_seg.copy()

        if self._other_noise_ratio > 0:
            points_count = len(other_points)
            select_count = max(1, int(points_count * self._other_noise_ratio))
            select_indices = np.random.choice(np.array(range(points_count), dtype=np.int), size=select_count)
            noise_shift_range = self._other_noise_range
            points = other_points
            for idx in select_indices:
                p = points[idx]
                sk_seg[p[0], p[1], p[2]] = 0
                delta = np.random.randint(-noise_shift_range, noise_shift_range + 1, size=(3,))
                p = p + delta
                sk_seg[p[0], p[1], p[2]] = 1
        if self._joint_noise_ratio > 0 and len(joint_points) > 0:
            points_count = len(joint_points)
            select_count = max(1, int(points_count * self._joint_noise_ratio))
            select_indices = np.random.choice(np.array(range(points_count), dtype=np.int), size=select_count)
            noise_shift_range = self._joint_noise_range
            points = joint_points
            for idx in select_indices:
                p = points[idx]
                add_count = random.randint(1, self._joint_noise_count)
                for _ in range(add_count):
                    delta = np.random.randint(-noise_shift_range, noise_shift_range + 1, size=(3,))
                    p_shift = p + delta
                    sk_seg[p_shift[0], p_shift[1], p_shift[2]] = 1
        return sk_seg

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
        vol = vol_np.astype('float32')
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        return vol


@DATASETS.register_module
class CorHeatmap_Sample_Dataset(DefaultSampleDataset):
    def __init__(self, dst_list_file, win_level, win_width, patch_size, constant_shift, shift_range, sample_frequent):
        self._win_level = win_level
        self._win_width = win_width
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._constant_shift = constant_shift
        self._shift_range = shift_range
        self._data_file_list = self._load_file_list(dst_list_file)
        self.draw_idx = 1
        # self._check_data_av()

    def _check_data_av(self):
        new_lst = []
        print('check all file ----------------------------')
        for file_name in tqdm(self._data_file_list):
            try:
                self._load_source_data(file_name)
                new_lst.append(file_name)
            except Exception:
                print(f'load file erro {file_name}')

        self._data_file_list = new_lst

    def _load_file_list(self, dst_list_file):
        data_file_list = []
        with open(dst_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not os.path.exists(line):
                    print(f'{line} not exist')
                    continue
                data_file_list.append(line)
        assert len(data_file_list) != 0, 'has no avilable file in dst_list_file'
        return data_file_list

    def _load_source_data(self, file_name):  #读取npz文件
        data = np.load(file_name, allow_pickle=True)
        result = {}
        vol = torch.from_numpy(data['vol'].astype(np.float32))
        vol = self._window_array(vol)

        seg = data['seg']
        tp_seg = (seg == 1)
        fp_seg = (seg == 2)
        tp_seg = torch.from_numpy(tp_seg.astype(np.float32))
        fp_seg = torch.from_numpy(fp_seg.astype(np.float32))

        sk_seg = torch.from_numpy(data['sk_seg'].astype(np.float32))
        heart_seg = torch.from_numpy(data['heart_seg'].astype(np.float32))
        art_seg = torch.from_numpy(data['art_seg'].astype(np.float32))

        result['fp_flag'] = data['fp_flag']

        compose_data = [vol, heart_seg, art_seg, tp_seg, fp_seg, sk_seg]
        compose_data = [v[None] for v in compose_data]
        compose_data = torch.cat(compose_data, dim=0)
        result['compose_data'] = compose_data.detach()

        ha_range = data['ha_range']

        shape = compose_data.size()[1:]
        c_min = [max(0, v - self._constant_shift) for v in ha_range[:3]]
        c_max = [min(cs, v + self._constant_shift) for v, cs in zip(ha_range[3:], shape)]
        ha_range = c_min + c_max
        result['ha_range'] = ha_range

        return result

    def _sample_source_data(self, idx, source_data_info):
        info, _source_data = source_data_info
        compose_data, ha_range, fp_flag = _source_data['compose_data'], _source_data['ha_range'], _source_data[
            'fp_flag']

        compose_data = self._crop_data(compose_data, ha_range, list(compose_data.size()[1:]))
        compose_data = self._resize_torch(compose_data[None], self._patch_size)[0]
        results = {'img': compose_data.detach()}
        return results

    def _crop_data(self, compose_data, ha_range, shape):
        zmin, ymin, xmin, zmax, ymax, xmax = ha_range
        zmin, ymin, xmin = [max(0, v + random.randint(-self._shift_range, self._shift_range)) for v in
                            [zmin, ymin, xmin]]
        zmax, ymax, xmax = [min(sc, v + random.randint(-self._shift_range, self._shift_range)) for v, sc in
                            zip([zmax, ymax, xmax], shape)]
        compose_data = compose_data[:, zmin:zmax, ymin:ymax, xmin:xmax]
        return compose_data

    def _resize_torch(self, data, scale):
        return torch.nn.functional.interpolate(data, size=scale, mode='trilinear')

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
        vol = vol_np.astype('float32')
        vol = np.clip(vol, win[0], win[1])
        vol -= win[0]
        vol /= self._win_width
        return vol
