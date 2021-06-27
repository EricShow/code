"""生成模型输入数据."""

import argparse
import glob
import os
import sys
import traceback

import numpy as np
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from skimage.morphology import skeletonize
from tqdm import tqdm
import SimpleITK as sitk
import torch
import json
import torch.nn.functional as F
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import morphology

from data import data_io as io
from data import data_process as dp
from get_skeleton import get_sk_radius, process_result, get_joint_point


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='train_data/origin/data')
    parser.add_argument('--tgt_path', type=str, default='train_data/processed_data/data/')
    parser.add_argument('--save_lst', type=str, default='train_data/processed_data/train.lst')
    args = parser.parse_args()
    return args


def gen_lst(tgt_path):
    save_file = os.path.join(tgt_path, 'train.lst')
    data_list = glob.glob(os.path.join(tgt_path, '*.npz'))
    print('num of traindata: ', len(data_list))
    with open(save_file, 'w') as f:
        for data in data_list:
            f.writelines(data + '\r\n')


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    vol = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing = spacing[::-1]
    return vol, img, spacing


def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    origin_coord = tmp_img.GetOrigin()
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, spacing, origin_coord


def recon_centerline(coord_radius, size):
    ret_array = np.zeros(shape=size, dtype=np.uint8)
    for k, v in coord_radius.items():
        k = np.array(k)
        k_start = k - 1
        k_end = k + 1 + 1
        ret_array[k_start[0]:k_end[0], k_start[1]:k_end[1], k_start[2]:k_end[2]] = 1
    return ret_array


def set_point_attribute(sk_dict, sk_seg):
    for point, radius in sk_dict.items():
        point_array = np.array(point)
        p_s = point_array - 1
        p_e = point_array + 2
        p_count = np.sum(sk_seg[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]])
        v = -1
        if p_count == 2:
            v = 1  # vertex
        elif p_count == 3:
            v = 2
        elif p_count > 3:
            v = 3
        else:
            v = -1
        sk_dict[point] = (radius, v)


def get_point_stat_info(sk_dict, sk_seg):
    stop_info = get_point_stop_info(sk_dict, sk_seg)
    neibor_point_info = {}
    point_type_info = {'joint': [], 'other': []}
    for point, info in sk_dict.items():
        radius, p_type = info
        if p_type == 3:
            point_type_info['joint'].append(point)
        else:
            point_type_info['other'].append(point)
        neibor_point_info[point] = []
        p_array = np.array(point)
        p_s = p_array - 1
        p_e = p_array + 2
        temp_cube = sk_seg[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]].copy()
        temp_cube[1, 1] = 0
        forward_point = np.argwhere(temp_cube)
        if forward_point.shape[0] != 0:
            forward_point = p_s + forward_point
            for p in forward_point:
                p = tuple(p)
                neibor_point_info[point].append(p)

    return stop_info, neibor_point_info, point_type_info


def get_point_stop_info(sk_dict, sk_seg):
    sk_seg = sk_seg.copy()
    stop_range = 5 * 2
    stop_dict = {}
    stop_info = {'end': [], 'not_end': []}
    for point, info in sk_dict.items():
        radius, p_type = info
        if p_type == 1:
            find_next_joint(point, sk_seg, sk_dict, stop_dict, range_thresh=stop_range,
                            end_point_array=np.array(point))
    for point, info in sk_dict.items():
        radius, p_type = info
        if p_type == -1:
            return
        if point in stop_dict:
            stop_info['end'].append(point)
        else:
            stop_info['not_end'].append(point)
    return stop_info


def find_next_joint(point, sk_seg, sk_dict, stop_dict, range_thresh, end_point_array):
    p_array = np.array(point)
    distance = np.linalg.norm(p_array - end_point_array)
    if distance > range_thresh:
        return
    p_s = p_array - 1
    p_e = p_array + 2
    temp_cube = sk_seg[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]]
    p_type = sk_dict[point][1]
    if p_type == 3:
        return
    temp_cube[1, 1] = 0
    stop_dict[point] = 2
    forward_point = np.argwhere(temp_cube)
    forward_point = p_s + forward_point
    for p in forward_point:
        p = tuple(p)
        find_next_joint(p, sk_seg, sk_dict, stop_dict, range_thresh, end_point_array)


def is_reverse(vol, mask):
    mask_rev = mask[::-1]

    vol_crop = vol.copy()
    vol_crop_rev = vol.copy()
    vol_crop[mask == 1] = 0
    vol_crop_rev[mask_rev == 1] = 0

    if np.sum(vol_crop_rev) < np.sum(vol_crop):
        return True
    else:
        return False


def resample(np_array, scale_factor, order=2):
    ret_array = zoom(np_array, scale_factor, mode='nearest', order=order)
    return ret_array


def window_dcm(vol, window_center, window_width):
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    vol[vol <= window_min] = window_min
    vol[vol >= window_max] = window_max
    vol -= window_min
    vol /= float(window_width)
    vol *= 255.
    return vol.astype('uint8')


def interplote_lines(lines, points, radius, step_size=0.1):
    ret_lines = []
    ret_points = list(points.copy())
    ret_radius = list(radius.copy())  

    points = np.array(points)
    lines = np.array(lines)

    for l in lines:
        p0 = points[l[0]]
        p1 = points[l[1]]
        distance = np.linalg.norm(p1 - p0)
        if distance <= step_size:
            ret_lines.append(l)
            continue
        else:

            cut_count = int(distance / step_size)
            direction = (p1 - p0) / distance
            pre_p_idx = l[0]

            r0 = radius[l[0]]
            r1 = radius[l[1]]
            r_step = (r1 - r0) / cut_count

        for idx_cut in range(cut_count):
            n_p = p0 + direction * step_size * (idx_cut + 1)
            ret_points.append(tuple(n_p))
            gen_cut_l = (pre_p_idx, len(ret_points) - 1)
            pre_p_idx = len(ret_points) - 1
            ret_lines.append(gen_cut_l)
            n_r = r0 + r_step * (idx_cut + 1)
            ret_radius.append(n_r)
        if np.linalg.norm(p1 - n_p) == 0:
            ret_points.pop()
            ret_radius.pop()
            gen_cut_l = ret_lines.pop()
            gen_cut_l = (gen_cut_l[0], l[1])
            ret_lines.append(gen_cut_l)
        else:
            gen_cut_l = (pre_p_idx, l[1])
            ret_lines.append(gen_cut_l)

    ret_lines = np.array(ret_lines, dtype=np.int)
    ret_points = np.array(ret_points)
    ret_radius = np.array(ret_radius)
    return ret_lines, ret_points, ret_radius


def get_smooth_mask(seg, sk_seg):
    try:
        mask = seg.astype(np.float)
        skeleton = sk_seg.astype(np.float)

        skeleton_dil = dp.binary_dilation(skeleton, [3, 3, 3], iter=1)
        skeleton_smooth = dp.gauss_smooth_3d(skeleton_dil, 1)

        mask_smooth = dp.gauss_smooth_3d(mask, sigma=1)
        mask_smooth = mask_smooth * 0.6 + skeleton_smooth * 0.6
        mask_smooth = np.clip(mask_smooth, 0, 1)
        return mask_smooth
    except Exception:
        traceback.print_exc()
        return None


def is_normal_point(idx_p, points, degree_info, adjs_info, distace_constrain=1.5):
    degree = degree_info[idx_p]

    pos_sample, distace_av = get_zone(idx_p, points, zone=distace_constrain)
    expect_degree = 0
    for idx_pos in pos_sample:
        adj_p = adjs_info[idx_pos]
        if not np.all(distace_av[adj_p]):
            expect_degree += 1
    return expect_degree == degree, pos_sample, expect_degree


def is_normal_joint(idx_p, degree_info, points, distace_constrain=3):
    pos, _ = get_zone(idx_p, points, distace_constrain, exclude=True)
    degrees_inner = degree_info[pos]
    pos = pos + [idx_p]
    if np.any(degrees_inner == 1):
        return False, pos
    else:
        return True, pos


def get_adjs(lines, point_count):
    assert lines.max() == (point_count - 1)
    adjs = [[] for _ in range(point_count)]
    for l in lines:
        p0 = int(l[0])
        p1 = int(l[1])
        adjs[p0].append(p1)
        adjs[p1].append(p0)
    return adjs


def get_ep_zone_outer_coord(seg, outer_range=60, slight_range=6):
    seg_outer = dp.binary_dilation(seg, [3, 3, 3], iter=outer_range)
    seg_slight_dilation = dp.binary_dilation(seg, [3, 3, 3], iter=slight_range)
    zone_av = seg_outer * (1 - seg_slight_dilation)
    coords = np.argwhere(zone_av)
    return coords


def get_ep_zone_inner(idx_p, points, radius_info, degree_info, zone=5, radius_constrain=1.5, degree_zone=5):
    pos_sample, _ = get_zone(idx_p, points, zone)
    if degree_zone != zone:
        pos_degree_sample, _ = get_zone(idx_p, points, degree_zone)
    else:
        pos_degree_sample = pos_sample
    degrees_inner = degree_info[pos_degree_sample]
    select_radius = radius_info[pos_sample]
    if select_radius.mean() > radius_constrain or np.any(degrees_inner >= 3):
        return False, pos_sample
    else:
        return True, pos_sample


def get_zone(idx_p, points, zone, exclude=False):
    p = points[idx_p]
    distance = np.linalg.norm(points - p[None, :], axis=1)
    distace_av = distance <= zone
    pos_sample = np.argwhere(distace_av)
    if exclude:
        pos_sample = [v[0] for v in pos_sample if v[0] != idx_p]
    else:
        pos_sample = [v[0] for v in pos_sample]

    return pos_sample, distace_av


def get_isot_sk_seg(ori_shape, spacing, tgt_spacing, points):
    ori_shape = np.array(ori_shape)
    tgt_shape = ori_shape * spacing / tgt_spacing
    tgt_shape = np.round(tgt_shape).astype(np.int)
    ret_array = np.zeros(shape=tuple(tgt_shape), dtype=np.uint8)
    points = points / tgt_spacing[None, :]
    points = np.round(points).astype(np.int)
    for pi in points:
        p_s = pi - 0
        p_e = pi + 1
        ret_array[p_s[0]:p_e[0], p_s[1]:p_e[1], p_s[2]:p_e[2]] = 1
    return ret_array


if __name__ == '__main__':
    args = parse_args()
    src_dir = args.src_path
    tgt_dir = args.tgt_path
    save_lst_file = args.save_lst

    src_dcm_dir = os.path.join(src_dir, 'dicom')
    src_seg_dir = os.path.join(src_dir, 'nii')
    src_centerline_dir = os.path.join(src_dir, 'centerline')
    os.makedirs(tgt_dir, exist_ok=True)

    ret_lines = []
    for seg_file in tqdm(os.listdir(src_seg_dir)):
        try:
            if '-seg.nii.gz' not in seg_file:
                print(f'abnormal seg file :{seg_file}')
                continue

            # if seg_file != 'CN010002-01466650-28880-seg.nii.gz':
            #     continue
            pid = seg_file.replace('-seg.nii.gz', '')
            vol_dir = os.path.join(src_dcm_dir, pid + '.nii.gz')
            save_file = pid + '.npz'
            save_file = os.path.join(tgt_dir, save_file)
            seg_file = os.path.join(src_seg_dir, seg_file)
            centerline_json_file = os.path.join(src_centerline_dir, pid + '_centerline.json')
            centerline_seg_file = os.path.join(src_centerline_dir, pid + '_centerline.nii.gz')

            flag = True
            for file in [vol_dir, centerline_json_file]:
                if not os.path.exists(file):
                    print(f'dont find file {file}')
                    flag = False
            if not flag:
                continue

            # vol, img, spacing_vol = load_scans(vol_dir)
            vol, spacing_vol, origin_coord = load_nii(vol_dir)
            seg, spacing, _ = load_nii(seg_file)
            seg = seg.astype(np.float32)

            spacing_missing = np.linalg.norm((np.array(spacing_vol) - np.array(spacing)))
            if vol.shape != seg.shape or spacing_missing > 0.1:
                print(f'shape :{vol.shape}!={seg.shape}, spacing: {spacing_vol}!={spacing}')
                continue

            sk_json = json.load(open(centerline_json_file))
            # isot_sk_seg = sitk.GetArrayFromImage(sitk.ReadImage(centerline_seg_file))
            spacing = np.array(spacing_vol)
            new_spacing = np.array([0.5, 0.5, 0.5])
            scale_factor = spacing / new_spacing

            points = sk_json['points']
            points = np.array(points)
            radius_info = sk_json['radius']
            radius_info = np.array(radius_info)
            lines = sk_json['lines']
            lines = np.array(lines, dtype=np.int)
            lines = np.unique(lines, axis=0)
            roots = sk_json['roots']
            lines, points, radius_info = interplote_lines(lines, points, radius_info)
            points = points[:, ::-1]
            degree_info = np.bincount(lines.flatten())
            assert points.shape[0] == degree_info.shape[0] and radius_info.shape[0] == points.shape[0]

            adjs_info = get_adjs(lines, points.shape[0])

            isot_sk_seg = get_isot_sk_seg(vol.shape, spacing, new_spacing, points)

            isot_seg = resample(seg, scale_factor)
            isot_seg = (isot_seg > 0.5).astype(np.uint8)

            # sitk_img = sitk.GetImageFromArray(isot_sk_seg)
            # sitk.WriteImage(sitk_img, save_file.replace('.npz', '-sk_seg.nii.gz'))
            #
            # sitk_img = sitk.GetImageFromArray(isot_seg)
            # sitk.WriteImage(sitk_img, save_file.replace('.npz', '-seg.nii.gz'))

            if isot_sk_seg.shape != isot_seg.shape:
                print(f'isot_sk_seg and isot_seg\'s shape :{isot_sk_seg.shape}!={isot_seg.shape}')
                continue

            ep_zone_outer = get_ep_zone_outer_coord(isot_seg, outer_range=60, slight_range=6)
            #reduce outer ep point count
            outer_av = ep_zone_outer % 4 == 0
            outer_av = np.all(outer_av, axis=1)
            ep_zone_outer = ep_zone_outer[outer_av, :]

            ep_zone_outer = ep_zone_outer * new_spacing[None, :]

            abnormal_zone = []
            ep_zone_inner = []
            other_zone_ep_inner = []
            joint_zone = []
            other_zone_joint = []
            ep_zone_inner_corase = []
            for idx_p, p in enumerate(points):
                degree = degree_info[idx_p]
                normal_flag, zone, expect_degree = is_normal_point(idx_p, points, degree_info, adjs_info,
                                                                   distace_constrain=1.5)

                if degree == 1:
                    ep_flag, ep_zone = get_ep_zone_inner(idx_p, points, radius_info, degree_info, zone=5,
                                                         radius_constrain=0.75,
                                                         degree_zone=5)
                    ep_zone_inner_corase.extend(ep_zone)
                    if normal_flag and ep_flag:
                        ep_zone_inner.extend(ep_zone)

                if expect_degree >= 3:
                    flag, _ = is_normal_joint(idx_p, degree_info, points, distace_constrain=1.5)
                    if flag:
                        joint_zone.append(idx_p)
                elif expect_degree == 2:
                    if degree == 2:
                        other_zone_joint.append(idx_p)

            roots_ep_zone = []
            for idx_p in roots:
                _zone, _ = get_zone(idx_p, points, zone=5)
                roots_ep_zone.extend(_zone)

            need_setlize_list = [abnormal_zone, ep_zone_inner, other_zone_ep_inner, joint_zone, other_zone_joint,
                                 ep_zone_inner_corase, roots_ep_zone]
            setlize_list = (set(n_l) for n_l in need_setlize_list)
            abnormal_zone, ep_zone_inner, other_zone_ep_inner, joint_zone, other_zone_joint, ep_zone_inner_corase, roots_ep_zone = setlize_list

            ep_zone_inner_corase = ep_zone_inner_corase - roots_ep_zone
            ep_zone_inner = ep_zone_inner - roots_ep_zone
            other_zone_ep_inner = set(list(range(points.shape[0]))) - ep_zone_inner_corase - ep_zone_inner

            # need_sub_set = [ep_zone_inner, other_zone_ep_inner, joint_zone, other_zone_joint]
            # subed_set = (s - abnormal_zone for s in need_sub_set)
            # ep_zone_inner, other_zone_ep_inner, joint_zone, other_zone_joint = subed_set
            print(
                f'joint zone: {len(joint_zone)} , other_zone_joint: {len(other_zone_joint)}, ep_zone_inner: {len(ep_zone_inner)},ep_zone_outer: {len(ep_zone_outer)}, other_zone_ep_inner: {len(other_zone_ep_inner)}')

            smooth_seg = get_smooth_mask(isot_seg, isot_sk_seg)
            if smooth_seg is None:
                print(f'{pid} get smooth seg failed')
                continue
            smooth_seg = resample(smooth_seg, new_spacing / spacing)
            smooth_seg = smooth_seg * 255
            smooth_seg = smooth_seg.astype(np.uint8)

            if smooth_seg.shape != vol.shape:
                print(f'smooth_seg and vol\'s shape :{smooth_seg.shape}!={vol.shape}')
                continue

            seg = seg.astype(np.uint8)
            spacing = spacing_vol
            ep_info = {'ep': {'inner': ep_zone_inner, 'outer': ep_zone_outer}, 'no_ep': other_zone_ep_inner}
            joint_info = {'joint': joint_zone, 'no_joint': other_zone_joint}
            sk_info = {'points': points, 'lines': lines, 'roots': roots, 'degree_info': degree_info,
                       'radius_info': radius_info, 'adjs_info': adjs_info}
            assert seg.shape == vol.shape and points.shape[0] == degree_info.shape[0] and points.shape[0] == \
                   radius_info.shape[0] and smooth_seg.shape == vol.shape

            np.savez_compressed(save_file, vol=vol, seg=seg, smooth_seg=smooth_seg, spacing=spacing, ep_info=ep_info,
                                joint_info=joint_info, sk_info=sk_info)

            # sitk_img = sitk.GetImageFromArray(sk_seg)
            # sitk.WriteImage(sitk_img, save_file.replace('.npz', '-sk_seg.nii.gz'))
            #
            # sitk_img = sitk.GetImageFromArray(seg)
            # sitk.WriteImage(sitk_img, save_file.replace('.npz', '-seg.nii.gz'))

            # sitk_img = sitk.GetImageFromArray(color_seg)
            # sitk.WriteImage(sitk_img, save_file.replace('.npz', '-seg_recon.nii.gz'))
            lst_line = save_file + '\n'
            ret_lines.append(lst_line)
        except:
            traceback.print_exc()
            print(f'find erro in {seg_file}')
    print(f'num of processed data: {len(ret_lines)} ')
    with open(save_lst_file, 'w') as f:
        f.writelines(ret_lines)
