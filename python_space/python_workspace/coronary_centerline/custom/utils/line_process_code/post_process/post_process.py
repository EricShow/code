import math
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
import sitktools as st
import torch
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize
from starship.segments import region_grow
from VesselProcessUtils.ConnectBrokenPart import ConnectBrokenPart

from .ley_process import line_process
from .utils.astar import astar
from .utils.common import get_cut_box, runGPURegiongrowth, torch_argwhere

growth_interval = 200
max_value = 99999999
is_multi_show = True


def relabelConnectedComponent(im):
    return sitk.RelabelComponent(sitk.ConnectedComponent(im > 0))


def get_joint_piont(center_line):
    blank_mask = np.zeros_like(center_line)
    pos = np.where(center_line == 1)
    x, y, z = pos
    process_list = []
    for i in range(x.shape[0]):
        tmp_cube = center_line[x[i] - 1:x[i] + 2, y[i] - 1:y[i] + 2, z[i] - 1:z[i] + 2]
        if np.sum(tmp_cube) == 2:
            blank_mask[x[i], y[i], z[i]] = 1
    blank_mask = blank_mask + center_line
    return blank_mask


def get_closest_terminal(p_terminals, center):
    if len(p_terminals) == 0:
        return None
    close_dis = max_value
    for p in p_terminals:
        dis = math.sqrt(math.pow(p[0] - center[0], 2) + math.pow(p[1] - center[1], 2) + math.pow(p[2] - center[2], 2))
        if dis < close_dis:
            close_dis = dis
            near_terminal = p
    return near_terminal


# @func_line_time
def connect_sitk_mask_image(mask, select_value=1, gpu=0):

    mask_array_ori = sitk.GetArrayFromImage(mask).astype('uint8')
    result_array = np.zeros_like(mask_array_ori)
    reslt_array_modified = np.zeros_like(mask_array_ori)
    array_cut_box = get_cut_box(mask, box_mode='array')
    mask_array = mask_array_ori[array_cut_box[0]:array_cut_box[3], array_cut_box[1]:array_cut_box[4],
                                array_cut_box[2]:array_cut_box[5]]
    mask_array_return = np.zeros_like(mask_array)
    mask_connected = sitk.GetImageFromArray(mask_array.astype('uint8'))
    vessel_skeleton = skeletonize((mask_array == select_value).astype('uint8'))
    total_mask_numpy = mask_array.astype(np.uint8)
    connected_area = total_mask_numpy > 0
    vessel_relabel = relabelConnectedComponent(mask_connected == select_value)
    vessel_relabel_numpy = sitk.GetArrayFromImage(vessel_relabel)
    part_list = np.unique(vessel_relabel_numpy)
    vessel_numpy = (mask_array == select_value).astype('uint8')
    center = np.average(torch_argwhere(vessel_relabel_numpy == 1, gpu), 0).astype(np.int)
    vessel_main_array = (vessel_relabel_numpy == 1).astype('uint8')
    track_array = (vessel_main_array * 2) + (mask_array > 0).astype('uint8')
    for part in part_list:
        if part == 0 or part == 1:
            continue
        p_terminals = torch_argwhere((vessel_relabel_numpy == part) * (vessel_skeleton), gpu)
        if p_terminals.shape[0] > 30:
            connect_limit = 40
        else:
            connect_limit = 15
        #     vessel_numpy = vessel_numpy*(vessel_relabel_numpy != part)
        #     continue
        near_terminal = get_closest_terminal(p_terminals, center)
        if near_terminal is None:
            continue
        seed_array = np.zeros_like(vessel_relabel_numpy)
        seed_array[near_terminal[0] - 2:near_terminal[0] + 3, near_terminal[1] - 2:near_terminal[1] + 3,
                   near_terminal[2] - 2:near_terminal[2] + 3] = 2

        coor_array, _ = ConnectBrokenPart(track_array, seed_array, [1, 4], gpu=gpu, main_part_value=3)
        # not found a point in this space
        if coor_array[0] == 0 and coor_array[1] == 0 and coor_array[2] == 0:
            continue
        end_tuple = (coor_array[0], coor_array[1], coor_array[2])
        start_tuple = (near_terminal[0], near_terminal[1], near_terminal[2])
        path = astar((mask_array == 0), start_tuple, end_tuple, connect_limit)
        if path is not None:
            grow1 = np.zeros_like(total_mask_numpy, dtype=np.uint8)
            for x, y, z in path:
                grow1[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] = 1
            vessel_numpy = grow1 * 2 + vessel_numpy
        else:
            # revert color
            vessel_numpy[vessel_relabel_numpy == part] = 0
            mask_array_return[vessel_relabel_numpy == part] = 1
    vessel_numpy = (vessel_numpy > 0).astype('uint8')
    result_array[array_cut_box[0]:array_cut_box[3], array_cut_box[1]:array_cut_box[4],
                 array_cut_box[2]:array_cut_box[5]] = vessel_numpy
    reslt_array_modified[array_cut_box[0]:array_cut_box[3], array_cut_box[1]:array_cut_box[4],
                         array_cut_box[2]:array_cut_box[5]] = mask_array_return
    return result_array, reslt_array_modified


def optimizeVesselLabel(imSeg, imgseg_class, possible_vessel_array, lung_bbox=None, growth_factor=4, gpu=0):
    with torch.cuda.device(gpu):
        torch.cuda.empty_cache()
    ori_shape = possible_vessel_array.shape
    if lung_bbox is not None:
        imSeg = imSeg[:, :, lung_bbox[0]:lung_bbox[3]]
        imgseg_class = imgseg_class[:, :, lung_bbox[0]:lung_bbox[3]]
        possible_vessel_array = possible_vessel_array[lung_bbox[0]:lung_bbox[3], :, :]

    imSegOptim = sitk.Mask(imSeg, relabelConnectedComponent(imSeg) == 1)

    # imgseg_class = sitk.Mask(imgseg_class, imSegOptim > 0)
    # sitk.WriteImage(imSegOptim,"imSegOptim.nii.gz")
    imSegOptim_array = sitk.GetArrayFromImage(imSegOptim)
    if lung_bbox is not None:
        result = np.zeros(ori_shape, dtype='uint8')
        result[lung_bbox[0]:lung_bbox[3], :, :] = imSegOptim_array
        return result
    return imSegOptim_array

    imgseg_class_array = sitk.GetArrayFromImage(imgseg_class).astype('uint8')
    imSegOptim_array = (imSegOptim_array == 3) * imgseg_class_array + imSegOptim_array * (imSegOptim_array != 3)
    possible_vessel_array[imSegOptim_array == 3] = 1
    # region growth
    combine_array = imSegOptim_array
    # sitk.WriteImage(sitk.GetImageFromArray(combine_array.astype("uint8")),"combine.nii.gz")

    optimizelabel_array, level_label_mask = line_process(combine_array, gpu=gpu)
    optimizelabel_sitk = sitk.GetImageFromArray(optimizelabel_array)
    # sitk.WriteImage(optimizelabel_sitk,"optimizelabel.nii.gz")
    artery_numpy, artery_numpy_modified = connect_sitk_mask_image(optimizelabel_sitk, select_value=1, gpu=gpu)
    optimizelabel_array = optimizelabel_array * (artery_numpy_modified != 1) + artery_numpy_modified * 2
    optimizelabel_sitk = sitk.GetImageFromArray(optimizelabel_array)
    vein_numpy, vein_numpy_modified = connect_sitk_mask_image(optimizelabel_sitk, select_value=2, gpu=gpu)
    optimizelabel_array = vein_numpy * 2 + artery_numpy
    optimizelabel_array_modified = vein_numpy_modified * 1 + (artery_numpy_modified * 2) * (vein_numpy_modified == 0)
    optimizelabel_array += optimizelabel_array_modified * (optimizelabel_array == 0)
    if is_multi_show:
        optimizelabel_array = optimizelabel_array + level_label_mask
        optimizelabel_array[optimizelabel_array == 10] = 0
        optimizelabel_array[optimizelabel_array == 13] = 3

    if lung_bbox is not None:
        result = np.zeros(ori_shape, dtype='uint8')
        result[lung_bbox[0]:lung_bbox[3], :, :] = optimizelabel_array
        return result
    return optimizelabel_array


def rebuild_vessel(vessel_mask):
    if vessel_mask is None:
        return vessel_mask

    kel = 1
    d, h, w = vessel_mask.shape
    if d < kel * 3 or h < kel * 3 or w < kel * 3:
        return vessel_mask

    # no copy. need save cut_vessel_mask[z][y][x] before changed.
    cut_vessel_mask = vessel_mask[kel:d - kel, kel:h - kel, kel:w - kel]
    fore_mask = (cut_vessel_mask != 0)
    if not np.any(fore_mask):
        return vessel_mask

    pos = np.where(fore_mask)
    zmin, ymin, xmin, zmax, ymax, xmax = min(pos[0]), min(pos[1]
                                                          ), min(pos[2]
                                                                 ), max(pos[0]) + 1, max(pos[1]) + 1, max(pos[2]) + 1
    cut_mask = fore_mask[zmin:zmax, ymin:ymax, xmin:xmax]

    center_line = skeletonize(cut_mask)
    pos = np.where(center_line != 0)

    if len(pos) != 3:
        return vessel_mask

    zs, ys, xs = pos
    pad_r = 2 * kel + 1

    line_pixes = []
    for i in range(len(xs)):
        z, y, x = zs[i] + zmin, ys[i] + ymin, xs[i] + xmin
        line_pixes.append(cut_vessel_mask[z][y][x])

    for i in range(len(xs)):
        z, y, x = zs[i] + zmin, ys[i] + ymin, xs[i] + xmin
        vessel_mask[z:z + pad_r, y:y + pad_r, x:x + pad_r] = line_pixes[i]
    return vessel_mask
