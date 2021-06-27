import copy
import math
import random

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from nibabel.affines import to_matvec
from scipy.ndimage import affine_transform, morphology, zoom
from scipy.ndimage.filters import gaussian_filter

# from nipy.algorithms.registration.affine import Affine
from .cal_short_vertexs import get_long_short_axis


def resize_cpu(image, dims, order=0):
    """rescale the dims, such as isotropic transform.

    :param image: 2D or 3D image, dim = depth/channel, height, width
    :param dims: the expected dim for output image
    :param order: order value [0,5], means use nearest or  b-spline method
    :return: resampled image
    """
    #
    # if dims[0] == image.shape[0]:
    #     new_img = np.zeros(dims, dtype=image.dtype)
    #     new_img.fill(image.min())
    #     for slice_id in range(image.shape[0]):
    #         new_img[slice_id] = cv2.resize(np.array(image[slice_id]), (dims[2], dims[1]),
    #                                        interpolation=cv2.INTER_NEAREST)
    # else:
    image_rs = zoom(
        image, np.array(dims) / np.array(image.shape, dtype=np.float32), order=order, mode='constant', cval=image.min()
    )
    return image_rs


def l2_distance(pos_a, pos_b):
    return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)


def cal_nodule_axes(mask, spacing):
    sum_z = np.sum(mask, axis=(1, 2))
    idx_max_z = np.argmax(sum_z)
    patch_2d = mask[idx_max_z]
    patch_2d[patch_2d > 0] = 1

    out = get_long_short_axis(patch_2d)
    if out is not None:
        _, long_vertexs, short_vertexs = out

        long_axis = l2_distance(long_vertexs[0], long_vertexs[1]) * spacing
        short_axis = l2_distance(short_vertexs[0], short_vertexs[1]) * spacing
        return [long_axis, short_axis]

    else:
        return [0, 0]

    # mask_copy = np.copy(mask)
    # mask_copy[mask_copy > 1e-5] = 1
    # long_axis_list = []
    # short_axis_list = []
    # for z in range(mask.shape[0]):
    #     patch_2d = mask_copy[z]
    #
    #     out = get_long_short_axis(patch_2d)
    #     if out != None:
    #         _, long_vertexs, short_vertexs = out
    #
    #         long_axis = l2_distance(long_vertexs[0], long_vertexs[1])
    #         short_axis = l2_distance(short_vertexs[0], short_vertexs[1])
    #         long_axis_list.append(long_axis)
    #         short_axis_list.append(short_axis)
    #     else:
    #         long_axis_list.append(0)
    #         short_axis_list.append(0)
    #
    # max_idx = long_axis_list.index(max(long_axis_list))
    # return [long_axis_list[max_idx], short_axis_list[max_idx]]

    # # debug
    # print("long_vertexs", long_vertexs)
    # print("short_vertexs", short_vertexs)
    # print("long_axis", long_axis)
    # print("short_axis", short_axis)
    # import matplotlib.pyplot as plt
    # plt.imshow(patch_2d)
    # plt.show()


# def affine_transform_3d_cpu(data, tar_size, scale=(1.0, 1.0, 1.0), angle=(0, 0, 0), shift=(0, 0, 0), order=1,
#                             trans_mode="nearest"):
#     '''
#     :param data: D * H * W
#     :param tar_size:
#     :param angle: 3
#     :param scale: 3
#     :param shift: 3
#     :param trans_orders: 0-5
#     :param trans_mode: nearest,
#     :return: trans_data
#     '''
#
#     trans_data = np.zeros(tuple(tar_size), data.dtype)
#     src_size = data.shape
#     aff = Affine()
#     aff.rotation = (angle[0] * np.pi / 180.0, angle[1] * np.pi / 180.0, angle[2] * np.pi / 180.0)
#     aff.scaling = scale
#     aff.translation = (shift[2], shift[1], shift[0])
#     mat_r, mat_t = to_matvec(aff.inv().as_affine())
#
#     # now the rotation center is zero, transfer to image center,
#     # need change translation
#     # T_new = T - R * Cr + Cf, (Cr = Cf)
#     # then, T_new = T - R*C + C
#     center_r = np.array([src_size[0] / 2, src_size[1] / 2, src_size[2] / 2])
#     center_f = np.array([tar_size[0] / 2, tar_size[1] / 2, tar_size[2] / 2])
#     mat_t = (mat_t - np.dot(mat_r, center_f) + center_r)
#     affine_transform(data, mat_r, mat_t, output_shape=tuple(tar_size),
#                      output=trans_data, order=order, mode=trans_mode)
#     return trans_data


def draw_gauss_map_2d(img_size, pt, sigma):
    # Draw a 2D gaussian

    img = np.zeros(img_size, dtype='float32')
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[1] - 3 * sigma), int(pt[0] - 3 * sigma)]
    br = [int(pt[1] + 3 * sigma + 1), int(pt[0] + 3 * sigma + 1)]
    if (ul[0] > img_size[1] or ul[1] >= img_size[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        print('----------------Draw Gauss Map Error!----------------')
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img_size[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img_size[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img_size[1])
    img_y = max(0, ul[1]), min(br[1], img_size[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def flood_fill_3d(vol, mask, start_point, fill_with=1):
    sizez = vol.shape[0] - 1
    sizex = vol.shape[1] - 1
    sizey = vol.shape[2] - 1
    items = []

    def enqueue(item):
        items.append(item)

    def dequeue():
        s = items.pop()
        return s

    mask_sp = copy.deepcopy(start_point)
    mask_ep = copy.deepcopy(start_point)

    mask[start_point[0], start_point[1], start_point[2]] = fill_with
    enqueue((start_point[0], start_point[1], start_point[2]))
    seed_value = vol[start_point[0], start_point[1], start_point[2]]
    # print(seed_value)
    while not items == []:
        z, x, y = dequeue()
        if x < sizex:
            tvoxel = vol[z, x + 1, y]
            if mask[z, x + 1, y] != fill_with and tvoxel == seed_value:
                mask[z, x + 1, y] = fill_with
                if x + 1 > mask_ep[1]:
                    mask_ep[1] = x + 1
                enqueue((z, x + 1, y))
        if x > 0:
            tvoxel = vol[z, x - 1, y]
            if mask[z, x - 1, y] != fill_with and tvoxel == seed_value:
                mask[z, x - 1, y] = fill_with
                if x - 1 < mask_sp[1]:
                    mask_sp[1] = x - 1
                enqueue((z, x - 1, y))
        if y < sizey:
            tvoxel = vol[z, x, y + 1]
            if mask[z, x, y + 1] != fill_with and tvoxel == seed_value:
                mask[z, x, y + 1] = fill_with
                if y + 1 > mask_ep[2]:
                    mask_ep[2] = y + 1
                enqueue((z, x, y + 1))
        if y > 0:
            tvoxel = vol[z, x, y - 1]
            if mask[z, x, y - 1] != fill_with and tvoxel == seed_value:
                mask[z, x, y - 1] = fill_with
                if y - 1 < mask_sp[2]:
                    mask_sp[2] = y - 1
                enqueue((z, x, y - 1))
        if z < sizez:
            tvoxel = vol[z + 1, x, y]
            if mask[z + 1, x, y] != fill_with and tvoxel == seed_value:
                mask[z + 1, x, y] = fill_with
                if z + 1 > mask_ep[0]:
                    mask_ep[0] = z + 1
                enqueue((z + 1, x, y))
        if z > 0:
            tvoxel = vol[z - 1, x, y]
            if mask[z - 1, x, y] != fill_with and tvoxel == seed_value:
                mask[z - 1, x, y] = fill_with
                if z - 1 < mask_sp[0]:
                    mask_sp[0] = z - 1
                enqueue((z - 1, x, y))

    return mask_sp, mask_ep


def draw_gauss_map_3d(vol_size, sigma_xy, sigma_z):
    # Draw a 3D gaussian
    size_z = vol_size[0]
    z = np.arange(0, size_z, 1, float)
    z0 = size_z // 2
    g = np.exp(-((z - z0)**2) / (2 * sigma_z**2))

    center_xy = [vol_size[1] / 2.0, vol_size[2] / 2.0]
    size_xy = (vol_size[1], vol_size[2])
    gauss_map_2d = draw_gauss_map_2d(size_xy, center_xy, sigma_xy)

    vol = np.zeros(vol_size, dtype='float32')
    for i in range(size_z):
        vol[i] = g[i] * gauss_map_2d

    return vol


def cal_iou(bbox_a, bbox_b, iou_little=False):
    """xmin, ymin, xmax, ymax.

    :param bbox_a:
    :param bbox_b:
    :return:
    """
    epsilon = 1e-5
    xmin = max(bbox_a[1], bbox_b[1])
    ymin = max(bbox_a[2], bbox_b[2])
    xmax = min(bbox_a[3], bbox_b[3])
    ymax = min(bbox_a[4], bbox_b[4])
    if xmin > xmax or ymin > ymax:
        return 0
    area_0 = (bbox_a[3] - bbox_a[1]) * (bbox_a[4] - bbox_a[2])
    area_1 = (bbox_b[3] - bbox_b[1]) * (bbox_b[4] - bbox_b[2])
    union_area = (xmax - xmin) * (ymax - ymin)
    if iou_little:
        return union_area / float(min(area_0, area_1) + epsilon)
    else:
        return union_area / float(area_1 + area_0 - union_area + epsilon)


def vol_copy(vol_tgt, vol_src, start_point):
    size_tgt = vol_tgt.shape
    size_src = vol_src.shape
    tgt_s = list(start_point)
    tgt_e = [tgt_s[i] + size_src[i] for i in range(len(tgt_s))]
    src_s = [0, 0, 0]
    src_e = list(size_src)
    for i in range(len(tgt_s)):
        if tgt_s[i] < 0:
            src_s[i] += (0 - tgt_s[i])
            tgt_s[i] = 0
        if tgt_e[i] > size_tgt[i]:
            src_e[i] -= (tgt_e[i] - size_tgt[i])
            tgt_e[i] = size_tgt[i]
        if src_e[i] <= src_s[i] or tgt_e[i] <= tgt_s[i]:
            return

    vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]] = \
        vol_src[src_s[0]:src_e[0], src_s[1]:src_e[1], src_s[2]:src_e[2]]


def vol_add(vol_tgt, vol_src, start_point):
    size_tgt = vol_tgt.shape
    size_src = vol_src.shape
    tgt_s = list(start_point)
    tgt_e = [tgt_s[i] + size_src[i] for i in range(len(tgt_s))]
    src_s = [0, 0, 0]
    src_e = list(size_src)
    for i in range(len(tgt_s)):
        if tgt_s[i] < 0:
            src_s[i] += (0 - tgt_s[i])
            tgt_s[i] = 0
        if tgt_e[i] > size_tgt[i]:
            src_e[i] -= (tgt_e[i] - size_tgt[i])
            tgt_e[i] = size_tgt[i]
        if src_e[i] <= src_s[i] or tgt_e[i] <= tgt_s[i]:
            return

    vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]] += \
        vol_src[src_s[0]:src_e[0], src_s[1]:src_e[1], src_s[2]:src_e[2]]


def vol_max(vol_tgt, vol_src, start_point):
    size_tgt = vol_tgt.shape
    size_src = vol_src.shape
    tgt_s = list(start_point)
    tgt_e = [tgt_s[i] + size_src[i] for i in range(len(tgt_s))]
    src_s = [0, 0, 0]
    src_e = list(size_src)
    for i in range(len(tgt_s)):
        if tgt_s[i] < 0:
            src_s[i] += (0 - tgt_s[i])
            tgt_s[i] = 0
        if tgt_e[i] > size_tgt[i]:
            src_e[i] -= (tgt_e[i] - size_tgt[i])
            tgt_e[i] = size_tgt[i]
        if src_e[i] <= src_s[i] or tgt_e[i] <= tgt_s[i]:
            return

    patch_tgt = vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]]
    patch_src = vol_src[src_s[0]:src_e[0], src_s[1]:src_e[1], src_s[2]:src_e[2]]
    patch_max = (np.array((patch_tgt, patch_src))).max(axis=0)
    vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]] = patch_max


def vol_max_gpu(vol_tgt, vol_src, start_point):
    size_tgt = [vol_tgt.size(0), vol_tgt.size(1), vol_tgt.size(2)]
    size_src = [vol_src.size(0), vol_src.size(1), vol_src.size(2)]
    tgt_s = list(start_point)
    tgt_e = [tgt_s[i] + size_src[i] for i in range(len(tgt_s))]
    src_s = [0, 0, 0]
    src_e = list(size_src)
    for i in range(len(tgt_s)):
        if tgt_s[i] < 0:
            src_s[i] += (0 - tgt_s[i])
            tgt_s[i] = 0
        if tgt_e[i] > size_tgt[i]:
            src_e[i] -= (tgt_e[i] - size_tgt[i])
            tgt_e[i] = size_tgt[i]
        if src_e[i] <= src_s[i] or tgt_e[i] <= tgt_s[i]:
            return

    patch_tgt = vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]]
    patch_src = vol_src[src_s[0]:src_e[0], src_s[1]:src_e[1], src_s[2]:src_e[2]]
    patch_max = torch.cat((patch_tgt.unsqueeze(0), patch_src.unsqueeze(0)), dim=0).max(dim=0)[0]
    vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]] = patch_max


def fill_hole(mask_in):
    mask = mask_in.copy()
    mask[mask >= 128] = 255
    mask[mask <= 128] = 0
    size_3d = mask.shape
    mask_out = mask_in.astype('int16')
    for i in range(size_3d[0]):
        mask_tmp = np.zeros([size_3d[1] + 2, size_3d[2] + 2], np.uint8)
        cv2.floodFill(mask[i], mask_tmp, (0, 0), 255)
        mask[i] = 255 - mask[i]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_out[i] += cv2.dilate(mask[i], kernel, iterations=1)

    mask_out = np.clip(mask_out, 0, 255)
    return mask_out.astype('uint8')


def fill_hole_separate(mask):

    def get_seed(mask):
        if 1 not in mask:
            return []
        else:
            seed = np.argwhere(mask == 1)[0]
            return seed

    mask_bin = np.zeros(mask.shape, dtype='uint8')
    mask_bin[mask > 0] = 1
    mask_out = np.zeros(mask.shape, dtype='float32')
    while (True):
        # # debug
        # save_nii(mask * 255, "temp/mask_o")

        seed = get_seed(mask_bin)
        if seed == []:
            break
        single_mask = np.zeros(mask_bin.shape, dtype='uint8')
        flood_fill_3d(mask_bin, single_mask, seed)
        mask_bin = mask_bin - single_mask

        single_mask_cp = mask.copy()
        single_mask_cp[single_mask == 0] = 0
        single_mask_cp = fill_hole(single_mask_cp)

        vol_max(mask_out, single_mask_cp, (0, 0, 0))
    return mask_out


def vol_crop(vol, start_point=None, end_point=None, bbox_3d=None, pad_value=0):
    if bbox_3d is not None:
        start_point = [bbox_3d[0], bbox_3d[1], bbox_3d[2]]
        end_point = [bbox_3d[3], bbox_3d[4], bbox_3d[5]]
    if start_point is None or end_point is None:
        return None
    start_point = [int(round(start_point[i])) for i in range(len(start_point))]
    end_point = [int(round(end_point[i])) for i in range(len(end_point))]

    vol_size = vol.shape
    patch_size = (end_point[0] - start_point[0], end_point[1] - start_point[1], end_point[2] - start_point[2])
    patch = np.ones(patch_size, dtype=vol.dtype) * pad_value

    src_s = list(start_point)
    src_e = list(end_point)
    tgt_s = [0, 0, 0]
    tgt_e = list(patch_size)
    for i in range(len(tgt_s)):
        if src_s[i] < 0:
            tgt_s[i] += (0 - src_s[i])
            src_s[i] = 0
        if src_e[i] > vol_size[i]:
            tgt_e[i] -= (src_e[i] - vol_size[i])
            src_e[i] = vol_size[i]
        if src_e[i] <= src_s[i] or tgt_e[i] <= tgt_s[i]:
            return None

    patch[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]] = \
        vol[src_s[0]:src_e[0], src_s[1]:src_e[1], src_s[2]:src_e[2]]

    return patch


def binary_opening(vol, iter=1):
    str_3D = morphology.generate_binary_structure(3, 1)
    vol_out = morphology.binary_opening(vol, str_3D, iterations=iter)
    vol_out = vol_out.astype('uint8')
    return vol_out


def grey_dilation(vol, size):
    vol_out = morphology.grey_dilation(vol, size=size)
    return vol_out


def binary_closing(vol, iter=1):
    str_3D = morphology.generate_binary_structure(3, 1)
    vol_out = morphology.binary_closing(vol, str_3D, iterations=iter)
    vol_out = vol_out.astype('uint8')
    return vol_out


def binary_erosion(vol, size, iter=1):
    str_3D = np.ones(size, dtype='float32')
    vol_out = morphology.binary_erosion(vol, str_3D, iterations=iter)
    vol_out = vol_out.astype('uint8')
    return vol_out


def binary_dilation(vol, size, iter=1):
    str_3D = np.ones(size, dtype='float32')
    vol_out = morphology.binary_dilation(vol, str_3D, iterations=iter)
    vol_out = vol_out.astype('uint8')
    return vol_out


def get_nodule_seed(label_patch):
    patch_shape = label_patch.shape
    patch_center = [int(size * 0.5) for size in patch_shape]
    center_start = [patch_center[0], patch_center[1] - 2, patch_center[2] - 2]
    center_end = [patch_center[0] + 1, patch_center[1] + 3, patch_center[2] + 3]
    center_area = label_patch[center_start[0]:center_end[0], center_start[1]:center_end[1],
                              center_start[2]:center_end[2]]

    if 1 not in center_area:
        return []
    else:
        seed = np.argwhere(center_area == 1)[0]
        seed += center_start
        return seed


def remove_other_nodules(label_patch, seed):
    # # debug
    # label_patch[1:4, 2:7, 3:15] = 1

    mask = np.zeros(label_patch.shape, dtype='uint8')
    flood_fill_3d(label_patch, mask, seed)
    # # debug
    # save_nii(label_patch, "/home/tx-deepocean/program/ssd/segmentation/temp/label")
    # save_nii(mask, "/home/tx-deepocean/program/ssd/segmentation/temp/mask")

    return mask


def get_bbox_size(bbox_3d):
    return (bbox_3d[3] - bbox_3d[0], bbox_3d[4] - bbox_3d[1], bbox_3d[5] - bbox_3d[2])


def get_bbox_center(bbox_3d):
    bbox_size = get_bbox_size(bbox_3d)
    bbox_center = (bbox_3d[2] + bbox_size[0] * 0.5, bbox_3d[1] + bbox_size[1] * 0.5, bbox_3d[0] + bbox_size[2] * 0.5)
    return bbox_center


def get_bbox_3d_axes(mask, bbox_3d_list, spacing):
    axes_list = []
    for i, bbox_3d in enumerate(bbox_3d_list):
        mask_patch = mask[bbox_3d[2]:bbox_3d[5] + 1, bbox_3d[1]:bbox_3d[4], bbox_3d[0]:bbox_3d[3]]
        seed = get_nodule_seed(mask_patch)
        if seed != []:
            single_mask_patch = remove_other_nodules(mask_patch, seed)
            axes = cal_nodule_axes(single_mask_patch, spacing)
            axes_list.append(axes)
        else:
            # print("------------No Mask!--------------")
            axes = cal_nodule_axes(mask_patch, spacing)
            axes_list.append(axes)
    return axes_list


def get_bbox_2d_axes(mask, bbox_2d_list, spacing=1.0):
    # bbox_2d: cls_name, xmin, ymin, xmax, ymax, z
    axes_list = []
    for bbox_2d in bbox_2d_list:
        z = int(bbox_2d[5])
        mask_patch = mask[z:z + 1, int(bbox_2d[2]):int(bbox_2d[4]), int(bbox_2d[1]):int(bbox_2d[3])]
        axes = cal_nodule_axes(mask_patch, spacing)
        axes_list.append(axes)
    return axes_list


def get_2d_axes_map(vol_size, bbox_2d_list, axes_list):

    def get_bbox_2d_center(bbox_2d):
        y_c = bbox_2d[2] + (bbox_2d[4] - bbox_2d[2]) * 0.5
        x_c = bbox_2d[1] + (bbox_2d[3] - bbox_2d[1]) * 0.5
        return [y_c, x_c]

    def get_bbox_2d_size(bbox_2d):
        return [bbox_2d[4] - bbox_2d[2] + 1, bbox_2d[3] - bbox_2d[1] + 1]

    long_axes_map = np.ones(vol_size, dtype='float32') * 1000
    short_axes_map = np.ones(vol_size, dtype='float32') * 1000
    size_z_map = np.ones(vol_size, dtype='float32') * 1000
    axes_loss_mask = np.zeros(vol_size, dtype='uint8')
    for i, bbox_2d in enumerate(bbox_2d_list):
        bbox_center = get_bbox_2d_center(bbox_2d)
        bbox_size = get_bbox_2d_size(bbox_2d)
        sp = [int(bbox_center[i] - bbox_size[i] / 6) for i in range(len(bbox_size))]
        ep = [int(bbox_center[i] + bbox_size[i] / 6) + 1 for i in range(len(bbox_size))]
        z = int(bbox_2d[5])
        long_patch = long_axes_map[z, sp[0]:ep[0], sp[1]:ep[1]]
        short_patch = short_axes_map[z, sp[0]:ep[0], sp[1]:ep[1]]
        size_z_patch = size_z_map[z, sp[0]:ep[0], sp[1]:ep[1]]
        long_axis = axes_list[i][0]
        short_axis = axes_list[i][1]
        size_z = bbox_2d[6]
        long_patch[long_patch > long_axis] = long_axis
        short_patch[short_patch > short_axis] = short_axis
        size_z_patch[size_z_patch > size_z] = size_z
        long_axes_map[z, sp[0]:ep[0], sp[1]:ep[1]] = long_patch
        short_axes_map[z, sp[0]:ep[0], sp[1]:ep[1]] = short_patch
        size_z_map[z, sp[0]:ep[0], sp[1]:ep[1]] = size_z_patch

        sm = [int(bbox_center[i] - bbox_size[i] / 2) for i in range(len(bbox_size))]
        em = [int(bbox_center[i] + bbox_size[i] / 2) + 1 for i in range(len(bbox_size))]
        axes_loss_mask[z, sm[0]:em[0], sm[1]:em[1]] = 1

    long_axes_map[long_axes_map > (1000 - 1)] = 0
    short_axes_map[short_axes_map > (1000 - 1)] = 0
    size_z_map[size_z_map > (1000 - 1)] = 0
    long_axes_map[(long_axes_map < 1e-5) * (axes_loss_mask == 1)] = -1
    short_axes_map[(short_axes_map < 1e-5) * (axes_loss_mask == 1)] = -1
    size_z_map[(size_z_map < 1e-5) * (axes_loss_mask == 1)] = -1

    return long_axes_map, short_axes_map, size_z_map


#
# def get_size_z_map(vol_size, bbox_2d_list):
#     size_z_map = np.ones(vol_size, dtype="uint16") * 1000
#     for bbox_3d in bbox_3d_list:
#         size_z = bbox_3d[5] - bbox_3d[2] + 1
#         bbox_center = get_bbox_center(bbox_3d)
#         bbox_size = get_bbox_size(bbox_3d)
#         sp = [bbox_3d[2], int(bbox_center[1] - bbox_size[1] / 6), int(bbox_center[2] - bbox_size[2] / 6)]
#         ep = [bbox_3d[5] + 1, int(bbox_center[1] + bbox_size[1] / 6), int(bbox_center[2] + bbox_size[2] / 6)]
#         size_z_patch = size_z_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
#         size_z_patch[size_z_patch > size_z] = size_z
#         size_z_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = size_z_patch
#     return size_z_map


def get_3d_axes_map(vol_size, bbox_3d_list, axes_list, thickness, spacing_xy=0.7):

    def add_bbox_z(bbox_3d):
        bbox_size = get_bbox_size(bbox_3d)
        size_z = bbox_size[1] * spacing_xy / thickness
        z_c = bbox_3d[2] + bbox_size[0] / 2.0
        z_s = int(round(z_c - size_z / 2.0))
        z_e = int(round(z_c + size_z / 2.0))
        return [bbox_3d[0], bbox_3d[1], z_s, bbox_3d[3], bbox_3d[4], z_e]

    long_axes_map = np.ones(vol_size, dtype='float32') * 1000
    short_axes_map = np.ones(vol_size, dtype='float32') * 1000
    axes_loss_mask = np.zeros(vol_size, dtype='uint8')
    for i, bbox_3d in enumerate(bbox_3d_list):
        bbox_new = add_bbox_z(bbox_3d)
        bbox_center = get_bbox_center(bbox_new)
        bbox_size = get_bbox_size(bbox_new)
        sp = [int(bbox_center[i] - bbox_size[i] / 6) for i in range(len(bbox_size))]
        ep = [int(bbox_center[i] + bbox_size[i] / 6) + 1 for i in range(len(bbox_size))]
        long_patch = long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
        short_patch = short_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
        long_axis = axes_list[i][0]
        short_axis = axes_list[i][1]
        long_patch[long_patch > long_axis] = long_axis
        short_patch[short_patch > short_axis] = short_axis
        long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = long_patch
        short_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = short_patch

        sm = [int(bbox_center[i] - bbox_size[i] / 2) for i in range(len(bbox_size))]
        em = [int(bbox_center[i] + bbox_size[i] / 2) + 1 for i in range(len(bbox_size))]
        axes_loss_mask[sm[0]:em[0], sm[1]:em[1], sm[2]:em[2]] = 1

    long_axes_map[long_axes_map > (1000 - 1)] = 0
    short_axes_map[short_axes_map > (1000 - 1)] = 0
    long_axes_map[(long_axes_map < 1e-5) * (axes_loss_mask == 1)] = -1
    short_axes_map[(short_axes_map < 1e-5) * (axes_loss_mask == 1)] = -1
    # long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = min(axes_list[i][0],
    #                                                            long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]])
    # short_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = min(axes_list[i][1],
    #                                                             short_axes_map[sp[0]:ep[0], sp[1]:ep[1],
    #                                                             sp[2]:ep[2]])

    # long_axes_map = np.zeros(vol_size, dtype="float32")
    # short_axes_map = np.zeros(vol_size, dtype="float32")
    # for i, bbox_3d in enumerate(bbox_3d_list):
    #     bbox_new = add_bbox_z(bbox_3d)
    #     bbox_center = get_bbox_center(bbox_new)
    #     bbox_size = get_bbox_size(bbox_new)
    #     sp = [int(bbox_center[i] - bbox_size[i] / 4) for i in range(len(bbox_size))]
    #     ep = [int(bbox_center[i] + bbox_size[i] / 4) + 1 for i in range(len(bbox_size))]
    #     orig_long_patch = long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
    #     orig_short_patch = long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]]
    #     long_axis = axes_list[i][0]
    #     short_axis = axes_list[i][1]
    #     orig_long_patch[orig_long_patch > 0 and long_axis < orig_long_patch] = long_axis
    #     orig_short_patch[orig_short_patch > 0 and]
    #     # long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = min(axes_list[i][0],
    #     #                                                            long_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]])
    #     # short_axes_map[sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = min(axes_list[i][1],
    #     #                                                             short_axes_map[sp[0]:ep[0], sp[1]:ep[1],
    #     #                                                             sp[2]:ep[2]])
    return long_axes_map, short_axes_map


def add_padding(vol, padding_size, padding_value=0):
    size_orig = vol.shape
    size_pad = [size_orig[i] + padding_size[i] * 2 for i in range(len(size_orig))]
    vol_o = np.ones(size_pad, dtype=vol.dtype) * padding_value
    vol_o[padding_size[0]:padding_size[0] + size_orig[0], padding_size[1]:padding_size[1] + size_orig[1],
          padding_size[2]:padding_size[2] + size_orig[2]] = vol
    return vol_o


def get_nodule_gauss_map(vol_size, thickness, bbox_3d_list, default_sigma=0.2, spacing_xy=0.7):

    def add_bbox_z(bbox_3d):
        bbox_size = get_bbox_size(bbox_3d)
        size_z = max(bbox_size[1] * spacing_xy / thickness, bbox_size[0])
        z_c = bbox_3d[2] + bbox_size[0] / 2.0
        z_s = int(round(z_c - size_z / 2.0))
        z_e = int(round(z_c + size_z / 2.0))
        return [bbox_3d[0], bbox_3d[1], z_s, bbox_3d[3], bbox_3d[4], z_e]

    gauss_map = np.zeros(vol_size, dtype='uint16')
    for bbox_3d in bbox_3d_list:
        bbox_new = add_bbox_z(bbox_3d)
        bbox_size_new = get_bbox_size(bbox_new)
        gauss_patch = draw_gauss_map_3d(
            bbox_size_new, default_sigma * bbox_size_new[1], default_sigma * bbox_size_new[0]
        )
        gauss_patch = (gauss_patch * 255).astype('uint16')
        bbox_center = get_bbox_center(bbox_new)
        start_point = [bbox_center[i] - bbox_size_new[i] / 2.0 for i in range(len(bbox_size_new))]
        start_point = [int(round(s)) for s in start_point]

        # # debug
        # test_one = np.ones((128, 128, 128), dtype="uint16")
        # test_one *= 200
        # vol_add(test_one, gauss_patch, (-10, -10, -10))
        # save_nii(test_one, "temp/test_one")

        vol_add(gauss_map, gauss_patch, start_point)

    gauss_map = np.clip(gauss_map, 0, 255)
    gauss_map = gauss_map.astype('uint8')

    return gauss_map


def get_nodule_gauss_map_2d(vol_size, bbox_2d_list, default_sigma=0.2, sigma_refine=True):
    # bbox2d: cls_name, xmin, ymin, xmax, ymax, z

    gauss_map = np.zeros(vol_size, dtype='uint8')

    for bbox_2d in bbox_2d_list:
        size_2d = [int(bbox_2d[4] - bbox_2d[2]), int(bbox_2d[3] - bbox_2d[1])]
        center_xy = [(size_2d[0] - 1) / 2.0, (size_2d[1] - 1) / 2.0]
        sigma = default_sigma * size_2d[0]
        if sigma_refine:
            # sigma = max(0.5 * (size_2d[0] - 20), 3)
            # debug
            # sigma = max(default_sigma * (size_2d[0] - 10), 3)
            sigma = max(default_sigma * (size_2d[0] - 15), 2)
        gauss_patch = draw_gauss_map_2d(size_2d, center_xy, sigma)
        gauss_patch = (gauss_patch * 255).astype('uint8')
        gauss_patch = np.expand_dims(gauss_patch, axis=0)

        start_point = [int(bbox_2d[5]), int(bbox_2d[2]), int(bbox_2d[1])]

        vol_max(gauss_map, gauss_patch, start_point)

        # # debug
        # save_nii(gauss_map, "temp/gauss_map")
        # print("debug")

    return gauss_map


# def get_nodule_circle_2d(vol_size, bbox_2d_list, default_sigma=0.2, sigma_refine=True):
#     # bbox2d: cls_name, xmin, ymin, xmax, ymax, z
#
#     circle_map = np.zeros(vol_size, dtype="uint8")
#
#     for bbox_2d in bbox_2d_list:
#         size_2d = [int(bbox_2d[4] - bbox_2d[2]), int(bbox_2d[3] - bbox_2d[1])]
#         center_xy = [(size_2d[0] - 1) / 2.0, (size_2d[1] - 1) / 2.0]
#         sigma = default_sigma * size_2d[0]
#         if sigma_refine:
#             # sigma = max(0.5 * (size_2d[0] - 20), 3)
#             # debug
#             # sigma = max(default_sigma * (size_2d[0] - 10), 3)
#             sigma = max(default_sigma * (size_2d[0] - 15), 2)
#         circle = draw_gauss_map_2d(size_2d, center_xy, sigma)
#         circle = (circle * 255).astype("uint8")
#         circle[circle > 128] = 255
#         circle[circle < 128] = 0
#         circle = np.expand_dims(circle, axis=0)
#         start_point = [int(bbox_2d[5]), int(bbox_2d[2]), int(bbox_2d[1])]
#
#         vol_max(circle_map, circle, start_point)
#
#     return circle_map


def get_bbox_circle_2d(vol_size, bbox_2d_list, bbox_2d_idx_list, default_sigma=0.2, sigma_refine=True):
    # bbox2d: cls_name, xmin, ymin, xmax, ymax, z

    # bbox_2d_list_sorted = sorted(bbox_2d_list, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True)
    circle_map = np.zeros(vol_size, dtype='int16')

    for idx, bbox_2d in enumerate(bbox_2d_list):
        size_2d = [int(bbox_2d[4] - bbox_2d[2]), int(bbox_2d[3] - bbox_2d[1])]
        center_xy = [(size_2d[0] - 1) / 2.0, (size_2d[1] - 1) / 2.0]
        sigma = default_sigma * size_2d[0]
        if sigma_refine:
            sigma = max(default_sigma * (size_2d[0] - 15), 2)
        circle = draw_gauss_map_2d(size_2d, center_xy, sigma)
        circle[circle >= 0.3] = 1.0
        circle[circle <= 0.3] = 0.0
        circle *= (bbox_2d_idx_list[idx] + 1)
        circle = circle.astype('uint16')

        # circle = (circle * 255).astype("uint8")
        # circle[circle > 128] = 255
        # circle[circle < 128] = 0
        circle = np.expand_dims(circle, axis=0)
        start_point = [int(bbox_2d[5]), int(bbox_2d[2]), int(bbox_2d[1])]

        vol_max(circle_map, circle, start_point)

    # intensity: background=-1, bbox_2d:0~len(bbox_2d_list)-1
    circle_map = circle_map - 1

    return circle_map


def check_out_of_range(start_point, end_point, vol_size):
    # z
    if start_point[0] < 0:
        end_point[0] += (0 - start_point[0])
        start_point[0] = 0
    if end_point[0] > vol_size[0]:
        start_point[0] -= (end_point[0] - vol_size[0])
        end_point[0] = vol_size[0]
    # y
    if start_point[1] < 0:
        end_point[1] += (0 - start_point[1])
        start_point[1] = 0
    if end_point[1] > vol_size[1]:
        start_point[1] -= (end_point[1] - vol_size[1])
        end_point[1] = vol_size[1]
    # x
    if start_point[2] < 0:
        end_point[2] += (0 - start_point[2])
        start_point[2] = 0
    if end_point[2] > vol_size[2]:
        start_point[2] -= (end_point[2] - vol_size[2])
        end_point[2] = vol_size[2]

    return start_point, end_point


def gauss_smooth_3d(vol, sigma, truncate=4.0):
    filtered = gaussian_filter(vol, sigma, truncate=truncate)
    return filtered


def remove_no_mask_bbox_2d(bbox_2d_list, mask):
    # bbox2d: cls_name, xmin, ymin, xmax, ymax, z
    for bbox_2d in bbox_2d_list:
        patch = mask[int(bbox_2d[5]), int(bbox_2d[2]):int(bbox_2d[4]), int(bbox_2d[1]):int(bbox_2d[3])]
        # shape = patch.shape
        # center = [shape[0] / 2, shape[1] / 2]
        # center_area = patch[int(center[0] - shape[0] / 4):int(center[0] + shape[0] / 4),
        #               int(center[1] - shape[1] / 4):int(center[1] + shape[1] / 4)]
        if 1 not in patch:
            bbox_2d_list.remove(bbox_2d)
    return bbox_2d_list


def generate_bbox_from_mask(mask, bbox_scale, is_rand=True):
    # bbox_2d: cls_name, xmin, ymin, xmax, ymax, z
    # bbox_3d: xmin, ymin, zmin, xmax, ymax, zmax
    def get_seed(mask):
        if 1 not in mask:
            return []
        else:
            seed = np.argwhere(mask == 1)[0]
            return seed

    def generate_bbox_from_single_mask(single_mask, bbox_scale, mask_sp, mask_ep):
        mask_size = [mask_ep[i] - mask_sp[i] + 1 for i in range(3)]
        center_3d = [mask_sp[i] + mask_size[i] / 2 for i in range(3)]
        max_mask_size = max(mask_ep[1] - mask_sp[1] + 1, mask_ep[2] - mask_sp[2] + 1)
        half_bbox_3d_size = (bbox_scale * max_mask_size + 10) * 0.5
        bbox_3d = [
            int(center_3d[2] - half_bbox_3d_size),
            int(center_3d[1] - half_bbox_3d_size),
            int(mask_sp[0]),
            int(center_3d[2] + half_bbox_3d_size + 1),
            int(center_3d[1] + half_bbox_3d_size + 1),
            int(mask_ep[0])
        ]
        bbox_2d_list = []
        for z in range(mask_sp[0], mask_ep[0] + 1):
            mask_img = single_mask[z]
            contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            half_bbox_2d_size = (bbox_scale * max(w, h) + 10) * 0.5
            bbox_2d = [
                '',
                int(cx - half_bbox_2d_size),
                int(cy - half_bbox_2d_size),
                int(cx + half_bbox_2d_size + 1),
                int(cy + half_bbox_2d_size + 1), z
            ]
            bbox_2d_list.append(bbox_2d)
        return bbox_2d_list, bbox_3d

    def bbox_2d_rand_shift(bbox_2d_list, bbox_3d, rand_scale=0.2, rand_shift=0.1):
        for bbox_2d in bbox_2d_list:
            size_2d = [bbox_2d[4] - bbox_2d[2], bbox_2d[3] - bbox_2d[1]]
            center_2d = [bbox_2d[2] + size_2d[0] / 2, bbox_2d[1] + size_2d[1] / 2]
            size_3d = get_bbox_size(bbox_3d)
            scale_max = min(2.0, size_3d[1] / size_2d[0] + rand_scale)
            scale_min = scale_max - 0.5
            scale = random.uniform(scale_min, scale_max)
            size_2d_new = size_2d[0] * scale
            shift_x = random.uniform(1 - size_2d_new * rand_shift, 1 + size_2d_new * rand_shift)
            shift_y = random.uniform(1 - size_2d_new * rand_shift, 1 + size_2d_new * rand_shift)
            bbox_2d[1] = int(center_2d[1] - size_2d_new / 2 + shift_x)  # x_min
            bbox_2d[2] = int(center_2d[0] - size_2d_new / 2 + shift_y)  # y_min
            bbox_2d[3] = int(center_2d[1] + size_2d_new / 2 + shift_x) + 1  # x_max
            bbox_2d[4] = int(center_2d[0] + size_2d_new / 2 + shift_y) + 1  # y_max

    mask_cp = mask.copy()

    bbox_2d_list = []
    bbox_3d_list = []
    while (True):
        seed = get_seed(mask_cp)
        if seed == []:
            break
        single_mask = np.zeros(mask_cp.shape, dtype='uint8')
        mask_sp, mask_ep = flood_fill_3d(mask_cp, single_mask, seed)
        mask_cp = mask_cp - single_mask

        if np.sum(single_mask) < 3:
            continue

        bbox_2d_list_single, bbox_3d = generate_bbox_from_single_mask(single_mask, bbox_scale, mask_sp, mask_ep)
        if is_rand:
            bbox_2d_rand_shift(bbox_2d_list_single, bbox_3d, rand_scale=0.2, rand_shift=0.1)

        bbox_2d_list += bbox_2d_list_single
        bbox_3d_list.append(bbox_3d)
    return bbox_2d_list, bbox_3d_list


def gauss_smooth_mask(mask, smooth=0.1):

    def get_seed(mask):
        if 1 not in mask:
            return []
        else:
            seed = np.argwhere(mask == 1)[0]
            return seed

    def get_sigma(single_mask):
        v = np.sum(single_mask)
        z = np.sum(single_mask, axis=(1, 2))
        z[z > 0] = 1
        z = np.sum(z)
        sq = float(v) / float(z)
        sigma = math.pow(sq, 0.5) * smooth
        return sigma

    mask_bin = np.zeros(mask.shape, dtype='uint8')
    mask_bin[mask > 0] = 1
    mask_out = np.zeros(mask.shape, dtype='float32')
    while (True):
        # # debug
        # save_nii(mask * 255, "temp/mask_o")

        seed = get_seed(mask_bin)
        if seed == []:
            break
        single_mask = np.zeros(mask_bin.shape, dtype='uint8')
        flood_fill_3d(mask_bin, single_mask, seed)
        mask_bin = mask_bin - single_mask

        # v = np.sum(single_mask)
        # sigma = max(math.pow(float(v), 1.0 / 3.0) / 20.0, 1.0)
        sigma = get_sigma(single_mask)
        single_mask_cp = mask.astype('float32').copy()
        single_mask_cp[single_mask == 0] = 0
        single_mask_cp = gauss_smooth_3d(single_mask_cp, (0, sigma, sigma))  # 2d gauss
        single_mask_cp = gauss_smooth_3d(single_mask_cp, (0.5, 0, 0))  # 2d gauss
        single_max = np.max(single_mask_cp)
        single_mask_cp *= 255.0 / single_max
        vol_max(mask_out, single_mask_cp, (0, 0, 0))
    return mask_out


def gauss_smooth_mask_2d(mask, smooth=0.1):

    def get_seed(mask):
        if 1 not in mask:
            return []
        else:
            seed = np.argwhere(mask == 1)[0]
            return seed

    def get_sigma(single_mask):
        v = np.sum(single_mask)
        sigma = math.pow(float(v), 0.5) * smooth
        return sigma

    mask_bin = np.zeros(mask.shape, dtype='uint8')
    mask_bin[mask > 0] = 1
    mask_out = np.zeros(mask.shape, dtype='float32')
    for i in range(mask.shape[0]):
        mask_slice = mask_bin[i:i + 1]
        while (True):
            seed = get_seed(mask_slice)
            if seed == []:
                break
            single_mask = np.zeros(mask_slice.shape, dtype='uint8')
            flood_fill_3d(mask_slice, single_mask, seed)
            mask_slice = mask_slice - single_mask

            sigma = get_sigma(single_mask)
            single_mask_cp = mask[i:i + 1].astype('float32').copy()
            single_mask_cp[single_mask == 0] = 0
            single_mask_cp = gauss_smooth_3d(single_mask_cp, (0, sigma, sigma))  # 2d gauss
            single_max = np.max(single_mask_cp)
            single_mask_cp *= 255.0 / single_max
            vol_max(mask_out[i:i + 1], single_mask_cp, (0, 0, 0))
    mask_out = np.clip(mask_out, 0, 255)
    return mask_out.astype('uint8')


def vol_min(vol_tgt, vol_src, start_point):
    size_tgt = vol_tgt.shape
    size_src = vol_src.shape
    tgt_s = list(start_point)
    tgt_e = [tgt_s[i] + size_src[i] for i in range(len(tgt_s))]
    src_s = [0, 0, 0]
    src_e = list(size_src)
    for i in range(len(tgt_s)):
        if tgt_s[i] < 0:
            src_s[i] += (0 - tgt_s[i])
            tgt_s[i] = 0
        if tgt_e[i] > size_tgt[i]:
            src_e[i] -= (tgt_e[i] - size_tgt[i])
            tgt_e[i] = size_tgt[i]
        if src_e[i] <= src_s[i] or tgt_e[i] <= tgt_s[i]:
            return

    patch_tgt = vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]]
    patch_src = vol_src[src_s[0]:src_e[0], src_s[1]:src_e[1], src_s[2]:src_e[2]]
    patch_min = (np.array((patch_tgt, patch_src))).min(axis=0)
    vol_tgt[tgt_s[0]:tgt_e[0], tgt_s[1]:tgt_e[1], tgt_s[2]:tgt_e[2]] = patch_min


def adjust_idx(sp, ep, size_3d):
    for i in range(len(sp)):
        if sp[i] < 0:
            ep[i] += (0 - sp[i])
            sp[i] = 0
        elif ep[i] > size_3d[i]:
            sp[i] -= (ep[i] - size_3d[i])
            ep[i] = size_3d[i]
    return sp, ep


# def interpolate3D_gpu(tensorInput, tensorFlow, mode="bilinear", padding_mode='zeros'):
#     tensorDepth = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1, 1).expand(
#         tensorFlow.size(0), -1, -1, tensorFlow.size(3), tensorFlow.size(4))
#     tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3), 1).expand(
#         tensorFlow.size(0), -1, tensorFlow.size(2), -1, tensorFlow.size(4))
#     tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(4)).view(1, 1, 1, 1, tensorFlow.size(4)).expand(
#         tensorFlow.size(0), -1, tensorFlow.size(2), tensorFlow.size(3), -1)
#     warped = torch.cat([tensorHorizontal, tensorVertical, tensorDepth], 1).cuda()
#
#     tensorFlow = torch.cat([tensorFlow[:, 2:3, :, :, :] / ((tensorInput.size(4) - 1.0) / 2.0),
#                             tensorFlow[:, 1:2, :, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
#                             tensorFlow[:, 0:1, :, :, :] / ((tensorInput.size(2) - 1.0) / 2.0), ], 1)
#
#     tensor_t = torch.nn.functional.grid_sample(input=tensorInput, grid=(warped + tensorFlow).permute(0, 2, 3, 4, 1),
#                                                mode=mode, padding_mode=padding_mode)
#     return tensor_t

# def find_bbox_3d_from_mask(mask, bbox_2d_map):
#     cc = get_connected_components(mask)
#     cc_num = np.max(cc)
#     for i in range()


def get_connected_components(mask):
    mask_img = sitk.GetImageFromArray(mask)
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOn()
    res = connectedFilter.Execute(mask_img)
    cc = sitk.GetArrayFromImage(res)
    return cc


def get_mask_centroid(mask, idx):
    mask_img = sitk.GetImageFromArray(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_img)
    centroid = stats.GetCentroid(idx)
    centroid = list(centroid)
    centroid = centroid[::-1]
    return centroid


def get_bbox_3d_from_mask(mask):
    # bbox_3d: zmin, ymin, xmin, zmax, ymax, xmax
    mask_img = sitk.GetImageFromArray(mask)
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOn()
    cc_img = connectedFilter.Execute(mask_img)
    cc = sitk.GetArrayFromImage(cc_img)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc_img)

    bbox_3d_list = []
    max_idx = np.max(cc)
    for i in range(1, max_idx + 1):
        bbox_3d = stats.GetBoundingBox(i)
        # if bbox_3d[3] == 1 and bbox_3d[4] == 1 and bbox_3d[5] == 1:
        #     continue
        bbox_3d_list.append(
            [
                bbox_3d[2], bbox_3d[1], bbox_3d[0], bbox_3d[2] + bbox_3d[5], bbox_3d[1] + bbox_3d[4],
                bbox_3d[0] + bbox_3d[3]
            ]
        )
    return bbox_3d_list, cc


def get_single_bbox_3d_from_mask(mask):
    mask_bin = (mask > 0).astype('uint8')
    mask_img = sitk.GetImageFromArray(mask_bin)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_img)
    bbox_3d = stats.GetBoundingBox(1)
    bbox_3d_o = [
        bbox_3d[2], bbox_3d[1], bbox_3d[0], bbox_3d[2] + bbox_3d[5], bbox_3d[1] + bbox_3d[4], bbox_3d[0] + bbox_3d[3]
    ]
    return bbox_3d_o


def binary_thinning(mask):
    # mask_img = sitk.GetImageFromArray(mask)
    # thinning_filter = sitk.BinaryThinningImageFilter()
    # thin_img = thinning_filter.Execute(mask_img)
    # thin_np = sitk.GetArrayFromImage(thin_img)

    mask_img = sitk.GetImageFromArray(mask)
    thin_img = sitk.BinaryThinning(mask_img)
    # thin_img = thinning_filter.Execute(mask_img)
    thin_np = sitk.GetArrayFromImage(thin_img)

    return thin_np


def get_bbox_3d_size_from_mask(mask):
    mask_img = sitk.GetImageFromArray(mask)
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOn()
    cc_img = connectedFilter.Execute(mask_img)
    cc = sitk.GetArrayFromImage(cc_img)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc_img)

    bbox_3d_size = []
    bbox_3d_size.append(0)  # background

    max_idx = np.max(cc)
    for i in range(1, max_idx + 1):
        bbox_3d = stats.GetBoundingBox(i)
        bbox_3d_size.append(bbox_3d[3] * bbox_3d[4] * bbox_3d[5])
    return bbox_3d_size, cc


def binary_fill_hole(mask, foreground_value=1):
    mask_img = sitk.GetImageFromArray(mask)
    mask_img = sitk.BinaryFillhole(mask_img, fullyConnected=True, foregroundValue=foreground_value)
    return sitk.GetArrayFromImage(mask_img)


def get_body_mask(vol, threshold=-200):
    mask = np.zeros(vol.shape, dtype='uint8')
    mask[vol > threshold] = 255
    bbox_3d_size, cc = get_bbox_3d_size_from_mask(mask)
    idx = list(range(len(bbox_3d_size)))
    sorted_idx = sorted(idx, key=lambda x: bbox_3d_size[x], reverse=True)
    mask[cc != sorted_idx[0]] = 0

    padding = 1
    mask = np.pad(mask, padding, 'constant', constant_values=0)
    mask = fill_hole(mask)
    mask = mask[padding:-padding, padding:-padding, padding:-padding]
    return mask


def get_max_bbox_connection(mask):
    bbox_3d_size, cc = get_bbox_3d_size_from_mask(mask)
    idx = list(range(len(bbox_3d_size)))
    sorted_idx = sorted(idx, key=lambda x: bbox_3d_size[x], reverse=True)
    mask[cc != sorted_idx[0]] = 0
    return mask


def get_max_bbox_from_mask(mask):
    mask_img = sitk.GetImageFromArray(mask)
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOn()
    cc_img = connectedFilter.Execute(mask_img)
    cc = sitk.GetArrayFromImage(cc_img)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc_img)

    bbox_3d_size = []
    max_idx = np.max(cc)
    bbox_3d_list = []
    for i in range(1, max_idx + 1):
        bbox_3d = stats.GetBoundingBox(i)
        bbox_3d_list.append(
            [
                bbox_3d[2], bbox_3d[1], bbox_3d[0], bbox_3d[2] + bbox_3d[5], bbox_3d[1] + bbox_3d[4],
                bbox_3d[0] + bbox_3d[3]
            ]
        )
        bbox_3d_size.append(bbox_3d[3] * bbox_3d[4] * bbox_3d[5])
    idx = list(range(len(bbox_3d_size)))
    sorted_idx = sorted(idx, key=lambda x: bbox_3d_size[x], reverse=True)
    return bbox_3d_list[sorted_idx[0]]


def normalization(vol_np, win_level, win_width):
    win = [win_level - win_width / 2, win_level + win_width / 2]
    vol = vol_np.astype('float32')
    vol = np.clip(vol, win[0], win[1])
    vol -= win[0]
    vol /= win_width
    return vol


def resize_gpu(vol, size_tgt, mode='bilinear'):
    aff_matrix = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    aff_matrix = aff_matrix.repeat(vol.size(0), 1, 1)
    size_tgt_help = torch.Size([int(vol.size(0)), 1] + list(size_tgt))
    grid = torch.nn.functional.affine_grid(aff_matrix, size_tgt_help).cuda()
    vol_t = torch.nn.functional.grid_sample(vol, grid, mode)
    return vol_t


def optimized_resize(vol, size_tgt, mode='trilinear', stride=16):
    # input: numpy or torch tensor
    # output: numpy

    size_orig = vol.shape
    if len(size_orig) == 3:
        vol = vol[np.newaxis, np.newaxis, :, :, :]
    elif len(size_orig) == 4:
        vol = vol[np.newaxis, :, :, :, :]
    elif len(size_orig) == 5:
        vol = vol
    else:
        return None
    size_in = vol.shape

    size_temp = [size_in[0], size_in[1], size_in[2], size_tgt[1], size_tgt[2]]
    vol_temp = np.zeros(size_temp, dtype='float32')
    size_out = [size_in[0], size_in[1], size_tgt[0], size_tgt[1], size_tgt[2]]
    vol_out = np.zeros(size_out, dtype=vol.dtype)

    if mode == 'nearest':
        align_corners = None
    else:
        align_corners = True

    for z_s in range(0, size_in[2], stride):
        z_e = min(z_s + stride, size_in[2])
        slice = torch.from_numpy(vol[:, :, z_s:z_e]).cuda().float()
        slice_temp = nn.functional.interpolate(
            slice, size=[(z_e - z_s), size_tgt[1], size_tgt[2]], mode=mode, align_corners=align_corners
        )
        vol_temp[:, :, z_s:z_e] = slice_temp.cpu().numpy()
    for x_s in range(0, size_tgt[2], stride):
        x_e = min(x_s + stride, size_tgt[2])
        slice = torch.from_numpy(vol_temp[:, :, :, :, x_s:x_e]).cuda().float()
        slice_temp = nn.functional.interpolate(
            slice, size=[size_tgt[0], size_tgt[1], (x_e - x_s)], mode=mode, align_corners=align_corners
        )
        vol_out[:, :, :, :, x_s:x_e] = slice_temp.cpu().numpy().astype(vol.dtype)

    if len(size_orig) == 3:
        return vol_out[0, 0]
    elif len(size_orig) == 4:
        return vol_out[0]
    else:
        return vol_out


if __name__ == '__main__':
    img_size = (128, 128)
    center = (64, 64)
    gauss = draw_gauss_map_2d(img_size, center, 20)
    # save_nii(gauss, "../../temp/gauss")

    # num = 100
    # tmp_img = sitk.ReadImage(
    #     "/home/tx-deepocean/data/Nodule_Segmentation/dataset/nodule_seg_bbox_no_aug/mask/%05d.nii.gz" % num)
    # vol = sitk.GetArrayFromImage(tmp_img)
    # vol = vol.astype("float32") * 255
    # vol_smooth = gauss_smooth_3d(vol, 5)
    # tmp_img_smooth = sitk.GetImageFromArray(vol_smooth)
    # sitk.WriteImage(tmp_img * 255, "../../temp/orig.nii.gz")
    # sitk.WriteImage(tmp_img_smooth, "../../temp/smooth.nii.gz")
