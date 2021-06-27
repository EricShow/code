import os
import SimpleITK as sitk
import numpy as np
import glob
import torch
from skimage.morphology import skeletonize
from VesselProcessUtils.distancetransform.DistanceTransform import DistanceTransform


def torch_argwhere(array, gpu, mode="argwhere"):
    with torch.no_grad(), torch.cuda.device(gpu):
        array_torch = torch.from_numpy(array).cuda(gpu)
        coor_tuple = torch.where(array_torch.char())  # char 类型
        coor_array_tuple = [coor.cpu().numpy() for coor in coor_tuple]
        coor_array_cat = np.stack(coor_array_tuple, axis=0)
        if mode == "argwhere":
            final_coor_array = coor_array_cat.transpose()
            return final_coor_array
        elif mode == "where":
            return coor_array_tuple
        else:
            raise NotImplementedError("not support this mode %s" % mode)


def get_sk_radius(array, gpu):
    distance_array = DistanceTransform(array, 8, gpu)
    skeleton_array = skeletonize(array)
    pos = torch_argwhere(skeleton_array, gpu)
    result_dict = {}
    for point in pos:
        tmp_cube = distance_array[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2, point[2] - 1:point[2] + 2]
        radius = np.max(tmp_cube)
        result_dict[tuple(point)] = radius
    return result_dict, skeleton_array

def get_joint_point(coords, sk_array):
    joint_corrds = []
    for coord in coords:
        coord = np.array(coord)
        c_s = coord - 1
        c_e = coord + 2

        cube_array = sk_array[c_s[0]:c_e[0], c_s[1]:c_e[1], c_s[2]:c_e[2]]
        if np.sum(cube_array) > 3:
            joint_corrds.append(coord)
    return joint_corrds


def process_result(coord_radius, size):
    ret_array = np.zeros(shape=size, dtype=np.uint8)
    for k, v in coord_radius.items():
        k = np.array(k)
        v = int(v)
        # k_start = k - v
        # k_end = k + v + 1
        k_start = k - 1
        k_end = k + 1 + 1
        ret_array[k_start[0]:k_end[0], k_start[1]:k_end[1], k_start[2]:k_end[2]] = 1
    return ret_array
