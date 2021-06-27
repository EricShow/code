import glob
import math
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
import torch
from skimage.morphology import ball, binary_dilation, skeletonize, thin
from starship.segments import region_grow
from VesselProcessUtils.distancetransform.DistanceTransform import DistanceTransform

from .line import Line

main_part_limit = 4
joint_point = 20
vertex_pont = 30
seg_point = 10


def relabelConnectedComponent(im):
    return sitk.RelabelComponent(sitk.ConnectedComponent(im > 0))#最大连通域


def get_cut_box(mask_img, box_mode):
    """input lung_mask(nii/mhd/nrrd), return 3D lung mask bbox index and croped
    mask."""
    if mask_img is None:
        return None
    stat = sitk.LabelShapeStatisticsImageFilter()
    stat.Execute(mask_img)
    bbox = stat.GetBoundingBox(1)
    img_bbox = [bbox[0], bbox[1], bbox[2], bbox[0] + bbox[3], bbox[1] + bbox[4], bbox[2] + bbox[5]]
    array_box = [bbox[2], bbox[1], bbox[0], bbox[2] + bbox[5], bbox[1] + bbox[4], bbox[0] + bbox[3]]
    if box_mode == 'img':
        return img_bbox
    elif box_mode == 'array':
        return array_box
    elif box_mode == 'both':
        return img_bbox, array_box
    else:
        print('nox mode error')
        return None


def show_center_line(center_line):
    pos = np.where(center_line != 0)
    result_center_line = np.zeros_like(center_line)
    x, y, z = pos
    for i in range(x.shape[0]):
        result_center_line[x[i] - 1:x[i] + 2, y[i] - 1:y[i] + 2, z[i] - 1:z[i] + 2] = center_line[x[i], y[i], z[i]]
    return result_center_line


def torch_argwhere(array, gpu, mode='argwhere'):
    with torch.no_grad(), torch.cuda.device(gpu):
        array_torch = torch.from_numpy(array).cuda(gpu)
        coor_tuple = torch.where(array_torch)
        coor_array_tuple = [coor.cpu().numpy() for coor in coor_tuple]
        coor_array_cat = np.stack(coor_array_tuple, axis=0)
        if mode == 'argwhere':
            final_coor_array = coor_array_cat.transpose()
            return final_coor_array
        elif mode == 'where':
            return coor_array_tuple
        else:
            raise NotImplementedError('not support this mode %s' % mode)


def get_different_points(center_line):   #blank_mask和分支点point  
    pos = np.where(center_line != 0)  #获取不为0的位置信息
    x, y, z = pos                              
    blank_mask = np.zeros_like(center_line) 
    join_point_mask = np.zeros_like(center_line)
    for i in range(x.shape[0]):
        tmp_cube = center_line[x[i] - 1:x[i] + 2, y[i] - 1:y[i] + 2, z[i] - 1:z[i] + 2]
        if np.sum(tmp_cube) == 2:
            blank_mask[x[i], y[i], z[i]] = vertex_pont  #顶点
        elif np.sum(tmp_cube) == 3:
            blank_mask[x[i], y[i], z[i]] = seg_point  #
        else:
            blank_mask[x[i], y[i], z[i]] = joint_point #分支点
            join_point_mask[x[i] - 1:x[i] + 2, y[i] - 1:y[i] + 2, z[i] - 1:z[i] + 2] = 1
    return blank_mask, join_point_mask


# @func_line_time
def get_main_part(seg_array, center_line, distance_array, gpu):  #input_array, center_line, distance_array, gpu
    main_part = (distance_array > main_part_limit)
    # main_part = binary_dilation(main_part, ball(radius=main_part_limit)).astype("uint8")
    main_part = runGPURegiongrowth((seg_array > 0).astype('uint8'), main_part, [1, 2], 1, main_part_limit + 1, gpu)
    main_part_sitk = sitk.GetImageFromArray(main_part)
    main_part_sitk = relabelConnectedComponent(main_part_sitk) == 1   #最大连通域
    main_part_sitk = sitk.Mask(main_part_sitk, relabelConnectedComponent(main_part_sitk) == 1)#最大连通域处理后，再次进行mask
    main_part = sitk.GetArrayFromImage(main_part_sitk)
    center_line_filtered = center_line.copy() # centerline_filtered
    center_line_filtered[main_part != 0] = 0 # mask不为0的部分 为centerlinefiltered
    center_line_inner = center_line.copy()
    center_line_inner[main_part == 0] = 0 # mask为0 为centerline inner
    center_line_mask, join_point_mask = get_different_points(center_line_filtered)
    return main_part, center_line_mask, join_point_mask, center_line_inner #centerline_inner是什么


def runGPURegiongrowth(track_array, seed_array, threshold, distance_limit, growth_limit, gpu):
    track_array = track_array.transpose((2, 1, 0))
    seed_array = seed_array.transpose((2, 1, 0))
    track_cuda = torch.from_numpy(track_array.astype('uint8')).cuda(gpu)
    seed_cuda = torch.from_numpy(seed_array.astype('uint8')).cuda(gpu)
    with torch.no_grad(), torch.cuda.device(gpu):
        region_grow(
            track_cuda, seed_cuda, threshold=threshold, distance_iteration_limit=1, growth_iteration_limit=growth_limit
        )
        result_array = seed_cuda.cpu().numpy()
        result_array = result_array.transpose((2, 1, 0))
        del track_cuda
        del seed_cuda
        torch.cuda.empty_cache()
    return result_array


def get_joint_point(center_line, main_part, center_line_main_part, mode='start'):
    if mode == 'start':
        check_value = seg_point
    elif mode == 'inner':
        check_value = joint_point
    else:
        raise RuntimeError('not in modes')
    pos = np.where(center_line >= check_value)
    x, y, z = pos
    process_list = []
    for i in range(x.shape[0]):
        tmp_cube = main_part[x[i] - 1:x[i] + 2, y[i] - 1:y[i] + 2, z[i] - 1:z[i] + 2]
        if np.sum(tmp_cube) > 0:
            if (center_line[x[i], y[i], z[i]] == vertex_pont):
                process_list.append([x[i], y[i], z[i]])
            else:
                if np.sum(center_line_main_part[x[i] - 1:x[i] + 2, y[i] - 1:y[i] + 2, z[i] - 1:z[i] + 2]) > 0:
                    process_list.append([x[i], y[i], z[i]])
    return process_list


def get_next_point(center_line, current_point):
    x, y, z = current_point
    tmp_cube = center_line[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
    ori_tmp_cube = tmp_cube.copy()
    tmp_cube[1, 1, 1] = 0
    # if center_line[x,y,z]!=seg_point and center_line[x,y,z]!=vertex_pont:
    #     return None
    if center_line[x, y, z] == seg_point:
        center_line[x, y, z] = 0
    if (np.sum(tmp_cube == seg_point) == 1):
        offset = np.argwhere(tmp_cube == seg_point)[0]
        offset = [offset[i] - 1 for i in range(3)]  # 坐标转换 转换为中心点坐标
        next_point = [offset[i] + current_point[i] for i in range(3)]
        return next_point, 'continue'
    elif (np.sum(tmp_cube == joint_point) == 1):
        offset = np.argwhere(tmp_cube == joint_point)[0]
        offset = [offset[i] - 1 for i in range(3)]  # 坐标转换 转换为中心点坐标
        next_point = [offset[i] + current_point[i] for i in range(3)]
        return next_point, 'joint_point'
    elif (np.sum(tmp_cube == vertex_pont) == 1):
        offset = np.argwhere(tmp_cube == vertex_pont)[0]
        offset = [offset[i] - 1 for i in range(3)]  # 坐标转换 转换为中心点坐标
        next_point = [offset[i] + current_point[i] for i in range(3)]
        return next_point, 'vertex_pont'
    return None


def exact_line(main_part_colored, pos, former_joint_point, center_line, line_list, distance_array, mode='root'):
    tmp_cube = main_part_colored[pos[0] - 1:pos[0] + 2, pos[1] - 1:pos[1] + 2, pos[2] - 1:pos[2] + 2]
    bin_array = np.bincount(tmp_cube.reshape(27, ))
    if bin_array.shape[0] == 1:
        color = None
    if bin_array.shape[0] > 1:
        color = np.argmax(bin_array[1:]) + 1
    point = [pos[0], pos[1], pos[2]]
    point_list = [point]
    while True:
        result = get_next_point(center_line, point)
        if result is None:
            return
        result, tag = result
        if tag == 'vertex_pont':
            point_list.append(result)
            next_joint_point = None
            break
        elif tag == 'joint_point':
            point_list.append(result)
            next_joint_point = result
            break
        point = result
        point_list.append(point)

    if mode == 'leaf' or color is None:
        line = Line(
            point_list,
            former_joint_point=former_joint_point,
            next_joint_point=next_joint_point,
            is_root=False,
            distance_array=distance_array
        )
    elif mode == 'root':
        line = Line(
            point_list,
            former_joint_point=former_joint_point,
            next_joint_point=next_joint_point,
            is_root=True,
            distance_array=distance_array
        )
        line.set_color(color)  
    else:
        raise RuntimeError('not surpport this mode_name %s' % mode)
    line_list.append(line)


def get_start_point_by_joint_point(center_line_mask, point):

    def get_next_point_inner(center_line, current_point, point_list):
        x, y, z = current_point
        tmp_cube = center_line[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
        ori_tmp_cube = tmp_cube.copy()
        tmp_cube[1, 1, 1] = 0
        start_points = np.argwhere(tmp_cube == seg_point)
        for s_p in start_points:
            offset = [s_p[i] - 1 for i in range(3)]
            point_list.append([current_point[i] + offset[i] for i in range(3)])

        if (np.sum(tmp_cube == joint_point) >= 1):
            offsets = np.argwhere(tmp_cube == joint_point)
            next_point_list = []
            for offset in offsets:
                offset = [offset[i] - 1 for i in range(3)]  # 坐标转换 转换为中心点坐标
                next_point = [offset[i] + current_point[i] for i in range(3)]
                next_point_list.append(next_point)
            return next_point_list
        return None

    point_find_list = [point]
    point_list = []
    while True:
        for p in point_find_list:
            result = get_next_point_inner(center_line_mask, p, point_list)
        if result is None:
            break
        point_find_list = result
    return point_list


def in_out_line_exact(center_line, main_part, nrrd_array, center_line_main_part, distance_array):
    center_line = center_line.copy()
    process_list = get_joint_point(center_line, main_part, center_line_main_part)
    line_list = []
    for point in process_list:
        if center_line[point[0], point[1], point[2]] == vertex_pont:
            exact_line(main_part, point, point, center_line, line_list, distance_array, mode='root')
        else:
            start_points_from_conjunction = get_start_point_by_joint_point(center_line, point)
            for s_p in start_points_from_conjunction:
                exact_line(main_part, s_p, point, center_line, line_list, distance_array, mode='root')
    root_line_list = [line_list]
    leaf_line_list = []
    while True:
        new_line_list = []
        for line in line_list:
            next_point = line.get_next_joint_point()
            if next_point is not None:
                start_points_from_conjunction = get_start_point_by_joint_point(center_line, next_point)
                for s_p in start_points_from_conjunction:
                    exact_line(main_part, s_p, next_point, center_line, new_line_list, distance_array, mode='leaf')
        line_list = new_line_list
        leaf_line_list.append(line_list)
        if len(line_list) == 0:
            break
    root_line_list.extend(leaf_line_list)
    return root_line_list


def get_all_lines(input_array, gpu):
    sk_input_array = (input_array > 0).astype('uint8')
    center_line = skeletonize(sk_input_array) #形态学操作  骨架化处理
    distance_array = DistanceTransform((input_array > 0).astype('uint8'), 8, gpu)
    main_part, center_line_mask, join_point_mask, center_line_main_part = get_main_part(
        input_array, center_line, distance_array, gpu
    ) #获取三个
    main_part_colored = input_array.copy()
    main_part_colored[main_part == 0] = 0
    line_list = in_out_line_exact(
        center_line_mask, main_part_colored, input_array, center_line_main_part, distance_array
    )
    return line_list, main_part_colored, center_line
