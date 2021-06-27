import glob
import random
import time

import numpy as np
import SimpleITK as sitk
import sitktools as st
import torch
import torch.nn.functional as F
from post_process.ley_process import line_process
from skimage.morphology import ball, binary_dilation, skeletonize
from starship.segments import region_grow


def transfer_line2array(lines_list):
    pick_datas = []
    for level_id, lines in enumerate(lines_list):
        for line in lines:
            p = [line.point_list, line.radius_list, level_id, 0 if (line.next_joint_point is None) else 1]
            pick_datas.append(p)

    np.savez_compressed('./temp.npz', c=pick_datas)


def get_training_zone(mask, lines_list, zone_range=5, end_ignore_zone_range=5):
    end_ignore_zone_range = int(end_ignore_zone_range / 2) + 1

    # mask_dia = binary_dilation(mask, ball(zone_range)).astype("uint8")
    mask_end_ehance = mask.copy()
    for level_id, lines in enumerate(lines_list):
        for line in lines:
            if (line.next_joint_point is None):
                p = np.array(line.point_list[-1])
                p_begin = p - end_ignore_zone_range
                p_end = p + end_ignore_zone_range + 1

                mask_end_ehance[p_begin[0]:p_end[0], p_begin[1]:p_end[1], p_begin[2]:p_end[2]] = 2

    mask_dia = torch.from_numpy(mask_end_ehance).cuda(1).half()
    mask_dia = F.max_pool3d(mask_dia[None, None], kernel_size=zone_range, stride=1, padding=int(zone_range / 2))
    mask_dia = mask_dia[0, 0].cpu().numpy().astype(np.uint8)

    image_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk.WriteImage(image_sitk, 'mask.nii.gz')

    image_sitk = sitk.GetImageFromArray(mask_dia.astype(np.uint8))
    sitk.WriteImage(image_sitk, 'mask_dia.nii.gz')

    # image_sitk = sitk.GetImageFromArray(mask_end_ehance.astype(np.uint8))
    # sitk.WriteImage(image_sitk, "mask_end_ehance.nii.gz")


def main(path):
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    image_array = (image_array > 0) * 1   #大于0的部分全部置为1
    image_array = image_array.astype(np.uint8)
    cut_array_total = np.zeros_like(image_array)
    total_mask, level_mask, lines_list = line_process(image_array, gpu=1)

    get_training_zone(image_array, lines_list, zone_range=15, end_ignore_zone_range=1)
    # transfer_line2array(lines_list)
    #
    # for line_id, lines in enumerate(lines_list):
    #     for line in lines:
    #         rand = random.randint(0, 1)
    #         try:
    #             if rand <= 1:
    #                 tic = time.time()
    #                 image_array, cut_array = line.random_cut(image_array)
    #                 cut_array_total += cut_array
    #                 toc = time.time()
    #                 print(f'elapse {toc-tic} s')
    #         except:
    #             pass

    return image_array, cut_array_total


if __name__ == '__main__':

    glob_path = '/media/tx-deepocean/Data/heart/358/nii/*-seg.nii.gz'
    for idx, file in enumerate(glob.iglob(glob_path)):
        if idx != 1:
            continue
        image_array, cut_array_total = main(file)
        image_sitk = sitk.GetImageFromArray(image_array)
        sitk.WriteImage(image_sitk, 'image.nii.gz')
        cut_sitk = sitk.GetImageFromArray((cut_array_total * 2 + image_array).astype('uint8'))
        sitk.WriteImage(cut_sitk, 'cut.nii.gz')
        break
