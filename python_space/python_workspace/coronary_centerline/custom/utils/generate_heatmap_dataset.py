import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, grey_dilation
from starship.sitktools import getBoundingBox

import cc3d
from skimage.morphology import skeletonize

cor_label = 1
vein_label = 4
seed_label = 3


def remove_no_cor_cc_vein(vein_mask, cor_mask):
    mask = (vein_mask + cor_mask) > 0

    write_mask = np.zeros_like(mask)

    cc_out, num = cc3d.connected_components(mask, return_N=True)

    for seg_id in range(1, num + 1):
        cur_mask = cc_out == seg_id
        # print(np.sum(cor_mask[cur_mask]))
        if np.sum(cor_mask[cur_mask]) == 0:
            continue

        write_mask += cur_mask

    return vein_mask * (write_mask > 0)


def remove_small_cc(mask, size=20):
    write_mask = np.zeros_like(mask)

    cc_out, num = cc3d.connected_components(mask, return_N=True)

    for seg_id in range(1, num + 1):
        cur_mask = cc_out == seg_id
        # print(np.sum(cur_mask))
        if np.sum(cur_mask) < size:
            continue

        write_mask += cur_mask

    return write_mask


def find_edge_points(sk_color, point):
    sk_color = sk_color.copy()
    stack = [point]
    edge_points = []
    while stack:
        p = stack.pop()
        sk_color[p[0], p[1], p[2]] = 0

        if np.any(sk_color[p[0] - 1 : p[0] + 2, p[1] - 1 : p[1] + 2, p[2] - 1 : p[2] + 2] == 1):
            edge_points.append(p)
        elif np.any(sk_color[p[0] - 1 : p[0] + 2, p[1] - 1 : p[1] + 2, p[2] - 1 : p[2] + 2] == 2):
            edge_points.append(p)

        ps = np.argwhere(sk_color[p[0] - 1 : p[0] + 2, p[1] - 1 : p[1] + 2, p[2] - 1 : p[2] + 2] == 3)
        for p1 in ps:
            stack.append(p + p1 - 1)

    return edge_points


def connect_2points(p1, p2, arr, color_label):
    points = []
    dist = np.linalg.norm((p1 - p2), 2)
    interplote_step = int(dist + 1) * 2
    points = [p1 + idx * (p2 - p1) / interplote_step for idx in range(interplote_step)]
    for p in points:
        p = np.round(p).astype(int)
        # print(p, arr.shape)
        arr[p[0], p[1], p[2]] = color_label

    return arr


def remove_small_line(p, sk, thresh):
    stack = [p]
    viewed = set()
    while stack:
        cur_p = stack.pop()

        if np.sum(sk[cur_p[0] - 1 : cur_p[0] + 2, cur_p[1] - 1 : cur_p[1] + 2, cur_p[2] - 1 : cur_p[2] + 2]) == 4:
            break
        else:
            viewed.add(str(cur_p.tolist()))
            ps = np.argwhere(
                sk[cur_p[0] - 1 : cur_p[0] + 2, cur_p[1] - 1 : cur_p[1] + 2, cur_p[2] - 1 : cur_p[2] + 2] > 0
            )
            for d in ps:
                d = d - 1
                if str((cur_p + d).tolist()) in viewed:
                    continue
                stack.append(cur_p + d)

    if len(viewed) < thresh:
        for p in viewed:
            p = eval(p)
            sk[p[0], p[1], p[2]] = 0

    return sk


def generate_data(seg_itk, heart_artery_seg_itk, hu_itk, save_path, name, distance_thresh=0, ignore_seg_line_len=5):
    box = getBoundingBox(heart_artery_seg_itk, label=2)

    box_xyz = [box[0], box[1], box[2], box[0] + box[3], box[1] + box[4], box[2] + box[5]]

    seg_arr = sitk.GetArrayFromImage(seg_itk)
    spacing_xyz = seg_itk.GetSpacing()

    cor_mask = seg_arr == cor_label
    cor_mask = binary_fill_holes(cor_mask)

    vein_mask = (seg_arr == vein_label) + (seg_arr == seed_label)
    vein_mask = remove_small_cc(vein_mask, 200)
    vein_mask = binary_fill_holes(vein_mask)

    mask = ((vein_mask + cor_mask) > 0).astype("uint8")

    sk = skeletonize(mask)
    sk_color = sk * cor_mask + sk * vein_mask * 2

    # we want cross over as coronary
    sk_color[sk_color > 2] = 1

    # if vein and cor are close, make them connect
    sk_cor_positions = np.argwhere(sk_color == 1)
    sk_vein_positions = np.argwhere(sk_color == 2)

    sk_vein_dists_array = np.zeros_like(sk_color) - 1
    sk2 = np.zeros_like(sk_color)
    for p in sk_vein_positions:
        dists = np.linalg.norm((sk_cor_positions - p) * spacing_xyz, 2, axis=1)
        idx, min_dist = np.argmin(dists), np.min(dists)
        if min_dist > distance_thresh:
            continue
        p_cor = sk_cor_positions[idx]
        sk2 = connect_2points(p, p_cor, sk2, 1)

    sk[sk2 > 0] = 1
    sk = skeletonize(sk)

    # remove small end line
    for _ in range(4):
        positions = np.argwhere(sk == 1)
        end_points = []
        for p in positions:
            if np.sum(sk[p[0] - 1 : p[0] + 2, p[1] - 1 : p[1] + 2, p[2] - 1 : p[2] + 2]) == 2:
                end_points.append(p)

        for p in end_points:
            sk = remove_small_line(p, sk, 4)

        sk_color = sk + sk * vein_mask
        positions = np.argwhere(sk == 1)
        for p in positions:
            if np.sum(sk[p[0] - 1 : p[0] + 2, p[1] - 1 : p[1] + 2, p[2] - 1 : p[2] + 2]) > 3:
                sk_color[p[0], p[1], p[2]] = 3

    # make sure line_segment only has vein or cor
    cc, cc_num = cc3d.connected_components((sk_color == 1) | (sk_color == 2), return_N=True)
    for cc_id in range(1, cc_num + 1):
        cur_cc = cc == cc_id
        points_num = np.sum(cur_cc)
        ave = np.sum(sk_color[cur_cc]) / np.sum(cur_cc)

        if points_num > 10:
            if ave > 1.5:
                sk_color[cur_cc] = 2
            else:
                sk_color[cur_cc] = 1
        else:
            if ave > 1.2:
                sk_color[cur_cc] = 2
            else:
                sk_color[cur_cc] = 1

    # sitk.WriteImage(
    #     sitk.GetImageFromArray(grey_dilation(sk_color, [3, 3, 3]).astype("uint8")),
    #     # sitk.GetImageFromArray(sk_color.astype("uint8")),
    #     f"/media/e/heart/coronary_seg_train_data/tmp/sk_{name}.nii.gz",
    # )
    # sitk.WriteImage(
    #     sitk.GetImageFromArray(grey_dilation(sk2, [3, 3, 3]).astype("uint8")),
    #     "/media/e/heart/coronary_seg_train_data/tmp/sk2.nii.gz",
    # )
    # sitk.WriteImage(
    #     sitk.GetImageFromArray(vein_mask.astype("uint8")), "/media/e/heart/coronary_seg_train_data/tmp/vein_mask.nii.gz"
    # )
    # sitk.WriteImage(
    #     sitk.GetImageFromArray(cor_mask.astype("uint8")), "/media/e/heart/coronary_seg_train_data/tmp/cor_mask.nii.gz"
    # )

    heart_artery_seg = sitk.GetArrayFromImage(heart_artery_seg_itk)

    np.savez_compressed(
        save_path,
        cor_mask=cor_mask,
        vein_mask=vein_mask,
        sk_line=sk_color,
        box_xyz=box_xyz,
        heart_seg=heart_artery_seg == 2,
        artery_seg=heart_artery_seg == 1,
        spacing_xyz=spacing_xyz,
        hu_vol=sitk.GetArrayFromImage(hu_itk),
    )


if __name__ == "__main__":
    seg_root = "/media/e/heart/coronary_seg_train_data/orgnized_data/nii"
    heart_artery_root = "/media/e/heart/coronary_seg_train_data/orgnized_data/heart_artery"
    dcm_root = "/media/e/heart/coronary_seg_train_data/orgnized_data/dicom"
    save_root = "/media/e/heart/coronary_seg_train_data/tmp"
    save_lst_file = "/media/e/heart/coronary_seg_train_data/tmp/train.lst"

    with open(save_lst_file, "w") as write_fp:
        for f in os.listdir(seg_root):
            print(f)

            name = f.replace("-seg.nii.gz", "")

            seg_path = os.path.join(seg_root, f)
            heart_artery_path = os.path.join(heart_artery_root, f.replace("-seg.nii.gz", "-heart_artery-seg.nii.gz"))
            dcm_path = os.path.join(dcm_root, f.replace("-seg.nii.gz", ".nii.gz"))

            seg_itk = sitk.ReadImage(seg_path)
            heart_artery_itk = sitk.ReadImage(heart_artery_path)
            hu_itk = sitk.ReadImage(dcm_path)

            if seg_itk.GetSize() != hu_itk.GetSize():
                print(f"{name} dicom and seg not equal")

            for distance_thresh in [0, 1.0]:
                save_path = os.path.join(save_root, f.replace("-seg.nii.gz", f"-{distance_thresh}.npz"))
                lst_line = save_path + "\n"

                generate_data(
                    seg_itk, heart_artery_itk, hu_itk, save_path, f"{name}-{distance_thresh}", distance_thresh
                )

                write_fp.write(lst_line)
