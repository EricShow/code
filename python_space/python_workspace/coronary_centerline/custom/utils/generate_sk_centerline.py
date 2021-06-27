import SimpleITK as sitk
import numpy as np
import starship.sitktools as st


import networkx as nx
from cpr_tool.coronary_vmtk_helper import (
    coronaryTree,
    coronaryTreePlus,
    vtkToGraph,
    skeletonGraphToVTK,
    vmtkCenterlines,
    mc,
)
from cpr_tool.cpr_common import arclenReparam
from cpr_tool.cpr_torch import getStraightenSliceProps, getStraightenGrid
from cpr_tool.cpr_vtk import npToVTKIm, drawCenterline
from cpr_tool import image_tool_torch as it
import starship.vtktools as vt
import glob
from starship.sitktools import resampleByRef, convIndexToPhysicalPoint
from skimage.morphology import skeletonize
from VesselProcessUtils.DistanceTransform import DistanceTransform

import os
import json
import traceback


def clean_data_by_maxconnect(mask_img):
    mask_connect = sitk.ConnectedComponent(mask_img)
    stat = sitk.LabelShapeStatisticsImageFilter()
    stat.Execute(mask_connect)
    main_label = max(stat.GetLabels(), key=lambda x: stat.GetNumberOfPixels(x))
    clean_mask = mask_img * (mask_connect == main_label)
    return clean_mask


def remove_mask_img_mask_noise(mask_img, size_thresh=5):
    mask_connect = sitk.ConnectedComponent(mask_img)
    stat = sitk.LabelShapeStatisticsImageFilter()
    stat.Execute(mask_connect)
    relabel_map = {i: 0 for i in stat.GetLabels() if stat.GetPhysicalSize(i) < size_thresh}
    mask_connect = sitk.ChangeLabel(mask_connect, relabel_map)

    return mask_img * (mask_connect > 0)


def get_sk_radius(array, points, radiu_scale, gpu):
    distance_array = DistanceTransform(array, 20, gpu)
    results = []
    for point in points:
        tmp_cube = distance_array[point[0] - 1 : point[0] + 2, point[1] - 1 : point[1] + 2, point[2] - 1 : point[2] + 2]
        radius = np.max(tmp_cube)
        results.append(radius * radiu_scale)
    return results


def get_line(points, sk):
    sk = sk.copy()
    lines = []
    lut = {}
    for i, p in enumerate(points):
        lut[tuple(p)] = i

    for i, p in enumerate(points):
        ps = np.argwhere(sk[p[0] - 1 : p[0] + 2, p[1] - 1 : p[1] + 2, p[2] - 1 : p[2] + 2] > 0)
        sk[p[0], p[1], p[2]] = 0
        for p1 in ps:
            p1 = p1 + p - 1
            idx = lut[tuple(p1)]
            if i == idx:
                continue
            lines.append([i, idx])

    return lines


output_path = "/media/e/heart/coronary_seg_train_data/358/centerline_sk2"

src_dir = "/media/e/heart/coronary_seg_train_data/358"
src_seg_dir = os.path.join(src_dir, "nii")
for fname in glob.glob(os.path.join(src_seg_dir, "*-seg.nii.gz")):
    patID = fname.split("/")[-1].replace("-seg.nii.gz", "")

    print(fname)
    seg = sitk.ReadImage(fname)
    aorta = seg == 3
    aorta = sitk.BinaryFillhole(aorta)
    aorta = clean_data_by_maxconnect(aorta)

    cor = seg == 1
    cor_aorta_bin = sitk.BinaryFillhole(sitk.Or(cor, aorta))
    # cor_aorta_bin = clean_data_by_maxconnect(cor_aorta_bin)
    cor = cor_aorta_bin - aorta
    cor = remove_mask_img_mask_noise(cor, 2)
    cor = resampleByRef(cor, spacing=[0.3, 0.3, 0.3])

    cor = sitk.SmoothingRecursiveGaussian(cor, 0.6)
    cor = cor > 0.1
    cor = sitk.BinaryFillhole(cor)

    cor = remove_mask_img_mask_noise(cor, 10)

    # sitk.WriteImage(cor, "/media/d/tx_data/tmp/heart/tmp/tmp.nii.gz")

    sk = skeletonize(sitk.GetArrayFromImage(cor))

    points = np.argwhere(sk > 0)

    lines = get_line(points, sk)

    radius = get_sk_radius(sitk.GetArrayFromImage(cor), points, 0.45, 0)

    points = points[:, ::-1]
    phy_points = convIndexToPhysicalPoint(points, cor.GetOrigin(), cor.GetSpacing(), cor.GetDirection())

    # resJson = {"points": [p.tolist() for p in phy_points]}
    resJson = {
        "points": phy_points.T.tolist(),
        "lines": lines,
        "radius": radius,
    }
    with open(output_path + "/%s_centerline.json" % patID, "w") as f:
        json.dump(resJson, f)

    # cor = resampleByRef(cor, spacing=np.array(cor.GetSpacing()) / 2.0)
    # corSurf = mc(cor, 1, iterations=1000, pass_band=0.2, gpuId=1)
    # corSurf = vt.triangle(vt.cleanPolyData(corSurf))
    # # corSurf = vt.cleanPolyData(corSurf)
    # points = vt.getPoints(corSurf)

    # bias = np.min(points, axis=0) - 10
    # points = points - bias
    # points = np.round(points).astype(int)

    # sk_arr = np.zeros(np.max(points, axis=0) + 10).astype("uint8")
    # for p in points:
    #     # print(p)
    #     sk_arr[p[0], p[1], p[2]] = 1
    # sitk_sk = sitk.GetImageFromArray(sk_arr)

    # sitk.WriteImage(cor, "/media/d/tx_data/tmp/heart/tmp/tmp.nii.gz")
