#!/usr/bin/env python
# coding: utf-8


import glob
import json
import os
import traceback

import numpy as np
import SimpleITK as sitk
import starship.sitktools as st
import starship.vtktools as vt

import networkx as nx
import vtk
from cpr_tool import image_tool_torch as it
from cpr_tool.coronary_vmtk_helper import (
    connectivity,
    coronaryTree,
    coronaryTreeLR,
    coronaryTreePlus,
    mc,
    skeletonGraphToVTK,
    threshold,
    vmtkCenterlines,
    vmtkExtractor,
    vtkToGraph,
)
from cpr_tool.cpr_common import arclenReparam
from cpr_tool.cpr_torch import getStraightenGrid, getStraightenSliceProps
from cpr_tool.cpr_vtk import drawCenterline, npToVTKIm
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


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


def removeIrregularCell(mod, seedPos, distThres=1.0):
    points = vt.getPoints(mod)
    cells = vt.getAllCellPointIds(mod)
    seedPos = np.array(seedPos).reshape([-1, 3])
    irregularCell = [k for k, i in enumerate(cells) if len(i) < 3]
    irregularCell += [
        k for k, i in enumerate(cells) if np.min(np.linalg.norm(points[i[0]] - seedPos, axis=1)) > distThres
    ]
    for i in irregularCell:
        mod.DeleteCell(i)
    mod.RemoveDeletedCells()
    return mod


def getNodeMapping(g, delta=0):
    return {i: k + delta for k, i in enumerate(g.nodes)}


output_path = "/media/e/heart/coronary_seg_train_data/358/center_line2"
src_dir = "/media/e/heart/coronary_seg_train_data/358"
src_seg_dir = os.path.join(src_dir, "nii")
for fname in glob.glob(os.path.join(src_seg_dir, "*-seg.nii.gz")):
    patID = fname.split("/")[-1].replace("-seg.nii.gz", "")
    try:
        print(fname)
        seg = sitk.ReadImage(fname)
        cor = seg == 1

        # seg = sitk.ReadImage(fname.replace("/nii", "/heart_artery").replace("-seg.nii.gz", "-heart_artery-seg.nii.gz"))
        # aorta = seg == 1

        # seg = sitk.ReadImage(fname.replace("/nii", "/heart_artery").replace("-seg.nii.gz", "-heart_artery-seg.nii.gz"))
        aorta = seg == 3

        aorta = sitk.BinaryFillhole(aorta)
        # aorta = clean_data_by_maxconnect(aorta)

        # aorta = st.resampleByRef(aorta, cor, interpolator=sitk.sitkNearestNeighbor)

        corTree1, corTree2 = coronaryTreeLR(cor, aorta)
        cntline1 = corTree1["skeletonModel"]
        cntline2 = corTree2["skeletonModel"]
        resCntline = vt.append([cntline1, cntline2])
        gCnt1 = corTree1["fullGraph"]
        gCnt2 = corTree2["fullGraph"]
        gCnt1_mapping = getNodeMapping(gCnt1)
        gCnt1_relabeled = nx.relabel_nodes(gCnt1, gCnt1_mapping)
        gCnt2_mapping = getNodeMapping(gCnt2, delta=max(gCnt1_relabeled.nodes) + 1)
        gCnt2_relabeled = nx.relabel_nodes(gCnt2, gCnt2_mapping)
        root = [gCnt1_mapping[gCnt1.graph["rootNode"]], gCnt2_mapping[gCnt2.graph["rootNode"]]]
        pnts = np.concatenate(
            [
                gCnt1.graph["pointsPhysicalPos"][list(gCnt1_mapping.keys())],
                gCnt2.graph["pointsPhysicalPos"][list(gCnt2_mapping.keys())],
            ]
        )
        radius1 = vt.getPointArray(cntline1, "Radius")
        radius2 = vt.getPointArray(cntline2, "Radius")
        radius = np.concatenate([radius1[list(gCnt1_mapping.keys())], radius2[list(gCnt2_mapping.keys())],])
        lines = np.concatenate([np.array(list(gCnt1_relabeled.edges())), np.array(list(gCnt2_relabeled.edges())),])
        assert pnts.shape[0] == np.max(lines) + 1
        resJson = {
            "points": (pnts - cor.GetOrigin()).tolist(),
            "lines": lines.tolist(),
            "radius": radius.tolist(),
            "roots": root,
        }
        with open(output_path + "/%s_centerline.json" % patID, "w") as f:
            json.dump(resJson, f)

        vt.saveVTP(resCntline, output_path + "/%s_centerline.vtp" % patID)
        resSeg = drawCenterline(resCntline, seg)
        sitk.WriteImage(resSeg, output_path + "/%s_centerline.nii.gz" % patID)
    except:
        traceback.print_exc()
        print(fname)
