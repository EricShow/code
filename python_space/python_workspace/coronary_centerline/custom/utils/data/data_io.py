import csv
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import nrrd
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_process import cal_iou
from .find_nodules import find_nodules


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def mkdir(path):
    """create folder of current path.

    :param path: current path
    :return: None
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def load_scans_by_name(dcm_path):
    dcm_list = sorted(os.listdir(dcm_path))
    slices = []
    for dcm in dcm_list:
        if not dcm.endswith('.dcm'):
            continue
        path = os.path.join(dcm_path, dcm)
        dcm = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(dcm)
        slices.append(img[0])

    return np.array(slices)


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    vol = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing = spacing[::-1]
    return vol, img, spacing


def load_dicom(dicom_path):
    try:
        dcm_list = sorted(os.listdir(dicom_path))
        slices = []
        for dcm in dcm_list:
            if not dcm.endswith('.dcm'):
                continue
            path = os.path.join(dicom_path, dcm)
            slices.append(pydicom.read_file(path, force=True))

        # for s in slice_files:
        #     slices.append(pydicom.read_file(s, force=True))
        #     if (not accept) and raise_512 and slices[-1].pixel_array.shape[0] != 512:
        #         print('find 512 scale dicom')
        #         raise NotImplemented
        #     accept = True

        spacing = float(slices[0].PixelSpacing[0])
        find_thickness = True
        slice_thickness = 0
        pre_find = -1
        if 'ImagePositionPatient' in slices[0].dir('ImagePositionPatient'):
            pre_find = 0
        for idx in range(1, len(slices)):
            if 'ImagePositionPatient' in slices[idx].dir('ImagePositionPatient'):
                if pre_find != -1:
                    slice_thickness = np.abs(
                        slices[idx].ImagePositionPatient[2] - slices[pre_find].ImagePositionPatient[2]
                    )
                    find_thickness = True
                    break
                else:
                    pre_find = idx
            else:
                pre_find = -1
        if not find_thickness:
            pre_find = -1
            if 'SliceLocation' in slices[0].dir('SliceLocation'):
                pre_find = 0
            for idx in range(1, len(slices)):
                if 'SliceLocation' in slices[idx].dir('SliceLocation'):
                    if pre_find != -1:
                        slice_thickness = np.abs(slices[pre_find].SliceLocation - slices[idx].SliceLocation)
                        find_thickness = True
                        break
                    else:
                        pre_find = idx
                else:
                    pre_find = -1
        if not find_thickness:
            slice_thickness = 1.0

        img = np.stack([s.pixel_array for s in slices]).astype(np.int16)
        # image = image.astype(np.int16)
        # print 'CT Hu range: ' + str(image.min()) + '--' + str(image.max())

        # Convert to Hounsfield units (HU)
        # intercept = slices[0].RescaleIntercept
        # slope = slices[0].RescaleSlope
        intercept = np.stack([s.RescaleIntercept for s in slices])
        slope = np.stack([s.RescaleSlope for s in slices])
        # if slope != 1:
        img = img.astype(np.float64) * slope[:, np.newaxis, np.newaxis]
        img = img.astype(np.int16)
        img += np.int16(intercept[:, np.newaxis, np.newaxis])
        # lungwin = np.array([window_center - window_width // 2., window_center + window_width // 2])
        # newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        # newimg[newimg < 0] = 0
        # newimg[newimg > 1] = 1
        # newimg = (newimg * 255).astype('uint8')

    except Exception:
        print('==============invalid DICOM===============')
        return None

    return img, slice_thickness, spacing


def load_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
    except Exception:
        print('==============invalid xml path===============')
        return []
    objs = tree.findall('object')
    num_objs = len(objs)
    if num_objs != 0:
        return_list = []
        for obj in objs:
            xml_box = obj.find('bndbox')
            cls_name = obj.find('name').text.lower()
            # if cls_name not in priority_classes:
            #     continue
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            return_list.append([cls_name, xmin, ymin, xmax, ymax])
        return return_list
    else:
        print('================no object=================')
        return []


def load_nrrd(nrrd_path, dicom_path=None):
    reader = sitk.ImageSeriesReader()

    nrrd_data, nrrd_options = nrrd.read(nrrd_path)
    nrrd_data = nrrd_data.swapaxes(0, 2)

    if dicom_path is not None:
        dcm_name = reader.GetGDCMSeriesFileNames(dicom_path)
        if int(dcm_name[0].split('_')[-1].split('.')[0]) > int(dcm_name[1].split('_')[-1].split('.')[0]):
            nrrd_data = np.flip(nrrd_data, axis=0)

    nrrd_data = nrrd_data.astype('uint8')

    return nrrd_data


def load_nrrd_slice(nrrd_path, vol_size, start_idx=1):
    mask_3d = np.zeros(vol_size, dtype='uint8')
    for mask_name in os.listdir(nrrd_path):
        if not mask_name.endswith('.nrrd'):
            continue
        nrrd_data, _ = nrrd.read(os.path.join(nrrd_path, mask_name))
        idx = int(mask_name[-8:-5]) - start_idx
        mask_3d[idx] = nrrd_data
    return mask_3d


def save_nii(data, nii_path, compress=True):
    tmp_img = sitk.GetImageFromArray(data)
    if compress:
        if nii_path.endswith('.nii.gz'):
            sitk.WriteImage(tmp_img, nii_path)
        else:
            sitk.WriteImage(tmp_img, nii_path + '.nii.gz')
    else:
        if nii_path.endswith('.nii'):
            sitk.WriteImage(tmp_img, nii_path)
        else:
            sitk.WriteImage(tmp_img, nii_path + '.nii')


def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np


def save_jpg(data, name):
    """save a 2d image as jpg format.

    :param data: numpy data with dim [3, width, height] or [width, height, 3]
    :param name: saved data path
    :return: None
    """
    if data.ndim != 3:
        print('wrong input data dim when saving as jpg, the dim should equal to 3')
        os._exit(0)

    mkdir(os.path.split(name)[0] + '/')
    if data.dtype != np.ubyte:
        data_f = data.astype(np.float32)
        maxv = np.max(data_f)
        minv = np.min(data_f)
        data = (data_f - minv) / (maxv - minv) * 255.0
        data = data.astype(np.ubyte)
    if data.shape[0] == 3:
        data = data.transpose((2, 0, 1))

    cv2.imwrite(str(name) + '.jpg', data)


def load_jpg(path, color='RGB'):
    """load jpg data.

    :param path: loading path
    :param color: load 'RGB' or 'L'
    :return: jpg numpy data, dim is 3 for 'RGB' or 1 for 'L'
    """
    img = Image.open(path).convert(color)
    img_array = np.asarray(img)
    return img_array


def load_model(net, load_dir, load_weights=True, distribution=False):
    """loading model function.

    :param net: defined network
    :param load_dir: the path for loading
    :return: network
    """
    if load_weights:
        checkpoint = torch.load(load_dir)
        net.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if distribution:
            torch.distributed.init_process_group(
                'nccl',
                init_method='file:///home/tx-deepocean/project/Airway_Segmentation/master/seg_airway/train/temp/method',
                world_size=1,
                rank=0
            )
            net = torch.nn.parallel.DistributedDataParallel(net.cuda())
        else:
            net = torch.nn.DataParallel(net).cuda()
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        net = net.cuda()
    return net


def save_model(net, epoch, save_dir):
    """saving model function.

    :param net:  defined network
    :param epoch: current saved epoch
    :param save_dir: the path for saving
    :return: None
    """

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save(
        {
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict
        }, os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch + 1))
    )


def load_csv(csv_path):
    if not os.path.exists(csv_path):
        print('csv file is not exist!')
        return []
    csv_data = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            csv_data.append(row)
    return csv_data


def load_numpy_vol(mask_path, data_size, start_idx=1):
    slice_list = [i for i in os.listdir(mask_path) if i.endswith('.npy')]
    if len(slice_list) != data_size[0]:
        return None

    mask_3d = np.zeros(data_size, dtype='uint8')
    for mask_name in os.listdir(mask_path):
        if not mask_name.endswith('.npy'):
            continue
        mask = np.load(os.path.join(mask_path, mask_name))
        idx = int(mask_name[-7:-4]) - start_idx
        mask_3d[idx] = mask
    return mask_3d


def get_nodule_class():
    classes = [
        '__background__', 'calcific_nodule', 'pleural_nodule', '0-3nodule', '3-6nodule', '6-10nodule', '10-30nodule',
        'mass', '0-5mGGN', '0-5pGGN', '5mGGN', '5pGGN'
    ]
    CLASSES = classes
    CLASS_DICT = {
        'calcific nodule': 'calcific_nodule',
        'pleural nodule': 'pleural_nodule',
        'solid nodule': '3-6nodule',
        '0-3nodule': '0-3nodule',
        'GGN': '5pGGN',
        '3-6nodule': '3-6nodule',
        '6-10nodule': '6-10nodule',
        'pleural calcific nodule': 'calcific_nodule',
        '10-30nodule': '10-30nodule',
        'mass': 'mass',
        '0-5GGN': '0-5pGGN',
        '5GGN': '5pGGN',
        '5pGGN': '5pGGN',
        '0-5pGGN': '0-5pGGN',
        '5mGGN': '5mGGN',
        '0-5mGGN': '0-5mGGN',
        '0-3GGN': '0-5pGGN',
        'suspect nodule': '0-3nodule',
        'suspect GGN': '0-5pGGN'
    }
    detailed_class = [s.lower() for s in CLASS_DICT.keys()]
    return detailed_class


def load_bbox_3d(path, remove_fp=True):
    '''
    :param path:
    :param iou_th:
    :return bbox3d: xmin, ymin, zmin, xmax, ymax, zmax
            nodule_type
    '''

    iou_th = 0.6

    def is_same_nodule(bbox_a, bbox_b):
        epsilon = 1e-5
        if cal_iou(bbox_a, bbox_b) > iou_th and abs(bbox_a[5] - bbox_b[5]) < 1 + epsilon:
            return True
        else:
            return False

    def push_bbox_list(bbox_list, bbox):
        is_new = True
        for nodule in bbox_list:
            if (is_same_nodule(nodule[-1], bbox)):
                is_new = False
                nodule.append(bbox)
        if is_new:
            bbox_list.append([bbox])

    xml_list = os.listdir(path)
    xml_list.sort(key=lambda x: int(x[-7:-4]))
    bbox_list = []
    for single_xml in xml_list:
        slice = float(single_xml[-7:-4]) - 1.0
        anno = load_xml(os.path.join(path, single_xml))
        for bbox in anno:
            if remove_fp and (bbox[0].lower() not in get_nodule_class()):
                continue
            bbox.append(slice)
            push_bbox_list(bbox_list, bbox)
    bbox3d = []
    nodule_type = []
    for bboxes in bbox_list:
        zmin = int(bboxes[0][5])
        zmax = int(bboxes[-1][5])
        bbox_mid = bboxes[len(bboxes) // 2]
        bbox3d.append([int(bbox_mid[1]), int(bbox_mid[2]), zmin, int(bbox_mid[3]), int(bbox_mid[4]), zmax])
        nodule_type.append(bbox_mid[0])
    return bbox3d, nodule_type

    # bbox_pd = pd.DataFrame({'instanceNumber': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [],
    #                         'name': [], 'prob': []})
    #
    # xml_list = os.listdir(path)
    # xml_list.sort(key=lambda x: int(x[-7:-4]))
    # for single_xml in xml_list:
    #     slice = float(single_xml[-7:-4]) - 1.0
    #     anno = load_xml(os.path.join(path, single_xml))
    #     for bbox in anno:
    #         if remove_fp and (bbox[0].lower() not in get_nodule_class()):
    #             continue
    #         df_add_row = {'instanceNumber': slice,
    #                       'xmin': bbox[1], 'ymin': bbox[2], 'xmax': bbox[3], 'ymax': bbox[4],
    #                       'name': bbox[0], 'prob': 1}
    #         bbox_pd = bbox_pd.append(df_add_row, ignore_index=True)
    # nodule_bbox, _ = find_nodules(bbox_pd)
    # x_min = nodule_bbox.groupby("nodule")["xmin"].min()
    # x_max = nodule_bbox.groupby("nodule")["xmax"].max()
    # y_min = nodule_bbox.groupby("nodule")["ymin"].min()
    # y_max = nodule_bbox.groupby("nodule")["ymax"].max()
    # z_min = nodule_bbox.groupby("nodule")["instanceNumber"].min()
    # z_max = nodule_bbox.groupby("nodule")["instanceNumber"].max()
    # nodule_type = nodule_bbox.groupby("nodule")["name"].agg(pd.Series.mode).reset_index()
    # nodule_type = nodule_type["name"].values.tolist()
    #
    # bbox_3d_list = []
    # for i in range(len(x_min)):
    #     bbox_3d_list.append([int(x_min.iloc[i]), int(y_min.iloc[i]), int(z_min.iloc[i]),
    #                          int(x_max.iloc[i]), int(y_max.iloc[i]), int(z_max.iloc[i])])
    # return bbox_3d_list, nodule_type


def load_bbox_2d_3d(path, remove_fp=True):
    '''
    :param path:
    :param iou_th:
    :return bbox3d: xmin, ymin, zmin, xmax, ymax, zmax
    bbox2d: cls_name, xmin, ymin, xmax, ymax, z, size_z
            nodule_type
    '''

    bbox_pd = pd.DataFrame(
        {
            'instanceNumber': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': [],
            'name': [],
            'prob': []
        }
    )

    xml_list = os.listdir(path)
    # xml_list = [xml_list[i].endwith("xml") for i in len(xml_list)]
    xml_list = [i for i in xml_list if i.endswith('xml')]
    xml_list.sort(key=lambda x: int(x[-7:-4]))
    for single_xml in xml_list:
        slice = float(single_xml[-7:-4]) - 1.0
        anno = load_xml(os.path.join(path, single_xml))
        for bbox in anno:
            if remove_fp and (bbox[0].lower() not in get_nodule_class()):
                continue
            df_add_row = {
                'instanceNumber': slice,
                'xmin': bbox[1],
                'ymin': bbox[2],
                'xmax': bbox[3],
                'ymax': bbox[4],
                'name': bbox[0],
                'prob': 1
            }
            bbox_pd = bbox_pd.append(df_add_row, ignore_index=True)

    bbox_pd.eval('size_2d = (ymax - ymin)*(xmax - xmin)', inplace=True)
    bbox_pd = bbox_pd.sort_values(by=['size_2d'], ascending=False)
    bbox_pd = bbox_pd.drop(['size_2d'], axis=1)
    bbox_pd = bbox_pd.reset_index(drop=True)

    nodule_bbox, _ = find_nodules(bbox_pd)
    x_min = nodule_bbox.groupby('nodule')['xmin'].min()
    x_max = nodule_bbox.groupby('nodule')['xmax'].max()
    y_min = nodule_bbox.groupby('nodule')['ymin'].min()
    y_max = nodule_bbox.groupby('nodule')['ymax'].max()
    z_min = nodule_bbox.groupby('nodule')['instanceNumber'].min()
    z_max = nodule_bbox.groupby('nodule')['instanceNumber'].max()
    nodule_type = nodule_bbox.groupby('nodule')['name'].agg(lambda x: x.value_counts().index[0]).reset_index()
    nodule_type = nodule_type['name'].values.tolist()

    bbox_2d_list = nodule_bbox.loc[:,
                                   ['name', 'xmin', 'ymin', 'xmax', 'ymax', 'instanceNumber', 'nodule']].values.tolist()
    for bbox_2d in bbox_2d_list:
        nodule_idx = bbox_2d[6] - 1
        bbox_2d.pop()
        bbox_2d.append(z_max.iloc[nodule_idx] - z_min.iloc[nodule_idx] + 2)

    bbox_3d_list = []
    # bbox_2d_3d_index = []
    # nodule_bbox_list = list(nodule_bbox.groupby("nodule"))
    for i in range(len(x_min)):
        bbox_3d_list.append(
            [
                int(x_min.iloc[i]),
                int(y_min.iloc[i]),
                int(z_min.iloc[i]),
                int(x_max.iloc[i]),
                int(y_max.iloc[i]),
                int(z_max.iloc[i])
            ]
        )

        # bbox_2d_3d_index.append(nodule_bbox_list[i][1].index.values.tolist())

    return bbox_2d_list, bbox_3d_list, nodule_type


def load_bbox_2d(path, remove_fp=True):
    '''
        :param path:
        :param iou_th:
        :return bbox2d: cls_name, xmin, ymin, xmax, ymax, z
        '''
    xml_list = os.listdir(path)
    xml_list.sort(key=lambda x: int(x[-7:-4]))
    bbox_list = []
    for single_xml in xml_list:
        slice = float(single_xml[-7:-4]) - 1.0
        anno = load_xml(os.path.join(path, single_xml))
        for bbox in anno:
            if remove_fp and (bbox[0].lower() not in get_nodule_class()):
                continue
            bbox_list.append(bbox + [slice])
    return bbox_list


def convert_to_uint8(vol, win_level=-600, win_width=1500):
    vol_f = vol.astype('float32')
    vol_f -= (win_level - win_width / 2)
    vol_f = vol_f / win_width * 255.0
    vol_f = np.clip(vol_f, 0.0, 255.0)
    vol_n = vol_f.astype('uint8')
    return vol_n


def save_vol_to_jpg(vol, path, win_level=-600, win_width=1500):
    assert len(vol.shape) == 3
    if vol.dtype != np.uint8:
        vol_n = convert_to_uint8(vol, win_level, win_width)
    else:
        vol_n = np.copy(vol)

    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(vol.shape[0]):
        img = vol_n[i]
        cv2.imwrite(os.path.join(path, '%04d.jpg' % i), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_slice_to_jpg(slice, path, win_level=-600, win_width=1500):
    assert len(slice.shape) == 2
    if vol.dtype != np.uint8:
        slice_n = convert_to_uint8(slice, win_level, win_width)
    else:
        slice_n = np.copy(slice)

    file_path, _ = os.path.split(path)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    cv2.imwrite(path, slice_n, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_contours_3d(vol, mask, path, win_level=-600, win_width=1500):
    assert len(vol.shape) == 3

    if vol.dtype != np.uint8:
        vol_n = convert_to_uint8(vol, win_level, win_width)
    else:
        vol_n = np.copy(vol)

    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(vol_n.shape[0]):
        image = vol_n[i]
        image = np.expand_dims(image, 2)
        image = np.tile(image, (1, 1, 3))
        image_mask = mask[i] * 255
        contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(path, '%04d.jpg' % (i + 1)), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_bbox_jpg(vol, bbox_2d_list, path, win_level=-600, win_width=1500):
    # bbox2d: cls_name, xmin, ymin, xmax, ymax, z
    def get_slice_bbox_idx(size_z, bbox_2d_list):
        slice_bbox_idx = [[] for i in range(size_z)]
        for i, bbox_2d in enumerate(bbox_2d_list):
            slice_bbox_idx[int(bbox_2d[5])].append(i)
        return slice_bbox_idx

    assert len(vol.shape) == 3

    if vol.dtype != np.uint8:
        vol_n = convert_to_uint8(vol, win_level, win_width)
    else:
        vol_n = np.copy(vol)

    slice_bbox_idx = get_slice_bbox_idx(vol.shape[0], bbox_2d_list)
    for i in range(vol_n.shape[0]):
        image = vol_n[i]
        image = np.expand_dims(image, 2)
        image = np.tile(image, (1, 1, 3))
        for bbox_idx in slice_bbox_idx[i]:
            bbox_2d = bbox_2d_list[bbox_idx]
            cv2.rectangle(image, (int(bbox_2d[1]), int(bbox_2d[2])), (int(bbox_2d[3]), int(bbox_2d[4])), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(path, '%04d.jpg' % (i + 1)), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_bbox2d_and_contours_jpg(vol, mask, bbox_2d_list, path, win_level=-600, win_width=1500):
    # bbox2d: cls_name, xmin, ymin, xmax, ymax, z
    assert len(vol.shape) == 3

    if not os.path.exists(path):
        os.mkdir(path)

    if vol.dtype != np.uint8:
        vol_n = convert_to_uint8(vol, win_level, win_width)
    else:
        vol_n = np.copy(vol)

    slice_bbox_2d_idx = get_slice_bbox_2d_idx(vol.shape[0], bbox_2d_list)

    # font = cv2.FONT_HERSHEY_PLAIN

    for i in range(vol_n.shape[0]):
        image = vol_n[i]
        image = np.expand_dims(image, 2)
        image = np.tile(image, (1, 1, 3))

        image_mask = mask[i] * 255
        contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        for bbox_idx in slice_bbox_2d_idx[i]:
            bbox_2d = bbox_2d_list[bbox_idx]
            patch = mask[i, int(bbox_2d[2]):int(bbox_2d[4]), int(bbox_2d[1]):int(bbox_2d[3])]
            if np.sum(patch) == 0:
                color = (255, 0, 0)  # blue
            else:
                color = (0, 255, 0)  # green
            cv2.rectangle(image, (int(bbox_2d[1]), int(bbox_2d[2])), (int(bbox_2d[3]), int(bbox_2d[4])), color, 1)

        # if axes_list != []:
        #     for bbox_idx in slice_bbox_3d_idx[i]:
        #         bbox_3d = bbox_3d_list[bbox_idx]
        #         text_coord = (int(bbox_3d[0]), int(bbox_3d[1]))
        #         text = "%.1f_%.1f" % (axes_list[bbox_idx][0], axes_list[bbox_idx][1])
        #         cv2.putText(image, text, text_coord, font, 1, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(path, '%04d.jpg' % (i + 1)), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# def save_bbox_and_contours_jpg(vol, bbox_2d_list, bbox_3d_list, axes_list, mask, path, win_level=-600, win_width=1500):
#     # bbox2d: cls_name, xmin, ymin, xmax, ymax, z
#     assert len(vol.shape) == 3
#
#     if not os.path.exists(path):
#         os.mkdir(path)
#
#     if vol.dtype != np.uint8:
#         vol_n = convert_to_uint8(vol, win_level, win_width)
#     else:
#         vol_n = np.copy(vol)
#
#     slice_bbox_2d_idx = get_slice_bbox_2d_idx(vol.shape[0], bbox_2d_list)
#     slice_bbox_3d_idx = get_slice_bbox_3d_idx(vol.shape[0], bbox_3d_list)
#
#     font = cv2.FONT_HERSHEY_PLAIN
#
#     for i in range(vol_n.shape[0]):
#         image = vol_n[i]
#         image = np.expand_dims(image, 2)
#         image = np.tile(image, (1, 1, 3))
#
#         image_mask = mask[i] * 255
#
#         image = cv2.resize(image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#         image_mask = cv2.resize(image_mask, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#         image_mask[image_mask > 127] = 255
#         image_mask[image_mask <= 127] = 0
#
#         contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
#
#         # for bbox_idx in slice_bbox_2d_idx[i]:
#         #     bbox_2d = bbox_2d_list[bbox_idx]
#         #     patch = mask[i, int(bbox_2d[2]):int(bbox_2d[4]), int(bbox_2d[1]):int(bbox_2d[3])]
#         #     if np.sum(patch) == 0:
#         #         color = (255, 0, 0)  # blue
#         #     else:
#         #         color = (0, 255, 0)  # green
#         #     cv2.rectangle(image, (int(bbox_2d[1]), int(bbox_2d[2])), (int(bbox_2d[3]), int(bbox_2d[4])), color, 1)
#         #
#         # for bbox_idx in slice_bbox_3d_idx[i]:
#         #     bbox_3d = bbox_3d_list[bbox_idx]
#         #     text_coord = (int(bbox_3d[0]), int(bbox_3d[1]))
#         #     text = "%.1f_%.1f" % (axes_list[bbox_idx][0], axes_list[bbox_idx][1])
#         #     cv2.putText(image, text, text_coord, font, 1, (0, 255, 0), 1)
#
#         cv2.imwrite(os.path.join(path, "%04d.jpg" % (i + 1)), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def save_heatmap_jpg(vol, heatmap, bbox_2d_list, path, win_level=-600, win_width=1500):
    # bbox2d: cls_name, xmin, ymin, xmax, ymax, z
    if not os.path.exists(path):
        os.mkdir(path)

    if vol.dtype != np.uint8:
        vol_n = convert_to_uint8(vol, win_level, win_width)
    else:
        vol_n = np.copy(vol)

    slice_bbox_2d_idx = get_slice_bbox_2d_idx(vol.shape[0], bbox_2d_list)
    for i in range(vol_n.shape[0]):
        image = vol_n[i]
        image = np.expand_dims(image, 2)
        image = np.tile(image, (1, 1, 3))
        for bbox_idx in slice_bbox_2d_idx[i]:
            bbox_2d = bbox_2d_list[bbox_idx]
            color = (0, 255, 0)  # green
            cv2.rectangle(image, (int(bbox_2d[1]), int(bbox_2d[2])), (int(bbox_2d[3]), int(bbox_2d[4])), color, 1)

        image_heatmap = (heatmap[i] * 255).astype('uint8')
        image_heatmap = cv2.applyColorMap(image_heatmap, cv2.COLORMAP_HOT)

        img_add = cv2.addWeighted(image, 0.5, image_heatmap, 0.5, 0)

        size = image.shape
        merge_img = np.ndarray((size[0], size[1] * 2, size[2]), dtype='uint8')
        merge_img[0:size[0], 0:size[1], :] = img_add
        merge_img[0:size[0], size[1]:size[1] * 2, :] = image

        cv2.imwrite(os.path.join(path, '%04d.jpg' % (i + 1)), merge_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def get_slice_bbox_2d_idx(size_z, bbox_2d_list):
    slice_bbox_idx = [[] for i in range(size_z)]
    for i, bbox_2d in enumerate(bbox_2d_list):
        slice_bbox_idx[int(bbox_2d[5])].append(i)
    return slice_bbox_idx


def get_slice_bbox_3d_idx(size_z, bbox_3d_list):
    slice_bbox_idx = [[] for _ in range(size_z)]
    for i, bbox_3d in enumerate(bbox_3d_list):
        for j in range(int(bbox_3d[2]), int(bbox_3d[5]) + 1):
            slice_bbox_idx[j].append(i)
    return slice_bbox_idx


def save_bbox3d_and_contours_jpg(vol, mask, bbox_3d_list, path, padding=5, win_level=-600, win_width=1500):
    # bbox_2d: cls_name, xmin, ymin, xmax, ymax, z
    # bbox_3d: xmin, ymin, zmin, xmax, ymax, zmax
    bbox_2d_list = []
    for bbox_3d in bbox_3d_list:
        for z in range(bbox_3d[2], bbox_3d[5]):
            bbox_2d_list.append(
                ['', bbox_3d[0] - padding, bbox_3d[1] - padding, bbox_3d[3] + padding, bbox_3d[4] + padding, z]
            )

    save_bbox2d_and_contours_jpg(vol, mask, bbox_2d_list, path, win_level, win_width)


def save_heatmap_and_contours_jpg(vol, heatmap, mask, bbox_2d_list, path, win_level=-600, win_width=1500):
    # bbox2d: cls_name, xmin, ymin, xmax, ymax, z
    if not os.path.exists(path):
        os.mkdir(path)

    if vol.dtype != np.uint8:
        vol_n = convert_to_uint8(vol, win_level, win_width)
    else:
        vol_n = np.copy(vol)

    slice_bbox_2d_idx = get_slice_bbox_2d_idx(vol.shape[0], bbox_2d_list)
    for i in range(vol_n.shape[0]):
        image = vol_n[i]
        image = np.expand_dims(image, 2)
        image = np.tile(image, (1, 1, 3))

        # draw bbox
        for bbox_idx in slice_bbox_2d_idx[i]:
            bbox_2d = bbox_2d_list[bbox_idx]
            patch = mask[i, int(bbox_2d[2]):int(bbox_2d[4]), int(bbox_2d[1]):int(bbox_2d[3])]
            if np.sum(patch) == 0:
                color = (255, 0, 0)  # blue
            else:
                color = (0, 255, 0)  # green
            cv2.rectangle(image, (int(bbox_2d[1]), int(bbox_2d[2])), (int(bbox_2d[3]), int(bbox_2d[4])), color, 1)

        # draw heatmap
        image_heatmap = (heatmap[i] * 255).astype('uint8')
        image_heatmap = cv2.applyColorMap(image_heatmap, cv2.COLORMAP_HOT)
        img_add = cv2.addWeighted(image, 0.5, image_heatmap, 0.5, 0)

        # draw contours
        image_mask = mask[i] * 255
        contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        size = image.shape
        merge_img = np.ndarray((size[0], size[1] * 2, size[2]), dtype='uint8')
        merge_img[0:size[0], 0:size[1], :] = img_add
        merge_img[0:size[0], size[1]:size[1] * 2, :] = image

        cv2.imwrite(os.path.join(path, '%04d.jpg' % (i + 1)), merge_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def merge_jpg(pred_path, true_path, merge_path):
    if not os.path.exists(merge_path):
        os.mkdir(merge_path)
    for img_name in os.listdir(pred_path):
        pred_img = cv2.imread(os.path.join(pred_path, img_name), cv2.IMREAD_COLOR)
        true_img = cv2.imread(os.path.join(true_path, img_name), cv2.IMREAD_COLOR)
        size = pred_img.shape
        merge_img = np.ndarray((size[0], size[1] * 2, size[2]), dtype='uint8')
        merge_img[0:size[0], 0:size[1], :] = pred_img
        merge_img[0:size[0], size[1]:size[1] * 2, :] = true_img
        cv2.imwrite(os.path.join(merge_path, img_name), merge_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def load_numpy_mask(mask_path, data_size):
    mask_3d = np.zeros(data_size, dtype='uint8')
    for mask_name in os.listdir(mask_path):
        if not mask_name.endswith('.npy'):
            continue
        mask = np.load(os.path.join(mask_path, mask_name))
        idx = int(mask_name[-7:-4]) - 1
        mask_3d[idx] = mask
    return mask_3d


if __name__ == '__main__':
    bbox_path = '/home/tx-deepocean/data/Nodule_Segmentation/all_data/part1/2019_02_22_doctor_results/anno/JSLSCT1803270099T125'
    load_bbox_2d_3d(bbox_path, remove_fp=False)

    # img = cv2.imread("../../temp/test.jpeg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # data_path = "/home/tx-deepocean/data/seg_train_debug/all_data/new_organized"
    # patient_id = "HEBFY0012549915T100"
    # save_path = "../../temp/save_contours"
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    #
    # vol, _ = load_dicom(os.path.join(data_path, "dicom", patient_id))
    # mask = load_numpy_mask(os.path.join(data_path, "mask", patient_id), vol.shape)
    # save_contours_3d(vol, mask, save_path)
