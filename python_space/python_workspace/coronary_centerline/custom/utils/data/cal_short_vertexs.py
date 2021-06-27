import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

pi = math.pi


def get_unit_vector(start_cor, end_cor):
    """返回一个单位向量.

    :param start_cor:起始坐标,[1,0]
    :param end_cor: 终点坐标,[1,0]
    :return: 单位向量,[1,0]
    """
    temp_vector = np.array(end_cor) - np.array(start_cor)
    length_vector = np.sqrt(temp_vector.dot(temp_vector))
    if length_vector == 0:
        return np.zeros(2)
    else:
        return temp_vector / float(length_vector)


def get_angle(long_vertexs_vector, short_vertexs_vector):
    """计算两个向量的乘积.

    :param long_vertexs_vector:np.array([0,1])
    :param short_vertexs_vector:np.array([0,1])
    :return:向量乘积
    """
    return long_vertexs_vector.dot(short_vertexs_vector)


def get_points(angle):
    """返回待选点对.

    :param angle:向量乘积结果, np.array, [0,1]
    :return: [[1,2],[2,3]]
    """
    indexs = []
    for i in range(len(angle) - 1):
        if angle[i] == 0:
            indexs.append([i, i])
        elif angle[i] * angle[i + 1] < 0:
            indexs.append([i, i + 1])
    if angle[0] * angle[-1] < 0:
        indexs.append([0, len(angle) - 1])
    return indexs


def cal_meet(contour, indexs, ori_cor, vector):
    """计算条直线的交点.

    :param contour:边沿，[[[1,0]],[[2,0]]]
    :param indexs: 交点点对, [[1,0],[1,2]]
    :return: 最长短径的坐标对和长度,[[1,0],[2,0]]
    """
    tar_cor = []
    tar_len = 0
    for i in range(len(indexs)):
        if indexs[i][0] == indexs[i][1]:
            temp_cor = np.array([contour[indexs[i][0]][0][0], contour[indexs[i][1]][0][1], ori_cor[0], ori_cor[1]])
        else:
            x0 = contour[indexs[i][0]][0][0]
            y0 = contour[indexs[i][0]][0][1]
            x1 = contour[indexs[i][1]][0][0]
            y1 = contour[indexs[i][1]][0][1]
            x2 = ori_cor[0]
            y2 = ori_cor[1]
            x3 = ori_cor[0] + vector[0]
            y3 = ori_cor[1] + vector[1]
            a = y1 - y0
            b = x1 * y0 - x0 * y1
            c = x1 - x0
            d = y3 - y2
            e = x3 * y2 - x2 * y3
            f = x3 - x2
            try:
                y_d = float(a * e - b * d) / (a * f - c * d)
                x_d = float(y_d * c - b) / a
            except Exception:
                y_d = y1
                x_d = -(y2 - y1) * (x3 - x2) / float(y3 - y2) + x2
            temp_cor = np.array([x_d, y_d, ori_cor[0], ori_cor[1]])
        temp_vector = temp_cor[:2] - temp_cor[2:]
        temp_len = np.sqrt(temp_vector.dot(temp_vector))
        if temp_len > tar_len:
            tar_cor = temp_cor
            tar_len = temp_len
    return tar_cor, tar_len


def get_short_vertexs(contour_nd, long_vertexs):
    """通过边沿坐标和长径计算短径坐标.

    :param contour: 边沿, [[[1,0]],[[2,0]]]
    :param long_vertexs:长径 , [[1,2],[3,3]]
    :return: 短径坐标对,[[1,2],[2,3]]
    """
    # 计算长径的单位向量
    import copy
    contour = copy.copy(contour_nd)
    long_vertexs_vector = get_unit_vector(long_vertexs[0], long_vertexs[1])
    tar_cor = []
    tar_len = 0
    try:
        contour = contour.tolist()
    except Exception:
        pass
    for i in range(len(contour)):
        angle = []
        for j in range(len(contour)):
            temp_unit_vector = get_unit_vector(contour[i][0], contour[j][0])
            angle.append(get_angle(long_vertexs_vector, temp_unit_vector))
        indexs = get_points(angle)
        temp_cor, temp_len = cal_meet(
            contour, indexs, contour[i][0], np.array([-long_vertexs_vector[1], long_vertexs_vector[0]])
        )
        if temp_len > tar_len:
            tar_len = temp_len
            tar_cor = temp_cor
    if tar_len == 0:
        print(1111)
    tar_cor = np.round(tar_cor)
    tar_cor = tar_cor.reshape((2, 2)).astype(int)
    return tar_len, tar_cor


def show_vertexs(shape, contour, long_vertexs, short_vertexs, box):
    temp = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(temp, contour, -1, 255, thickness=1)
    cv2.rectangle(temp, box[0], box[1], 255, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(temp, tuple(long_vertexs[0]), tuple(long_vertexs[1]), (150))
    cv2.line(temp, tuple(short_vertexs[0]), tuple(short_vertexs[1]), (100))
    cv2.imshow('kk', temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calc_polar_angle(pos1, pos2):
    rad = math.atan(float(pos2[1] - pos1[1]) / (pos2[0] - pos1[0] + 1e-6))
    if rad < 0:
        rad += pi
    if rad == 0:
        if pos2[0] < pos1[0]:
            rad += pi
    return rad


def graham_scan(anchor_idx, poses):
    hull_stack = [anchor_idx]
    unhull_stack = [x for x in range(len(poses)) if x != anchor_idx]
    polar_angles = [calc_polar_angle(poses[anchor_idx], poses[idx]) for idx in unhull_stack]
    idx_polar = range(len(polar_angles))
    sorted_idx_polar = sorted(idx_polar, key=lambda x: polar_angles[x])
    i_sorted_polar = 0
    len_sorted = len(sorted_idx_polar)
    while i_sorted_polar < len_sorted:
        idx_p = sorted_idx_polar[i_sorted_polar]
        idx = unhull_stack[idx_p]
        if len(hull_stack) == 1:
            pos1 = poses[hull_stack[0]]
            pos2 = poses[idx]
            if not ((pos2[0] == pos1[0]) and (pos2[1] == pos1[1])):
                hull_stack.append(idx)
            i_sorted_polar += 1
        else:
            idx0 = hull_stack[-2]
            idx1 = hull_stack[-1]
            pos0 = poses[idx0]
            pos1 = poses[idx1]
            pos2 = poses[idx]
            cross_prod = (pos1[0] - pos0[0]) * (pos2[1] - pos1[1]) - (pos2[0] - pos1[0]) * (pos1[1] - pos0[1])
            if cross_prod > 0:
                hull_stack.append(idx)
                i_sorted_polar += 1
            elif cross_prod == 0:
                if not ((pos2[0] == pos1[0]) and (pos2[1] == pos1[1])):
                    dot_prod = (pos1[0] - pos0[0]) * (pos2[0] - pos1[0]) + (pos1[1] - pos0[1]) * (pos2[1] - pos1[1])
                    if dot_prod > 0:
                        hull_stack.append(idx)
                i_sorted_polar += 1
            else:
                hull_stack.pop()
    return hull_stack


def square_3points(pos0, pos1, pos2):
    s = 0.5 * abs(
        float(pos0[0]) * pos1[1] + float(pos0[1]) * pos2[0] + float(pos1[0]) * pos2[1] - float(pos0[0]) * pos2[1] -
        float(pos0[1]) * pos1[0] - float(pos1[1]) * pos2[0]
    )
    return s


def calc_dis(pos0, pos1):
    dis = (float(pos0[0] - pos1[0])**2 + float(pos0[1] - pos1[1])**2)**0.5
    return dis


def rotating_calipers(convex_hull, poses):
    max_dis = -9999
    vertexs = [[-1, -1], [-1, -1]]
    q0 = -9999
    length = len(convex_hull)
    p_idx = 0
    p0 = convex_hull[p_idx]
    p = p0
    pnext = convex_hull[(p_idx + 1) % length]
    q_idx = (p_idx + 1) % length
    q = convex_hull[q_idx]
    qnext = convex_hull[(q_idx + 1) % length]
    qlasts = []
    while True:
        s0 = square_3points(poses[p], poses[pnext], poses[q])
        s1 = square_3points(poses[p], poses[pnext], poses[qnext])
        pos_q = poses[q]
        pos_qnext = poses[qnext]
        while s0 <= s1:
            if s0 == s1:
                qlasts.append(convex_hull[(q_idx - 1) % length])
            else:
                qlasts = [convex_hull[(q_idx - 1) % length]]
            q_idx = (q_idx + 1) % length
            q = convex_hull[q_idx]
            qnext = convex_hull[(q_idx + 1) % length]
            s0 = square_3points(poses[p], poses[pnext], poses[q])
            s1 = square_3points(poses[p], poses[pnext], poses[qnext])
        # if q0 == -9999:
        #     q0 = q
        # if s0==s1:
        if True:
            for qlast in qlasts:
                dis = calc_dis(poses[pnext], poses[qlast])
                if dis > max_dis:
                    max_dis = dis
                    vertexs = [poses[pnext], poses[qlast]]
                dis = calc_dis(poses[p], poses[qlast])
                if dis > max_dis:
                    max_dis = dis
                    vertexs = [poses[p], poses[qlast]]

        dis = calc_dis(poses[pnext], poses[q])
        if dis > max_dis:
            max_dis = dis
            vertexs = [poses[pnext], poses[q]]
        dis = calc_dis(poses[p], poses[q])
        if dis > max_dis:
            max_dis = dis
            vertexs = [poses[p], poses[q]]
        p_idx = (p_idx + 1) % length
        p = convex_hull[p_idx]
        pnext = convex_hull[(p_idx + 1) % length]
        if p == p0:
            break
    return max_dis, vertexs


def contours_to_pos_list(contours):
    pos_list = []
    for i_pos in range(len(contours)):
        pos = [contours[i_pos, 0, 0], contours[i_pos, 0, 1]]
        pos_list.append(pos)
    return pos_list


def get_long_short_axis(input_mask):
    '''

    :param real_max_contour: contour on the mask
    :return:
    '''
    tic = time.time()
    contours, _ = cv2.findContours(input_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        real_max_contour = max(contours, key=cv2.contourArea)
    else:
        real_max_contour = []
    pos_list = contours_to_pos_list(real_max_contour)
    toc = time.time()
    # print("find_contour: " + str(toc - tic))

    tic = time.time()
    if len(pos_list) < 3:
        # print("The number of points on the contour is lesss than three")
        return None
    num = len(pos_list)
    idx = range(num)
    sorted_idx = sorted(idx, key=lambda x: (pos_list[x][1], pos_list[x][0]), reverse=False)
    anchor_idx = sorted_idx.pop(0)
    convex_hull = graham_scan(anchor_idx=anchor_idx, poses=pos_list)
    long_diameter, long_vertexs = rotating_calipers(convex_hull, pos_list)
    try:
        short_diameter, short_diameter_version2 = get_short_vertexs(
            real_max_contour, list([long_vertexs[1], long_vertexs[0]])
        )
        short_pos_1 = list(short_diameter_version2[0])
        short_pos_2 = list(short_diameter_version2[1])
        short_vertexs = [short_pos_1, short_pos_2]
    except Exception:
        print('can not get short diameter')
        short_vertexs = None
    toc = time.time()
    # print("cal_long_short " + str(toc - tic))
    return real_max_contour, long_vertexs, short_vertexs


def get_dice(input_mask, ref_mask):
    '''

    :param input_mask:
    :param output_mask: all of variables must be in the same shape only contain 0 and 1
    :return: dice_score
    '''
    if not input_mask.shape == ref_mask.shape:
        raise NotImplementedError('input shape is different')
    temp_input_mask = np.zeros(shape=input_mask.shape)
    temp_ref_mask = np.zeros(shape=ref_mask.shape)
    temp_input_mask[input_mask > 0] = 1
    temp_ref_mask[ref_mask > 0] = 1
    combine_mask = temp_input_mask + temp_ref_mask
    bottom = np.sum(combine_mask)
    upper = 2 * np.sum(combine_mask == 2)
    dice = upper / float(bottom)
    return dice


def get_tight_box(input_mask):
    '''

    :param input_mask: 0 represent background while 1 represent foreground
    :return:  tight_box which fit mask well
    '''
    tic = time.time()
    input_mask = input_mask.astype(np.uint8)

    contours, _ = cv2.findContours(input_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        real_max_contour = max(contours, key=cv2.contourArea)
    else:
        real_max_contour = []
    pos_list = contours_to_pos_list(real_max_contour)
    pos_array = np.array(pos_list)
    x_min = np.min(pos_array[:, 0], axis=0)
    x_max = np.max(pos_array[:, 0], axis=0)
    y_min = np.min(pos_array[:, 1], axis=0)
    y_max = np.max(pos_array[:, 1], axis=0)
    leftup = (x_min, y_min)
    rightbottom = (x_max, y_max)
    return leftup, rightbottom


if __name__ == '__main__':
    np_array = np.load(
        '/home/tx-deepocean/data/Nodule_Segmentation/all_data/new_organized/mask/AHXK171226985T100/AHXK171226985T100_103.npy'
    ).astype(np.uint8)
    zero_array = np.zeros(np_array.shape)
    zero_array[np_array == 1] = 1

    plt.imshow(zero_array)
    plt.show()

    # dice = get_dice(zero_array, zero_array)
    # leftup, rightbottom = get_tight_box(zero_array)

    real_max_contour, long_vertexs, short_vertexs = get_long_short_axis(zero_array.astype(np.uint8))
    print(long_vertexs)
    print(short_vertexs)
    # len, cor = get_short_vertexs(contour, long_vertexs)
    # show_vertexs((512, 512), [real_max_contour], long_vertexs, short_vertexs, box=(leftup, rightbottom))
    # print(11)
