import math
import random

import numpy as np


class Line:

    def __init__(
        self, point_list, former_joint_point, next_joint_point=None, distance_array=None, is_root=False, fix_length=15
    ):
        self.point_list = point_list #
        self.distance_array = distance_array
        self.set_radius()
        self.color = None
        self.next_joint_point = next_joint_point
        self.fix_length = fix_length
        self.former_joint_point = former_joint_point
        self.get_orient_vector()
        self.is_root = is_root

    def random_cut(self, label_mask):  #防止血管连接出现断开，随机将连续血管切割开，作为训练数据
        plength = len(self.point_list)
        start_point = random.randint(0, max(plength - 3, 2))
        end_point = random.randint(start_point + 1, plength)
        cut_mask = np.zeros_like(label_mask)
        for i in range(start_point, end_point):
            radius = self.radius_list[i]
            p = self.point_list[i]
            offset = int(max(radius / 2, 1)) + 2
            cut_mask[p[0] - offset:p[0] + offset + 1, p[1] - offset:p[1] + offset + 1,
                     p[2] - offset:p[2] + offset + 1] = 1
        cut_mask = cut_mask * label_mask
        after_mask = (cut_mask == 0) * label_mask
        return after_mask, cut_mask

    def set_radius(self):
        self.radius_list = []
        for p in self.point_list:
            tmp_cube = self.distance_array[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2, p[2] - 1:p[2] + 2]
            radius = np.max(tmp_cube)  #distance_array中最大的作为血管的半径
            self.radius_list.append(radius)

    def get_color_according_start(self, mask):
        self.color = mask[self.former_joint_point[0], self.former_joint_point[1], self.former_joint_point[2]]
        return self.color

    def get_color_according_majority(self, mask):
        value_list = []
        for p in self.point_list:
            value_list.append(mask[p[0], p[1], p[2]])
        value_array = np.array(value_list)
        bin_array = np.bincount(value_array)
        if bin_array.shape[0] > 1:
            self.color = np.argmax(bin_array[1:]) + 1
        return self.color

    def get_color_litter_trick(self, mask):
        if len(self.point_list) < 1000000 and self.if_no_next():
            self.get_color_according_start(mask)
        else:
            self.get_color_according_majority(mask)

    def if_paint(self):
        if self.color is None:
            return False
        else:
            return True

    def paint_color_on_mask(self, mask, color=None):
        if color is None:
            color = self.color
        else:
            color = color
        for p in self.point_list:
            mask[p[0], p[1], p[2]] = color

    def set_color(self, color):
        self.color = color

    def paint_color(self, mask, mode='start'):
        if mode == 'start':
            self.color = mask[self.former_joint_point[0], self.former_joint_point[1], self.former_joint_point[2]]

    def if_no_next(self):
        if self.next_joint_point is None:
            return True
        else:
            return False

    def get_orient_vector(self):
        if len(self.point_list) < self.fix_length:
            self.vector = [self.point_list[0][i] - self.point_list[-1][i] for i in range(3)]
        else:
            self.vector = [self.point_list[0][i] - self.point_list[self.fix_length - 1][i] for i in range(3)]

    def get_similarity_for_color(self):
        pass

    def get_start_point(self):
        return self.point_list[0]

    def get_end_point(self):
        return self.point_list[-1]

    def get_next_joint_point(self):
        return self.next_joint_point

    def find_line_by_point(self, point):
        if point in self.point_list:
            return True
        else:
            return False
